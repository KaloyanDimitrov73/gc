import sqlite3
import json
import threading
import time
import hashlib
from typing import Any, Optional

from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


class CacheManager:
    """
    The Cachemanager class is a singleton class that manages a SQLite database
    to allow other classes to cache data between runs.

    The SQLite database is serialized to disk to allow for persistence between 
    runs.

    Learn more about this implementation here:
    https://www.sqlitetutorial.net/sqlite-python/
    """
    _instance = None
    _instance_lock = threading.RLock()
    _is_initialized = False

    def __new__(cls):
        with cls._instance_lock:
            if not cls._instance:
                cls._instance = super(CacheManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._is_initialized:
            self.file_path_manager = FilePathManager()
            self.db_path = self.file_path_manager.get_path(
                "cached_data.sqlite")
            self.file_path_manager.ensure_dir_exists(self.db_path)
            self._is_initialized = True

    def _get_connection(self) -> sqlite3.Connection:
        """
        Create a new SQLite connection.
        """
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30,
                check_same_thread=False
            )
            # WAL improves concurrency by allowing read and write operations to occur
            # simultaneously on the database. Read more here: https://www.sqlite.org/wal.html
            conn.execute("PRAGMA journal_mode = WAL;")
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to SQLite database: {e}")
            raise

    def _create_table_if_not_exists(self, meta_key: str):
        """
        Create a table for the given meta_key if it does not exist.

        Args:
            meta_key: The key to create a table for.
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {meta_key} (
                    dict_key TEXT PRIMARY KEY,
                    value TEXT
                );
            """)
            conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Error creating table '{meta_key}': {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def _execute_with_retry(self, func, *args, **kwargs) -> Any:
        """
        Execute a database operation with retries in case of locks.
        The function will retry the given function as parameter until 
        max retries are reached.

        Args:
            func: The function to execute
            *args: The arguments to pass to the function
            **kwargs: The keyword arguments to pass to the function
        """
        max_retries = 5
        delay = 0.1
        for _ in range(max_retries):
            try:
                return func(*args, **kwargs)
            except sqlite3.OperationalError as e:
                if "locked" in str(e):
                    logger.warning(
                        f"Database is locked. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    # The exponential backoff
                    delay *= 2
                else:
                    logger.error(
                        f"Operational error during database operation: {e}")
                    raise
        logger.error(
            "Max retries exceeded for database operation due to persistent locks.")
        raise RuntimeError(
            "Database is locked, and max retries were exceeded.")

    def add_data(self, meta_key: str, dict_key: str, value: dict, silent: bool = False):
        """
        Add or update data in the cache.

        Args:
            meta_key: The key to store the data under.
            dict_key: The key to store the data under.
            value: The value to store in the cache.
            silent: Whether to suppress log messages.
        """
        def operation():
            self._create_table_if_not_exists(meta_key)
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                json_value = json.dumps(value)
                cursor.execute(f"""
                    INSERT INTO {meta_key} (dict_key, value)
                    VALUES (?, ?)
                    ON CONFLICT(dict_key) DO UPDATE SET value=excluded.value;
                """, (dict_key, json_value))
                conn.commit()
                if not silent:
                    logger.debug(
                        f"""Added/Updated data in cache under meta_key='{meta_key}', 
                        dict_key='{dict_key}'.""")
            except sqlite3.Error as e:
                logger.error(f"Error adding/updating data in cache: {e}")
                raise
            finally:
                cursor.close()
                conn.close()

        self._execute_with_retry(operation)

    def add_data_batch(self, meta_key: str, rows: list[tuple[str, str]]):
        """
        Insert multiple rows into a cache table in a single transaction.
        This is significantly faster than calling add_data() in a loop,
        as it uses a single connection and commit for all rows.

        Args:
            meta_key: The table name to insert the data into.
            rows: A list of (dict_key, value) tuples to insert.
                  Values should already be JSON-serialized strings.
        """
        def operation():
            self._create_table_if_not_exists(meta_key)
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.executemany(f"""
                    INSERT INTO {meta_key} (dict_key, value)
                    VALUES (?, ?)
                    ON CONFLICT(dict_key) DO UPDATE SET value=excluded.value;
                """, rows)
                conn.commit()
                logger.debug(
                    f"Batch inserted {len(rows)} rows into '{meta_key}'.")
            except sqlite3.Error as e:
                logger.error(f"Error batch inserting data into '{meta_key}': {e}")
                raise
            finally:
                cursor.close()
                conn.close()

        self._execute_with_retry(operation)

    def get_data(self, meta_key: str, dict_key: str, silent: bool = False) -> Optional[Any]:
        """
        Retrieve data from the cache under the specified meta_key and dict_key.

        Args:
            meta_key: The key to retrieve the data from.
            dict_key: The key to retrieve the data from.
            silent: Whether to suppress log messages.

        Returns:
            The value stored in the cache under the specified meta_key and dict_key.
            If no data is found, returns None.
        """
        def operation():
            self._create_table_if_not_exists(meta_key)
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    SELECT value FROM {meta_key}
                    WHERE dict_key = ?;
                """, (dict_key,))
                row = cursor.fetchone()
                if row:
                    value = json.loads(row[0])
                    if not silent:
                        logger.debug(
                            f"""Retrieved data from cache under meta_key='{meta_key}', 
                            dict_key='{dict_key}'.""")
                    return value
                if not silent:
                    logger.debug(
                        f"No data found in cache under meta_key='{meta_key}', " +
                        f" dict_key='{dict_key}'.")
                return None
            except sqlite3.Error as e:
                logger.error(f"Error retrieving data from cache: {e}")
                raise
            finally:
                cursor.close()
                conn.close()

        return self._execute_with_retry(operation)

    def get_table(self, meta_key: str) -> Optional[dict]:
        """
        Retrieve all data from a table in the cache.

        Args:
            meta_key: The key of the table to retrieve the data from.

        Returns:
            A dictionary containing all data in the table, where keys are dict_keys
            and values are the corresponding values. If no data is found, returns None.
        """
        def operation():
            self._create_table_if_not_exists(meta_key)
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    SELECT * FROM {meta_key};
                """)
                rows = cursor.fetchall()
                if rows:
                    data = {}
                    for row in rows:
                        data[row[0]] = json.loads(row[1])
                    logger.debug(
                        f"Retrieved data from cache under meta_key='{meta_key}'.")
                    return data
                logger.debug(
                    f"No data found in cache under meta_key='{meta_key}'.")
                return None
            except sqlite3.Error as e:
                logger.error(f"Error retrieving data from cache: {e}")
                raise
            finally:
                cursor.close()
                conn.close()

        return self._execute_with_retry(operation)

    def remove_data(self, meta_key: str, dict_key: str):
        """
        Remove specific data from the cache.

        Args:
            meta_key: The meta key of the table to remove the data from.
            dict_key: The key in the table to remove the data from.
        """
        def operation():
            self._create_table_if_not_exists(meta_key)
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(f"""
                    DELETE FROM {meta_key}
                    WHERE dict_key = ?;
                """, (dict_key,))
                conn.commit()
                logger.debug(
                    f"Removed data from cache under meta_key='{meta_key}', dict_key='{dict_key}'.")
            except sqlite3.Error as e:
                logger.error(f"Error removing data from cache: {e}")
                raise
            finally:
                cursor.close()
                conn.close()

        self._execute_with_retry(operation)

    def delete_table(self, meta_key: str):
        """
        Delete a table from the cache.

        Args:
            meta_key: The meta key of the table to delete.
        """
        def operation():
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {meta_key};")
                conn.commit()
                logger.debug(f"Dropped table '{meta_key}'.")
            except sqlite3.Error as e:
                logger.error(f"Error deleting table '{meta_key}': {e}")
                raise
            finally:
                cursor.close()
                conn.close()

        self._execute_with_retry(operation)

    def clear_cache(self):
        """
        Clear the entire cache by deleting all tables.
        """
        def operation():
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table';
                """)
                tables = cursor.fetchall()
                for (table_name,) in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
                    logger.debug(f"Dropped table '{table_name}'.")
                conn.commit()
                logger.debug("Cleared entire cache by dropping all tables.")
            except sqlite3.Error as e:
                logger.error(f"Error clearing cache: {e}")
                raise
            finally:
                cursor.close()
                conn.close()

        self._execute_with_retry(operation)

    def get_hash_value(self, text: str):
        """
        Generate an MD5 hash value for a given text.

        Args:
            text: The input string to hash.
        """
        return hashlib.md5(text.encode()).hexdigest()

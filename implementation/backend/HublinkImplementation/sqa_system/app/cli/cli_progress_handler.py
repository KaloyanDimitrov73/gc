import threading
import sys
from rich.progress import Progress, TaskID, BarColumn, TextColumn, Task

from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


class ProgressHandler:
    """
    Handler that manages a single, shared rich progress bar across all modules.
    Utilizes a thread-safe singleton pattern to ensure
    only one progress display is active at any given time.
    """
    _instance = None
    _instance_lock = threading.RLock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                logger.debug("Creating a new ProgressHandler instance.")
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        with self._instance_lock:
            if not hasattr(self, '_initialized'):
                self._progress_lock = threading.RLock()
                self._initialized = True
                self.disabled = False
                self._in_notebook = False
                self._progress = None
                self._tasks = {}
                self._saved_state = None

    def add_task(self, string_id: str, description: str, total: int, reset: bool = False) -> str | None:
        """
        Add a new task to the progress bar.

        Args:
            string_id (str): The string ID of the task.
            description (str): The description of the task.
            total (int): The total number of steps in the task.
            reset (bool, optional): Whether to reset the task if it already exists.

        Returns:
            TaskID: The ID of the added or updated task.
        """
        if self.disabled:
            return None
        with self._progress_lock:
            if not self.is_running():
                self._start()
            task_id = self._get_task_id_by_string_id(string_id)
            task = self._get_task_by_id(task_id)
            if not task:
                task_id = self._progress.add_task(
                    description=description,
                    completed=0,
                    start=True,
                    total=total,
                    visible=True
                )
                self._tasks[string_id] = {
                    "task_id": task_id,
                    "description": description,
                    "total": total
                }
                return string_id

            # If the task already exists, check if it should be reset
            if reset:

                self._progress.update(
                    task_id=task_id,
                    description=description,
                    total=total)
                self._progress.stop_task(task_id)
                self._progress.start_task(task_id)
                return string_id

            # Else it exists and we need to update the tasks length
            current_total_of_task = 0
            if task is not None:
                current_total_of_task = task.total
            new_total = current_total_of_task + total
            self.update_task_length(string_id, new_total)
            return string_id

    def update_task_by_string_id(self, string_id: str, advance: int = 1, remove_if_completed: bool = True):
        """
        Update the progress of a task by its string ID.

        Args:
            string_id (str): The string ID of the task to update.
            advance (int, optional): The amount to advance the task. Defaults to 1.
        """
        if self.disabled:
            return
        with self._progress_lock:
            task_id = self._get_task_id_by_string_id(string_id)
            if task_id is None:
                logger.debug(
                    f"Task with string ID '{string_id}' not found.")
                return
            if not self._is_task_running(task_id):
                self.add_task(string_id,
                              self._tasks[string_id]["description"],
                              self._tasks[string_id]["total"])
                return
            self._update_task(
                task_id=task_id,
                advance=advance,
                remove_if_completed=remove_if_completed
            )

    def is_running(self) -> bool:
        """
        Returns whether the progress bar is currently running
        """
        if self.disabled:
            return False
        if not self._progress:
            return False
        return True

    def update_task_length(self, string_id: str, total: int):
        """
        Update the total length of a task.

        Args:
            string_id (str): The string ID of the task.
            total (int): The new total length of the task.
        """
        if self.disabled:
            return
        task_id = self._get_task_id_by_string_id(string_id)
        if task_id is not None and self._is_task_running(task_id):
            self._update_task_length(task_id, total)
        elif task_id is not None:
            task_data = self._tasks.get(string_id)
            if task_data is not None:
                description = task_data["description"]
                self.add_task(string_id, description, total, reset=True)
        else:
            raise RuntimeError(
                f"Task with string ID '{string_id}' not found.")

    def finish_by_string_id(self, string_id: str):
        """
        Mark a task as completed by setting its progress to its total value.

        Args:
            string_id (str): The string ID of the task to finish.
        """
        if self.disabled:
            return
        with self._progress_lock:
            data = self._tasks.get(string_id)
            task_id = data["task_id"] if data else None
            task = self._get_task_by_id(task_id)
            if task is not None:
                self._remove_task(task_id)

    def disable(self):
        """
        Disables the progress handler and stores the current
        state.
        """ 
        with self._progress_lock:
            saved_tasks = {}
            for string_id, task_data in self._tasks.items():
                task_id = task_data["task_id"]
                task = self._get_task_by_id(task_id)

                if task:
                    saved_tasks[string_id] = {
                        "task_id": task_id,
                        "description": task_data["description"],
                        "total": task.total,
                        "completed": task.completed,
                        "finished": task.finished
                    }
                else:
                    saved_tasks[string_id] = {
                        "task_id": task_id,
                        "description": task_data["description"],
                        "total": task_data["total"],
                        "completed": 0,
                        "finished": True
                    }

            self._saved_state = {
                "tasks": saved_tasks,
                "was_running": self.is_running()
            }

            self._stop()
            self.disabled = True
            
    def clear(self):
        """
        Clears all tasks from the progress bar.
        """
        if self.disabled:
            return
        with self._progress_lock:
            if not self.is_running():
                return
            for task_id in list(self._tasks.values()):
                self._remove_task(task_id["task_id"])
            self._tasks = {}
            self._stop()

    def enable(self):
        """
        Enables the progress handler and restores the sate before it
        was disabled
        """
        self.disabled = False

        if self._saved_state is None:
            return

        with self._progress_lock:
            if self._saved_state["was_running"]:
                self._start()

                # Restore all tasks with their previous state
                for string_id, task_data in self._saved_state["tasks"].items():
                    description = task_data["description"]
                    total = task_data["total"]

                    if not task_data["finished"]:
                        task_id = self._progress.add_task(
                            description=description,
                            start=True,
                            total=total,
                            visible=True,
                            completed=task_data["completed"],
                        )
                    else:
                        task_id = task_data["task_id"]
                        
                    self._tasks[string_id] = {
                        "task_id": task_id,
                        "description": description,
                        "total": total,
                    }

            self._saved_state = None

    def _initialize_progress(self):
        """
        Initializes the progress handler.
        """
        if self.disabled:
            return
        logger.debug("Initializing ProgressHandler.")
        self._in_notebook = 'ipykernel' in sys.modules
        self._progress = Progress()
        progress_columns = [
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}")
        ]

        # Use a specialized console/refresh rate when in Jupyter
        if self._in_notebook:
            self._progress = Progress(
                *progress_columns,
                auto_refresh=True,
                refresh_per_second=2,
                transient=True
            )
        else:
            self._progress = Progress(
                *progress_columns,
                transient=True
            )

    def _start(self):
        if self.disabled:
            return
        with self._progress_lock:
            if not self.is_running():
                logger.debug("Starting Progress.")
                self._initialize_progress()
                self._progress.start()

    def _stop(self):
        with self._progress_lock:
            try:
                if self.is_running():
                    logger.debug("Stopping Progress.")
                    self._progress.stop()
                    self._progress = None
                    self._tasks = {}
            except Exception as e:
                logger.error(f"Error while stopping Progress: {e}")

    def _get_task_id_by_string_id(self, string_id: str) -> TaskID | None:
        """
        Get the task ID by its string ID.

        Args:
            string_id (str): The string ID of the task.
        """
        with self._progress_lock:
            for str_id, task in self._tasks.items():
                if str_id == string_id:
                    return task["task_id"]
            return None

    def _get_string_id_by_task_id(self, task_id: TaskID) -> str | None:
        """
        Get the string ID by its task ID.

        Args:
            task_id (TaskID): The ID of the task.
        """
        with self._progress_lock:
            for string_id, data in self._tasks.items():
                if data["task_id"] == task_id:
                    return string_id
            return None

    def _update_task_length(self, task_id: TaskID, total: int):
        with self._progress_lock:
            # Check if the task is running
            if not self._is_task_running(task_id):
                return
            self._progress.update(task_id, total=total, visible=True)

    def _get_task_by_id(self, task_id: TaskID) -> Task | None:
        if task_id is None:
            return None
        for task in self._progress.tasks:
            if task.id == task_id:
                return task
        return None

    def _update_task(self, task_id: TaskID, advance: int = 1, remove_if_completed: bool = True):
        """
        Update the progress of a task.

        Args:
            task_id (TaskID): The ID of the task to update.
            advance (int, optional): The amount to advance the task. Defaults to 1.
        """
        if self.disabled:
            return
        with self._progress_lock:
            if not self.is_running():
                raise RuntimeError("Progress is not running.")
            self._progress.update(task_id, advance=advance, visible=True)

            # Remove the task if it is completed
            if task_id is not None and remove_if_completed:
                task = self._get_task_by_id(task_id)
                if task is not None and task.finished:
                    self._remove_task(task_id)

                self._check_if_should_stop()

    def _check_if_should_stop(self):
        """
        Checks whether the progress should stop and stops if all tasks are completed.
        """
        if not self.is_running():
            return

        if self._progress.finished:
            self._stop()

    def _remove_task(self, task_id: TaskID):
        """
        Removes a task from the progress bar.

        Args:
            task_id (TaskID): The ID of the task to remove.
        """
        self._progress.remove_task(task_id)

    def _is_task_running(self, task_id: TaskID) -> bool:
        task = self._get_task_by_id(task_id)
        if task is None:
            return False
        return not task.finished

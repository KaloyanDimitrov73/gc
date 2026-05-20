"""
Async SQLAlchemy engine and session factory for SQLite persistence.
"""
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from .models import Base

# Database file lives in backend/data/ — created automatically on startup
# parents[5] resolves to the backend/ package root from this file's location
_DB_DIR = Path(__file__).resolve().parents[5] / "data"
_DB_PATH = _DB_DIR / "hublink.db"
DATABASE_URL = f"sqlite+aiosqlite:///{_DB_PATH}"

engine = create_async_engine(DATABASE_URL, connect_args={"check_same_thread": False})
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def init_db() -> None:
    """Create all tables on first startup. Safe to call repeatedly (idempotent)."""
    _DB_DIR.mkdir(parents=True, exist_ok=True)
    async with engine.begin() as conn:
        await conn.run_sync(lambda c: Base.metadata.create_all(c, checkfirst=True))
        # Migrate: add user_id column if it does not exist yet
        try:
            await conn.exec_driver_sql(
                "ALTER TABLE conversations ADD COLUMN user_id VARCHAR(36)"
            )
        except Exception:
            pass  # Column already exists


async def get_db():
    """FastAPI dependency: yields a per-request AsyncSession."""
    async with AsyncSessionLocal() as session:
        yield session

import os
import sys
import sqlite3
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import settings


def is_network_filesystem(path: str) -> bool:
    """Detect if a path is on a remote/network filesystem (UNC share or mapped drive).

    WAL mode requires shared memory primitives (shm files) that are unsupported
    or unreliable on network filesystems (NFS, SMB/CIFS, UNC paths).
    """
    if not path or path == ":memory:":
        return False
    # Check UNC network paths (e.g. \\server\share or //server/share)
    if path.startswith("\\\\") or path.startswith("//"):
        return True
    if sys.platform == "win32":
        try:
            import ctypes
            drive = os.path.splitdrive(os.path.abspath(path))[0]
            if drive:
                drive_root = drive + "\\" if not drive.endswith("\\") else drive
                # DRIVE_REMOTE = 4 in Windows API
                if ctypes.windll.kernel32.GetDriveTypeW(drive_root) == 4:
                    return True
        except Exception:
            pass
    return False


# For SQLite, check_same_thread must be False for multithreaded requests,
# and timeout is configured to 30.0s for busy-wait resilience.
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False
    connect_args["timeout"] = 30.0

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable foreign key enforcement, busy timeout, WAL mode, and normal synchronous on SQLite."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        old_isolation = dbapi_connection.isolation_level
        try:
            # Set isolation level to None (autocommit) to ensure PRAGMAs execute outside any active transaction
            dbapi_connection.isolation_level = None
            cursor = dbapi_connection.cursor()

            # Enforce foreign key constraints
            cursor.execute("PRAGMA foreign_keys=ON;")

            # 30-second busy timeout for database-lock contention resilience
            cursor.execute("PRAGMA busy_timeout=30000;")

            # Inspect connected database file to check if it is a local filesystem database
            db_file = None
            cursor.execute("PRAGMA database_list;")
            for row in cursor.fetchall():
                if row[1] == "main":
                    db_file = row[2]
                    break

            # Apply WAL mode and synchronous=NORMAL only to local file-based databases
            if db_file and not is_network_filesystem(db_file):
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")

            cursor.close()
        finally:
            dbapi_connection.isolation_level = old_isolation


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency that yields a database session for a request and cleans it up."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


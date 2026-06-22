from __future__ import annotations
import sqlite3
import os
from typing import Optional

# Default database path, can be overridden by DB_PATH environment variable
_DEFAULT_DB_PATH = 'state/hepo.db'
DB_PATH: str = os.getenv('DB_PATH', _DEFAULT_DB_PATH)

def get_db_connection(db_file: Optional[str] = None) -> sqlite3.Connection:
    """Establish and return a connection to the SQLite database."""
    path = db_file if db_file else DB_PATH
    return sqlite3.connect(path)

def db_exists(db_file: Optional[str] = None) -> bool:
    """Check if the SQLite database file exists."""
    path = db_file if db_file else DB_PATH
    return os.path.exists(path)

def get_db_path() -> str:
    """Return the default database path (respects DB_PATH environment variable)."""
    return DB_PATH

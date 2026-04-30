import sqlite3
import os

DB_PATH = 'hepo.db'

def get_db_connection(db_file=None):
    """Establish and return a connection to the SQLite database."""
    path = db_file if db_file else DB_PATH
    return sqlite3.connect(path)

def db_exists(db_file=None):
    """Check if the SQLite database file exists."""
    path = db_file if db_file else DB_PATH
    return os.path.exists(path)

def get_db_path():
    """Return the default database path."""
    return DB_PATH

"""
Migration: rename battery_action 'idle' → 'follow'.

Historical 'idle' entries now map to the renamed 'follow' action (load-following
behaviour). The new true 'idle' action means the battery does nothing at all.
"""

import sqlite3
import os


def migrate():
    db_path = os.getenv('HEPO_DB_PATH', 'hepo.db')
    if not os.path.exists(db_path):
        print(f"DB not found at {db_path}, skipping migration.")
        return

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    for table in ('predictions', 'performance_analysis'):
        cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
        if not cur.fetchone():
            continue
        cur.execute(f"UPDATE {table} SET battery_action='follow' WHERE battery_action='idle'")
        print(f"Migrated {cur.rowcount} rows in '{table}'.")

    conn.commit()
    conn.close()
    print("Migration complete.")


if __name__ == '__main__':
    migrate()

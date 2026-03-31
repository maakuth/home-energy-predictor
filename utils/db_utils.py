import os
import pandas as pd
import psycopg2
from datetime import datetime, timedelta

def get_db_connection():
    """Establish and return a connection to the PostgreSQL database."""
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10
    )

def fetch_states_history(entity_ids, hours=1, start_time=None):
    """
    Fetch history for multiple entities from PostgreSQL.
    Returns a dictionary of DataFrames, keyed by entity_id.
    """
    if isinstance(entity_ids, str):
        entity_ids = [entity_ids]
    
    if not entity_ids:
        return {}

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # 1. Get metadata_ids for all requested entities
        cur.execute("SELECT metadata_id, entity_id FROM states_meta WHERE entity_id IN %s", (tuple(entity_ids),))
        meta = {row[0]: row[1] for row in cur.fetchall()}
        
        if not meta:
            cur.close()
            conn.close()
            return {eid: pd.DataFrame() for eid in entity_ids}

        # 2. Determine start timestamp
        if start_time:
            start_ts = start_time.timestamp()
        else:
            start_ts = (datetime.now() - timedelta(hours=hours)).timestamp()

        # 3. Fetch states for all relevant metadata_ids in one query if possible, 
        # or iterate if that's cleaner for mapping. Let's do one query.
        metadata_ids = tuple(meta.keys())
        query = "SELECT metadata_id, last_updated_ts, state FROM states WHERE metadata_id IN %s AND last_updated_ts > %s"
        cur.execute(query, (metadata_ids, start_ts))
        rows = cur.fetchall()
        
        cur.close()
        conn.close()

        # 4. Group rows by entity
        data_by_entity = {eid: [] for eid in entity_ids}
        for mid, ts, state in rows:
            eid = meta.get(mid)
            if eid:
                data_by_entity[eid].append({'ts': ts, 'state': state})

        # 5. Convert to DataFrames
        results = {}
        for eid, entries in data_by_entity.items():
            if not entries:
                results[eid] = pd.DataFrame()
                continue
            
            df = pd.DataFrame(entries)
            df['timestamp'] = pd.to_datetime(df['ts'], unit='s', utc=True)
            df['state'] = pd.to_numeric(df['state'], errors='coerce')
            results[eid] = df.dropna().sort_values('timestamp').drop(columns=['ts'])

        return results

    except Exception as e:
        print(f"⚠️ Error fetching history from PostgreSQL: {e}")
        return {eid: pd.DataFrame() for eid in entity_ids}

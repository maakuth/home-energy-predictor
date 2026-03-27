import os
import psycopg2
from dotenv import load_dotenv

load_dotenv(override=True)

def connect():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        database=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )

def explore():
    conn = connect()
    cur = conn.cursor()
    
    # List tables
    cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name")
    tables = [t[0] for t in cur.fetchall()]
    print(f"Tables: {', '.join(tables)}")
    
    # Check states and states_meta tables if they exist
    potential_tables = ['states', 'states_meta', 'recorder_runs', 'schema_changes']
    for table in potential_tables:
        if table in tables:
            print(f"\nPeek into '{table}':")
            cur.execute(f"SELECT * FROM {table} LIMIT 2")
            colnames = [desc[0] for desc in cur.description]
            print(f"Columns: {colnames}")
            rows = cur.fetchall()
            for row in rows:
                print(row)
    
    # Specifically look for how to query entities
    if 'states_meta' in tables:
        print("\nSearching for relevant entities in 'states_meta'...")
        entities = [
            'sensor.outside_temperature',
            'sensor.mlp_teho',
            'sensor.saikaan_olohuone_current_power',
            'sensor.mokkimokin_ilp_power',
            'sensor.mummun_energy',
            'sensor.mlp_varaajan_lampotila',
            'sensor.xpz_491_battery_level',
            'device_tracker.xpz_491_position',
            'sensor.sahkokauppa_nyt',
            'sensor.solcast_pv_forecast_forecast_tomorrow'
        ]
        query = "SELECT metadata_id, entity_id FROM states_meta WHERE entity_id IN %s"
        cur.execute(query, (tuple(entities),))
        found = set()
        for metadata_id, entity_id in cur.fetchall():
            print(f"Found: {entity_id} (metadata_id: {metadata_id})")
            found.add(entity_id)
        
        missing = [e for e in entities if e not in found]
        if missing:
            print("\nSearching for missing entities using LIKE...")
            for e in missing:
                search_term = f"%{e.split('.')[-1]}%"
                cur.execute("SELECT metadata_id, entity_id FROM states_meta WHERE entity_id LIKE %s", (search_term,))
                results = cur.fetchall()
                if results:
                    print(f"Potential matches for {e}:")
                    for mid, eid in results:
                        print(f"  - {eid} (metadata_id: {mid})")
                else:
                    print(f"No matches for {e}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    explore()

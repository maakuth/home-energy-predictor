#!/usr/bin/env python3
"""
Find Home Assistant entities for battery planning test data.

This script queries your Home Assistant instance and shows you which entities
exist and what their current values are. Use this to customize dump_battery_data.py
for your specific setup.

Usage:
    python find_ha_entities.py                  # List all entities
    python find_ha_entities.py --search battery # Find entities matching "battery"
    python find_ha_entities.py --search sensor  # Find all sensor entities
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from utils.ha_utils import get_ha_state

def find_all_entities():
    """Try to list all entities from Home Assistant."""
    from utils.ha_utils import HA_HOST, HEADERS
    import requests
    
    if not HA_HOST:
        print("❌ HA_HOST not configured in .env")
        return []
    
    try:
        url = f'{HA_HOST}/api/states'
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            states = response.json()
            return states
        else:
            print(f"❌ Error fetching states: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return []


def main():
    load_dotenv(override=True)
    
    parser = argparse.ArgumentParser(
        description='Find Home Assistant entities for battery test data dumper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all entities
  python find_ha_entities.py
  
  # Find battery-related entities
  python find_ha_entities.py --search battery
  
  # Find sensors
  python find_ha_entities.py --search sensor
  
  # Show more detail
  python find_ha_entities.py --search solar --detail
        """
    )
    
    parser.add_argument('--search', type=str,
                        help='Search for entities containing this string (case-insensitive)')
    parser.add_argument('--detail', '-d', action='store_true',
                        help='Show detailed state information')
    
    args = parser.parse_args()
    
    print("🔍 Fetching entities from Home Assistant...\n")
    
    entities = find_all_entities()
    
    if not entities:
        print("❌ Could not fetch entities. Check HA_HOST and HA_TOKEN in .env")
        sys.exit(1)
    
    # Filter if search term provided
    if args.search:
        search_lower = args.search.lower()
        entities = [e for e in entities if search_lower in e['entity_id'].lower()]
        print(f"Found {len(entities)} entities matching '{args.search}':\n")
    else:
        print(f"Found {len(entities)} total entities.\n")
        print("Battery-related entities:")
        print("-" * 80)
    
    # Categorize by domain
    domains = {}
    for entity in entities:
        entity_id = entity['entity_id']
        domain = entity_id.split('.')[0]
        
        if domain not in domains:
            domains[domain] = []
        domains[domain].append(entity)
    
    # Show interesting entities
    if not args.search:
        interesting_domains = ['sensor', 'switch', 'input_boolean', 'binary_sensor']
        for domain in interesting_domains:
            if domain in domains:
                for entity in domains[domain]:
                    eid = entity['entity_id']
                    state = entity.get('state', 'unknown')
                    
                    # Highlight potentially relevant ones
                    keywords = ['battery', 'solar', 'grid', 'power', 'import', 'export', 
                               'load', 'temp', 'price', 'gshp', 'sauna']
                    is_relevant = any(kw in eid.lower() for kw in keywords)
                    
                    prefix = "📍" if is_relevant else "  "
                    print(f"{prefix} {eid:50s} = {state}")
    else:
        # Show search results
        for domain in sorted(domains.keys()):
            if domain in domains and entities and any(
                args.search.lower() in e['entity_id'].lower() for e in domains[domain]
            ):
                print(f"\n{domain.upper()}:")
                print("-" * 80)
                
                for entity in domains[domain]:
                    if args.search.lower() in entity['entity_id'].lower():
                        eid = entity['entity_id']
                        state = entity.get('state', 'unknown')
                        
                        print(f"  {eid:50s} = {state}")
                        
                        if args.detail:
                            attrs = entity.get('attributes', {})
                            for key, val in list(attrs.items())[:5]:  # Show first 5 attributes
                                print(f"      {key}: {val}")
    
    # Show example configuration
    print("\n" + "=" * 80)
    print("💡 TO CUSTOMIZE dump_battery_data.py FOR YOUR SETUP:")
    print("=" * 80)
    
    # Find relevant entities
    relevant = []
    for entity in entities:
        eid = entity['entity_id']
        keywords = ['battery', 'solar', 'grid', 'power', 'import', 'export', 
                   'load', 'temp', 'price', 'gshp', 'sauna']
        if any(kw in eid.lower() for kw in keywords):
            relevant.append(eid)
    
    if relevant:
        print("\nFound these relevant entities on your system:\n")
        print("Option 1: Add to .env file:")
        print("-" * 80)
        entity_list = ','.join(relevant)
        print(f"BATTERY_TEST_HA_ENTITIES={entity_list}\n")
        
        print("Option 2: Edit dump_battery_data.py get_ha_relevant_entities() function")
        print("         and replace the default list with:")
        print("-" * 80)
        print("return [")
        for eid in relevant:
            print(f"    '{eid}',")
        print("]")
    else:
        print("\n⚠️ No obviously relevant entities found.")
        print("You may need to manually identify which entities to monitor.")
        print("\nCommon entity patterns:")
        print("  - sensor.battery_* or sensor.battery_storage_*")
        print("  - sensor.solar_* or sensor.pv_*")
        print("  - sensor.*_power_* or sensor.*_energy_*")
        print("  - sensor.grid_* or sensor.*import* or sensor.*export*")


if __name__ == '__main__':
    main()

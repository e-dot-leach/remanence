import pandas as pd
import sys
import json

def convert_cia_to_lookup():
    with open('cia_journal_links.json') as f:
        data = json.load(f)

    # Group by linked_journal_id
    lookup = {}
    for entry in data:
        key = entry['linked_journal_id']
        if key not in lookup:
            lookup[key] = []
        lookup[key].append(entry)

    with open('cia_journal_links.json', 'w') as f:
        json.dump(lookup, f, indent=2)

def base_convert(name):
    df = pd.read_csv(name)
    df.to_json(name.replace('.csv','.json'), orient='records', indent=2)

if __name__ == "__main__":
    base_convert(sys.argv[1])
    #convert_cia_to_lookup()
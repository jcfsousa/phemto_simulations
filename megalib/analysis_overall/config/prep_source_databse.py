#!/usr/bin/env python3
import json
import numpy as np
import os

def append_to_sources_database(new_sources, filename='sources_database.json'):
    """Append new sources to existing sources_database.json"""
    
    
    # Load existing database (create if doesn't exist)
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            sources_db = json.load(f)
        print(f"Loaded existing database: {len(sources_db)} sources")
    else:
        sources_db = {}
        print("Created new sources_database.json")
    
    # Append new sources (skip duplicates)
    added_count = 0
    for source_name, data in new_sources.items():
        if source_name not in sources_db:
            sources_db[source_name] = data
            added_count += 1
        else:
            print(f"Skipping duplicate: {source_name}")
    
    # Write back updated database
    with open(filename, 'w') as f:
        json.dump(sources_db, f, indent=2)
    
    print(f"Added {added_count} new sources")
    print(f"Total sources: {len(sources_db)}")
    return sources_db

if __name__ == '__main__':

    CdTe_matrix_size = [4, 5, 6, 7]
    dists = np.arange(0.5, 10.5, 1)
    Log_E = [50, 100, 200, 300, 400, 500, 600, 700]
    source_basename = 'CollimatedBeamPol'
    
    new_sources = {}
    for matrix_size in CdTe_matrix_size:
        for dist in dists:
            for energy in Log_E:
                config = f"config{matrix_size}x{matrix_size}_{dist}cm"
                source_name = f"{source_basename}%dkeV_{config}" % energy
                new_sources[source_name] = {"e0": [energy]}


    append_to_sources_database(new_sources)
    
    # OR add specific sources manually:
    # custom_sources = {
    #     "MyNewSource_100keV": {"e0": [100]},
    #     "TestBeam_500keV": {"e0": [500]}
    # }
    # append_to_sources_database(custom_sources)


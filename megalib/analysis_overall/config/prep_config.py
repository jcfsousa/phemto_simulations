#!/usr/bin/env python3
import json
import numpy as np
import re
import os

def generate_config_json():
    # Template
    config_template = {
        "config_chips": "/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/config/chip_config.txt",
        "input_dir": "/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/input_polarimetry",
        "sources": [],
        "sources_peaks": {},
        "output_folder": "/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/output_polarimetry",
        "abct_folder": "/home/josesousa/Documents/thor/detector/detSoftware/calibration/pp-calib/",
        "calib_folder": "/media/josesousa/joseHard/quad_characterization/1-QuadCharacterizationResults/calib_grenoble",
        "res_folder": "/media/josesousa/joseHard/quad_characterization/1-QuadCharacterizationResults/resolution_grenoble",
        "scratch_folder": "/media/josesousa/joseScratch/quad_characterization"
    }
    
    # Generate all source_names matching your pattern
    CdTe_matrix_size = [4, 5, 6, 7]
    dists = np.arange(0.5, 10.5, 1)
    Log_E = [50, 100, 200, 300, 400, 500, 600, 700]
    source_basename = 'GaussBeamPol'
    
    sources = []
    sources_peaks = {}
    
    for matrix_size in CdTe_matrix_size:
        for dist in dists:
            for energy in Log_E:
                config = f"config{matrix_size}x{matrix_size}_{dist}cm"
                source_name = f"{source_basename}%dkeV_{config}" % energy  # Matches your format
                sources.append(source_name)
                sources_peaks[source_name] = [energy]
    
    config_template["sources"] = sources
    config_template["sources_peaks"] = sources_peaks
    
    # Write JSON
    with open("config_gaussian.json", "w") as f:
        json.dump(config_template, f, indent=2)
    
    print(f"Generated config_gaussian.json with {len(sources)} sources")
    print(f"First few sources: {sources[:3]}")
    print("File ready for your polarimetry analysis!")

if __name__ == '__main__':
    generate_config_json()

import subprocess
import manalysis.specLib as specLib
import manalysis.configlib as configlib

import os
import glob


if __name__ == '__main__':
    current_dir = os.getcwd()
    
    parent_dir = os.path.dirname(current_dir)

    selected_config = configlib.initialconfig(parent_dir)
    print(selected_config)
    
    specLib.global_config = specLib.Config(selected_config)              


    sources = specLib.global_config.sources
    chip_dict = specLib.global_config.chip_dict
    
    print('Chip configuration loaded:')
    print(chip_dict)


    for source in sources:
        specLib.pre_process_source(source)
        specLib.process_event_multiplicity(source)

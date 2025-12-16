import os
import glob
import json
import pandas as pd
import sys



def initialconfig(base_dir):
    
    config_files = glob.glob(os.path.join(base_dir, "config", "*.json"))
    

    if not config_files:
        raise FileNotFoundError(f"No config files found in {config_dir} directory")
    
    # Present options to user
    print("Available configuration files:")
    for i, config_file in enumerate(config_files, 1):
        print(f"{i}. {os.path.basename(config_file)}")
    
    # Get user selection
    while True:
        try:
            selection = int(input("Select a config file (number): "))
            if 1 <= selection <= len(config_files):
                selected_config = config_files[selection-1]
                break
            else:
                print(f"Please enter a number between 1 and {len(config_files)}")
        except ValueError:
            print("Please enter a valid number")
    
    print(f"\nUsing config file: {selected_config}")

    return selected_config


def query_user_chips():

    user_chips = input("Which chips do you want to analyze? Write the numbers of the chips separated by a comma (e.g., '1,2,4' or 'all'). \n Available chips: 0,1,2,3 \n Input: ")

# Remove whitespace and split into a list
    if user_chips.strip().lower() == "all":
        chips_to_analyze = [0,1,2,3]  # or process all chips
    else:
        chips_to_analyze = [int(chip.strip()) for chip in user_chips.split(",")]
    
    for chip in chips_to_analyze:
        if chip != 0 and chip != 1 and chip != 2 and chip != 3:
            print(chip)
            print("Please provide valid input")
            chips_to_analyze = query_user_chips()

        return chips_to_analyze
    return chips_to_analyze


global_config = None

class Config:
    def __init__(self, config_file):

        with open(config_file, 'r') as file:
            config_data = json.load(file)
            file.close()

        self.config_chips = config_data["config_chips"]
        self.sources = config_data["sources"]
        self.input_dir = config_data["input_dir"]
        self.output_folder = config_data["output_folder"]
        self.abct_folder = config_data["abct_folder"]
        self.scratch_folder = config_data["scratch_folder"]
        self.calib_folder = config_data["calib_folder"]
        self.res_folder = config_data["res_folder"]
        self.calib_dict = {}
        self.chip_dict = {}
        self.res_dict = {}
        
        source_database = f"{os.path.dirname(config_file)}/sources_database.json"
        with open(source_database, "r") as file:
            source_db = json.load(file)
            file.close()
        self.sources_peaks = source_db

        try:
            with open(self.config_chips, 'r') as file:
                chip_config = file.read().splitlines()
                for line in chip_config:
                    split_line = line.split('=')
                   
                    chip_id = split_line[0]
                    chip = split_line[1]
                    self.chip_dict[int(chip_id)] = chip
        except Exception as e:
            print(f"\033[31m Critical ERROR, chip config did not load. ERROR: {e}\033[0m")
            sys.exit(1)
        
        try:
            for chip_id, chip in self.chip_dict.items():
                calib_file = f'{self.calib_folder}/{chip}/calibCurve_{chip}_singles.csv'
                
                calib_consts = pd.DataFrame()                                                               
                calib_consts = pd.read_csv(calib_file)                                                      

                a = float(calib_consts.loc[calib_consts['Parameter'] == 'a', 'Value'].values[0])
                b = float(calib_consts.loc[calib_consts['Parameter'] == 'b', 'Value'].values[0])

                self.calib_dict[int(chip_id)] = (a,b)
        except Exception as e:
            print(f"\033[33m WARNING: Calibration not completly loaded, check {e} \033[0m")

        try:
            with open(self.config_chips, 'r') as file:
                chip_config = file.read().splitlines()
                for line in chip_config:
                    split_line = line.split('=')
                   
                    chip_id = split_line[0]
                    chip = split_line[1]

                    calib_file = f'{self.res_folder}/{chip}/resolutionCurve_{chip}_singles.csv'
                    
                    calib_consts = pd.DataFrame()                                                               
                    calib_consts = pd.read_csv(calib_file)                                                      

                    a = float(calib_consts.loc[calib_consts['Parameter'] == 'a', 'Value'].values[0])
                    b = float(calib_consts.loc[calib_consts['Parameter'] == 'b', 'Value'].values[0])
                    c = float(calib_consts.loc[calib_consts['Parameter'] == 'c', 'Value'].values[0])

                    self.res_dict[int(chip_id)] = (a,b,c)
        except Exception as e:
            print(f"\033[33m WARNING: Resolution not completly loaded, check {e} \033[0m")
            
    def __str__(self):
        return json.dumps({
            "config_chips": self.config_chips,
            "sources": self.sources,
            "sources_peaks": self.sources_peaks,
            "input_dir": self.input_dir,
            "output_folder": self.output_folder,
            "abct_folder": self.abct_folder,
            "calib_folder": self.calib_folder,
            "scratch_folder": self.scratch_folder,
            "calib_dict": self.calib_dict
        }, indent=4)

import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
import toml
import manalysis.specLib as specLib

# Calibration constants
#CHIPS = ['D03-W0060', 'D04-W0060', 'F04-W0060', 'K10-W0060']
CONSTANTS = ['a', 'b', 'c', 't']
CHIPS = ['D03-W0060', 'D04-W0060', 'F04-W0060', 'K10-W0060', 'constant'] # adicionei (mariana), constant is the minipix

DET_SIDE = 256

class Calibration:

    def __init__(self, outputFolder, _):
        self.outputFolder = outputFolder
	
    def set_up(self, file_path, *args):
		# Define the path to the config file
        #config_file_path = os.path.join(file_path, 'config.toml')

		# Check if the config.toml file exists, if not, terminate the program with an error message
        #if not os.path.exists(config_file_path):
        #    print(f"Error: The configuration file 'config.toml' does not exist at the specified path: {file_path}")
        #    sys.exit(1)  # Exit the program with a non-zero status indicating an error

		# Define the results directory and CSV path
        results_folder = os.path.join(self.outputFolder, 'results')
        os.makedirs(results_folder, exist_ok=True)  # Check if 'results' folder exists, create if it does not       	
        outputFolder_csv = os.path.join(results_folder, 'csv')
        os.makedirs(outputFolder_csv, exist_ok=True)            	
        # Load the TOML file
        #config = toml.load(config_file_path)

		# Access the manalysis section
        #manalysis = config.get('manalysis', {})

		# Create a list to store the results
        results = []

        # Define the keys you want to retrieve based on args
        #for key in args:
            # Attempt to get the value from the manalysis section
        #    value = manalysis.get(key, None)
        #    if value is not None:
        #        results.append(value) 

        # Convert the list to a tuple before returning
        return tuple(results)
	
	
    def load_calibration_constants(self, abct_folder): # adicionei (mariana)
		#if chip not in CHIPS:
		#    raise ValueError(f"Invalid chip: {chip}. Must be one of {CHIPS}.")
        calibration_data = {}

        for chip in CHIPS:
            calibration_data[chip] = {}
            try:
                for constant in CONSTANTS:
                        calibration_data[chip][constant] = np.loadtxt(f"{abct_folder}/{chip}_{constant}.txt", unpack=True)
            except Exception as e:
                print(f"WARNING: chip {chip} not in the config file. {e}")
                continue

        return calibration_data
	
    def get_calibration_constants(self, calibration_data, x, y, chip): # adicionei (mariana)
        return [calibration_data[chip][constant][x,y] for constant in CONSTANTS]

    def determine_chip(self, chip_num):
        """
    	Cluster data points for each event and assign globally unique cluster IDs.
    
    	Parameters:
    	- data (pd.DataFrame): DataFrame containing the data to be clustered.
    	- global_cluster_id (int): Initial global cluster ID to start with.

    	Returns:
    	- cluster_dict (dict): A dictionary mapping index to global cluster ID.
    	- global_cluster_id (int): Updated global cluster ID after clustering.
        """	
        # do not remove the comments bellow, they are used by the bash script to change the detector config

        # CHIP CONFIGURATION START 

        #chip_config = {
        #        0: "D04-W0060",
        #        1: "D03-W0060",
        #        2: "K10-W0060",
        #        3: "F04-W0060"
        #}
        # CHIP CONFIGURATION END

        chip_config = specLib.global_config.chip_dict

        if chip_num in chip_config:
            return chip_config[chip_num]
        else:
            raise ValueError(f'Invalid chip number: {chip_num}')
        ## Lab nov2024 config
		#if chip_num == 0:
		#	return "F04-W0060"
		#elif chip_num == 1:
		#	return "K10-W0060"
		#elif chip_num == 2:
		#	return "D03-W0060"
		#elif chip_num == 3:
		#	return "D04-W0060"
		#else:
		#	raise ValueError(f"Invalid chip number: {chip_num}")
        ## Grenoble standard Config
     
		#if chip_num == 0:
		#	return "D04-W0060"
		#elif chip_num == 1:
		#	return "D03-W0060"
		#elif chip_num == 2:
		#	return "K10-W0060"
		#elif chip_num == 3:
		#	return "F04-W0060"
		#else:
		#	raise ValueError(f"Invalid chip number: {chip_num}")
        

    def get_toa(self, toa, ftoa):
        """
    	Cluster data points for each event and assign globally unique cluster IDs.
    
    	Parameters:
    	- data (pd.DataFrame): DataFrame containing the data to be clustered.
    	- global_cluster_id (int): Initial global cluster ID to start with.

    	Returns:
    	- cluster_dict (dict): A dictionary mapping index to global cluster ID.
    	- global_cluster_id (int): Updated global cluster ID after clustering.
        """		
        ns = toa * 25 - ftoa * 1.5625
        ns = ns.apply(lambda x: 0.0 if x <= 0.0 else x)
        

        return ns # ns

    def get_coordinate_x(self, pixel_id):
        if pixel_id in range(0, 65536): #DET 0
            return (pixel_id % 256)

        elif pixel_id in range(65536, 131072): #DET 1
            new_pixel_id = pixel_id - 65535
            return (256 + (new_pixel_id % 256))
		    
        elif pixel_id in range(131072, 196608): #DET 2
            new_pixel_id = pixel_id - 131072
            return 2*256 + (new_pixel_id % 256)

        elif pixel_id in range(196608, 262144): #DET 3
            new_pixel_id = pixel_id - 196608
            return 3*256 + (new_pixel_id % 256)
		
    def get_coordinate_y(self, pixel_id):
        if pixel_id in range(0, 65536): #DET 0
            return (pixel_id // 256)

        elif pixel_id in range(65536, 131072): #DET 1
            new_pixel_id = pixel_id - 65536
            return (new_pixel_id // 256)
		
        elif pixel_id in range(131072, 196608): #DET 2
            new_pixel_id = pixel_id - 131072
            return (new_pixel_id // 256)

        elif pixel_id in range(196608, 262144): #DET 3
            new_pixel_id = pixel_id - 196608
            return (new_pixel_id // 256)

    def get_coordinate_x_det0(self, pixel_id):
        sqrd_256 = 256*256

        if pixel_id in range(0,sqrd_256): #DET 0
            return (pixel_id % 256)

        elif pixel_id in range(sqrd_256, 2*sqrd_256): #DET 1
            new_pixel_id = pixel_id - sqrd_256
            return (new_pixel_id % 256)
		
        elif pixel_id in range(2*sqrd_256, (2*sqrd_256 + sqrd_256)): #DET 2
            new_pixel_id = pixel_id - (2*sqrd_256)
            return (new_pixel_id % 256)

        elif pixel_id in range(3*sqrd_256, (3*sqrd_256 + sqrd_256)): #DET 3
            new_pixel_id = pixel_id - (3 * sqrd_256)
            return (new_pixel_id % 256)

    def get_coordinate_y_det0(self, pixel_id):
        sqrd_256 = 256*256

        if pixel_id in range(0,sqrd_256): #DET 0
            return (pixel_id // 256)

        elif pixel_id in range(sqrd_256, 2*sqrd_256): #DET 1
            new_pixel_id = pixel_id - sqrd_256
            return (new_pixel_id // 256)
		
        elif pixel_id in range(2*sqrd_256, (2*sqrd_256 + sqrd_256)): #DET 2
            new_pixel_id = pixel_id - (2 * sqrd_256)
            return (new_pixel_id // 256)

        elif pixel_id in range(3*sqrd_256, (3*sqrd_256 + sqrd_256)): #DET 3
            new_pixel_id = pixel_id - (3 * sqrd_256)
            return (new_pixel_id // 256)

    def calc_energy(self, tot, a, b, c, t):
        """
    	Cluster data points for each event and assign globally unique cluster IDs.
    
    	Parameters:
    	- data (pd.DataFrame): DataFrame containing the data to be clustered.
    	- global_cluster_id (int): Initial global cluster ID to start with.

    	Returns:
    	- cluster_dict (dict): A dictionary mapping index to global cluster ID.
    	- global_cluster_id (int): Updated global cluster ID after clustering.
        """		
        p = (b-tot)/a - t
        q = (1/a) * (t*(tot-b) - c)
        det = (p**2)/4 - q
		
        return -p/2 + np.sqrt(det)

    def do_calibration(self, df): # acrescentei calibration_data (mariana)

		
		#calibration_data = self.load_calibration_constants() # adicionei (mariana)
		#final_df = pd.DataFrame()

        df['X0'] = df['Matrix Index'].apply(self.get_coordinate_x_det0)
        df['Y0'] = df['Matrix Index'].apply(self.get_coordinate_y_det0)
        #df['Chip'] = df['Overflow'].apply(self.determine_chip)

		# Function to apply calibration constants row by row
        #def apply_calibration(row):
        #    x0 = row['X0']
        #    y0 = row['Y0']
        #    chip = row['Chip']

                # Get calibration constants for this row (pass single values)
         #   a, b, c, t = self.get_calibration_constants(calibration_data, x0, y0, chip) # alterei (mariana)
                
            # Calculate pre-calibrated energy   
            #return self.calc_energy(row['ToT'], a, b, c, t)

		# Apply calibration row by row using `apply`
        #df['ToT (keV)'] = df.apply(apply_calibration, axis=1)
		#final_df.insert(0, 'ToT (keV)', ToT_keV)

        return df

    def calculate_observation_time(self, df):
        """
    	Cluster data points for each event and assign globally unique cluster IDs.
    
    	Parameters:
    	- data (pd.DataFrame): DataFrame containing the data to be clustered.
    	- global_cluster_id (int): Initial global cluster ID to start with.

    	Returns:
    	- cluster_dict (dict): A dictionary mapping index to global cluster ID.
    	- global_cluster_id (int): Updated global cluster ID after clustering.
    	"""		
        observation_time = (df['Ns'].max() - df['Ns'].min()) * 1E-9  # Convert nanoseconds to seconds
		
        return observation_time

    def resolution(self, E):
        """
    	Cluster data points for each event and assign globally unique cluster IDs.
    
    	Parameters:
    	- data (pd.DataFrame): DataFrame containing the data to be clustered.
    	- global_cluster_id (int): Initial global cluster ID to start with.

    	Returns:
    	- cluster_dict (dict): A dictionary mapping index to global cluster ID.
    	- global_cluster_id (int): Updated global cluster ID after clustering.
    	"""		
        #a = 12.40   
        #b = 0.64
        #c = 0.0514

        #aE = a**2 * E**(-2)
        #bE = b**2 * E**(-1)
        #c2 = c**2

        #R = math.sqrt(aE + bE + c2)
        
        return 1 #20% for tests


    def list_files_in_folder(self, folder_path, extension=".t3pa"):	
        return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension) and not f.endswith(extension + ".info")]


    def check_folders(self, folder):
        """
		List all subdirectories in the specified folder that contain a config.toml file.

		Parameters:
		- folder (str): Path to the folder to check for subdirectories.

		Returns:
		- input_folders (list): A list of paths to subdirectories containing a config.toml file.
		"""
        input_folders = []
        subdirs = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]

# Check for config.toml files
        valid_subdirs = [
            os.path.join(folder, d) for d in subdirs 
            if os.path.isfile(os.path.join(folder, d, 'config.toml'))
        ]

# Append valid subdirectories to input_folders
        input_folders.extend(valid_subdirs)

        return input_folders
	
    def check_input_folders(self, folder):
        """
        Cluster data points for each event and assign globally unique cluster IDs.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the data to be clustered.
        - global_cluster_id (int): Initial global cluster ID to start with.

        Returns:
        - cluster_dict (dict): A dictionary mapping index to global cluster ID.
        - global_cluster_id (int): Updated global cluster ID after clustering.
        """		
        inputFiles = self.list_files_in_folder(folder)

        return inputFiles
	
    def define_chips(self, data, **kwargs):
# Define the mapping of chip indices to their corresponding names
        chip_to_overflow = {index: key for index, key in enumerate(kwargs.keys())}

# Get selected chips from kwargs based on their truthy values
        selected_chips = [index for index, key in chip_to_overflow.items() if kwargs[key]]

# Set nchips to the length of selected chips
        nchips = len(selected_chips)

# Ensure valid selection
        if nchips == 0:
            print("Error: No chips selected. Please provide at least one chip.")
            return None

# Get the chips to exclude from the overflow filter
        exclude_chips = {0, 1, 2, 3} - set(selected_chips)

# Create a mask by excluding overflow values corresponding to chips not selected
        mask = ~data['Overflow'].isin(exclude_chips)

# Filter the DataFrame based on the mask
        chips_df = data[mask]
        chips_df.reset_index(drop=True, inplace=True)

        return chips_df
	
    def sort_df(self, inputFile):
        """
        This function tranforms the ToT to keV values and sorts the dataframe by Ns.	
        """
        #print(inputFile)
        # Apply Pre-calibration
        df = pd.read_csv(inputFile, sep=",")
        
        abc = os.path.basename(inputFile).split('.')[0]
        
        #df_cal = self.do_calibration(df)
        df_cal = df


        chip2_df = self.define_chips(df_cal, chip0=True, chip1=True, chip2=True, chip3=True)

        #mask_chips = (df_cal['Overflow'] != 0) & (df_cal['Overflow'] != 1) & (df_cal['Overflow'] != 3)    
        #mask = mask_chips

        #chip2_df = df_cal[mask]
        #chip2_df.reset_index(drop=True, inplace=True)

        # Apply conversions to get ToA (ns), X, Y and Log Energy
        ns = df['Ns']

        #chip2_df.insert(1, 'Ns', ns)


        #x = df['Matrix Index'].apply(self.get_coordinate_x)
        #chip2_df.insert(2, 'X', x)

        #y = df['Matrix Index'].apply(self.get_coordinate_y)
        #chip2_df.insert(3, 'Y', y)

        # Sort by Ns
        df_sorted = df.sort_values(by='Ns', ascending=True)

        return df_sorted

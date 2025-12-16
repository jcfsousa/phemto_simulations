import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math

# Calibration constants
#CHIPS = ['D03-W0060', 'D04-W0060', 'F04-W0060', 'K10-W0060']
CONSTANTS = ['a', 'b', 'c', 't']
CHIPS = ["D04-W0060", "D03-W0060", "K10-W0060", "F04-W0060"] # adicionei (mariana)

DET_SIDE = 256

class Calibration:

# 	 def __init__(self, data_flag):
#  		 self.data_flag = data_flag
	
	def __init__(self): None
	
	'''def get_calibration_constants(self, x, y, chip):
		#if chip not in CHIPS:
		#    raise ValueError(f"Invalid chip: {chip}. Must be one of {CHIPS}.")
		cal_const = {}

		for constant in CONSTANTS:
			cal_const[constant] = np.loadtxt(f"calibration/pp-calib/{chip}_{constant}.txt", unpack=True)

		return [cal_const[constant][x, y] for constant in CONSTANTS]''' # comentei (mariana)
	
	def load_calibration_constants(self): # adicionei (mariana)
		#if chip not in CHIPS:
		#    raise ValueError(f"Invalid chip: {chip}. Must be one of {CHIPS}.")
		calibration_data = {}

		for chip in CHIPS:
			calibration_data[chip] = {}
			for constant in CONSTANTS:
				calibration_data[chip][constant] = np.loadtxt(f"calibration/pp-calib/{chip}_{constant}.txt", unpack=True)

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
		if chip_num == 0:
			return "D04-W0060"
		elif chip_num == 1:
			return "D03-W0060"
		elif chip_num == 2:
			return "K10-W0060"
		elif chip_num == 3:
			return "F04-W0060"
		else:
			raise ValueError(f"Invalid chip number: {chip_num}")

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

	def do_calibration(self, df, calibration_data): # acrescentei calibration_data (mariana)

		
		#calibration_data = self.load_calibration_constants() # adicionei (mariana)
		#final_df = pd.DataFrame()

		df['X0'] = df['Matrix Index'].apply(self.get_coordinate_x_det0)
		df['Y0'] = df['Matrix Index'].apply(self.get_coordinate_y_det0)
		df['Chip'] = df['Overflow'].apply(self.determine_chip)

		# Function to apply calibration constants row by row
		def apply_calibration(row):
			x0 = row['X0']
			y0 = row['Y0']
			chip = row['Chip']

			# Get calibration constants for this row (pass single values)
			a, b, c, t = self.get_calibration_constants(calibration_data, x0, y0, chip) # alterei (mariana)
			
			# Calculate pre-calibrated energy
			return self.calc_energy(row['ToT'], a, b, c, t)

		# Apply calibration row by row using `apply`
		df['ToT (keV)'] = df.apply(apply_calibration, axis=1).round()
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
		a = 10.43
		b = 0.58
		c = 0.0499

		aE = a**2 * E**(-2)
		bE = b**2 * E**(-1)
		c2 = c**2

		R = math.sqrt(aE + bE + c2)

		return R

	def create_scatter_plot(self, x, y, intensity, input_dir):

		plt.figure(figsize=(40, 10))

		# Create a scatter plot
		scatter = plt.scatter(x, y, c=intensity, cmap='Reds', s=2, norm=mcolors.LogNorm())
		
		# Add color bar
		cbar = plt.colorbar(scatter, label='Energy (keV)')
		cbar.set_label('Energy (keV)')

		# Set labels and title
		plt.xlabel('X-coordinate')
		plt.ylabel('Y-coordinate')
		plt.title('Pixel Matrix Heat Intensity Plot')

		plt.xlim(1, 1024)
		plt.ylim(1, 256)

		#plt.xticks(np.arange(0, 256, 10), rotation=45)
		#plt.yticks(np.arange(0, 256, 10))
		#plt.grid()

		#plt.figure(figsize=(40,10))

		# Save the plot to the specified directory
		if not os.path.exists(input_dir):
			os.makedirs(input_dir)
		plt.savefig(os.path.join(input_dir, 'scatter_plot.png'), dpi = 500)

		# Show the plot (optional)
		plt.show()

	def list_files_in_folder(self, folder_path, extension=".t3pa"):	
		return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension) and not f.endswith(extension + ".info")]
		
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
		# input_filename = inputFiles[1]
		# split_input_filename = input_filename.split('_')
		# energy_input_filename = split_input_filename[2]
		# degree_input_filename = split_input_filename[6]
		
		return inputFiles
	
	def sort_df(self, inputFile, calibration_data):
		"""
    	Cluster data points for each event and assign globally unique cluster IDs.
    
    	Parameters:
    	- data (pd.DataFrame): DataFrame containing the data to be clustered.
    	- global_cluster_id (int): Initial global cluster ID to start with.

    	Returns:
    	- cluster_dict (dict): A dictionary mapping index to global cluster ID.
    	- global_cluster_id (int): Updated global cluster ID after clustering.
    	"""
		
		# Apply Pre-calibration
		df = pd.read_csv(inputFile, sep="\t")
		df_cal = self.do_calibration(df, calibration_data)

		#mask_chips = (df_cal['Overflow'] != 0) & (df_cal['Overflow'] != 1) & (df_cal['Overflow'] != 3)    
		#mask = mask_chips

		#chip2_df = df_cal[mask]
		#chip2_df.reset_index(drop=True, inplace=True)

		# Apply conversions to get ToA (ns), X, Y and Log Energy
		NS = self.get_toa(df['ToA'], df['FToA'])
		df_cal.insert(1, 'Ns', NS)

		X = df['Matrix Index'].apply(self.get_coordinate_x)
		df_cal.insert(2, 'X', X)

		Y = df['Matrix Index'].apply(self.get_coordinate_y)
		df_cal.insert(3, 'Y', Y)

		# Sort by Ns
		df_sorted = df_cal.sort_values(by='Ns')

		return df_sorted
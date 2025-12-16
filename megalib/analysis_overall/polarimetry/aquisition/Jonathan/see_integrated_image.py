import os
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from collections import defaultdict


from tpx_analysis3 import combine_events, plot_hist, get_counts, print_obs_time
from file_handling import try_remove_file, try_create_dir, check_dir

# FILE_DIR = 'C:/Users/jonat/Documents/LIP Coimbra/PIXET/out-files/'
#FILE_DIR = '/home/lipg17/Documents/Timepix3-Python-API/out-files/'
FILE_DIR = '/home/mariana/Documents/THOR-SR/LARIX-analysis/out-files/'
#EVENT_LENGTH = 100 # nanoseconds
EVENT_LENGTH = 40 #ns - por observação do gráfico da Fig 34 do Zé, para uma espessura de 2mm e bias 500V demoram cerca de 30ns, 40 para ter margem

if not os.path.exists(FILE_DIR):
    FILE_DIR = "out-files/"

SPEC = {
    'Cs137': 1000,
    'Ba133': 500,
    'Na22': 2000,
    'Eu152': 2000,
    'Eu154': 2000,
    'Am241': 15000,
    'background-night': 15000,
    'background-day': 15000,
    'ALarix': 300,
    'Larix': 2500
}

def matrix_id(file):
	matrix_id = [x.split()[1] for x in file]
	matrix_id.pop(0)
	#matrix_id[0] = 0
	matrix_id = list(map(int, matrix_id))
	return matrix_id

def ToA(file):
	ToA = [x.split()[2] for x in file]
	ToA.pop(0)
	#ToA[0] = 0
	ToA = list(map(int, ToA))
	return ToA

def ToT(file):
	ToT = [x.split()[3] for x in file]
	ToT.pop(0)  
	#ToT[0] = 0
	ToT = list(map(int, ToT))
	return ToT

def FToA(file):
	FToA= [x.split()[4] for x in file]
	FToA.pop(0)
	#FToA[0] = 0
	FToA = list(map(int, FToA))
	return FToA

def get_coordinate_x(pixel_id):
    if pixel_id < 0 or pixel_id > 65535:
        raise ValueError("Pixel ID must be between 0 and 65535.")

    x = pixel_id % 256

    return x

def read_data(filepath_in):
	#reads the file path#
	with open(filepath_in) as f:
		file = f.readlines()
	return matrix_id(file), ToA(file), ToT(file), FToA(file)

def get_coordinate_y(pixel_id):
    if pixel_id < 0 or pixel_id > 65535:
        raise ValueError("Pixel ID must be between 0 and 65535.")

    y = pixel_id // 256

    return y


def get_overall_image(merged_file_filepath):
    #matrix_id = np.loadtxt(merged_file_filepath, usecols=(0,), skiprows=1)

    matrix_id, ToA, ToT, FToA = read_data(merged_file_filepath)

    counts_dict = defaultdict(int)  # Dictionary to store precomputed counts

    # Precompute counts
    for value in matrix_id:
        counts_dict[value] += 1

    coordinates_print = []

    coordinate_x= []
    coordinate_y= []
    counts = []

    for i in range(65535):
        counts = counts_dict[i]  # Use precomputed counts
        coordinate_x = get_coordinate_x(i)
        coordinate_y = get_coordinate_y(i)
        
        coordinates_print.append([coordinate_x, coordinate_y, counts])


    x_values = [data[0] for data in coordinates_print]
    y_values = [data[1] for data in coordinates_print]
    counts = [data[2] for data in coordinates_print]

    trimmed_list = counts.copy()

    for _ in range(3):                                  #mask 3 noisy pixels
        max_value = max(trimmed_list)
        max_index = trimmed_list.index(max_value)
        trimmed_list[max_index] = 0

    #print("trimmed_list",trimmed_list)  

    plt.scatter(x_values, y_values, c=trimmed_list, cmap='hot',s=1)


    plt.colorbar(label='Counts')
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Heat Intensity Plot')
    #fig = plt.gcf()
    #plt.savefig(img_path)
    plt.show()


def main():
    out_merged_file_path = input('Enter input directory name: ')
    #out = input('Enter output directory name: ')

    get_overall_image(out_merged_file_path)


if __name__ == "__main__":
    main()

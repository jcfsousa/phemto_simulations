import pypixet
import os
import traceback
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


from glob import glob

from tpx_analysis3 import combine_events, get_counts, print_obs_time
from file_handling import try_remove_file, try_create_dir, check_dir

############# ACQUISITION #############

# ascii file as output
FILETYPE = '.t3pa' 

# specify output file directory
OUT_DIR = '/home/mariana/Documents/THOR-SR/LARIX/out-files/'
#OUT_DIR = '/media/mariana/T7 Shield/THOR-SR at Larix/out-files/'
#OUT_DIR = '/media/mariana/Extreme SSD/THOR-SR/out-files/'
FILE_DIR_LOG = 'logfiles/'
#FILE_DIR_LOG = '/media/mariana/Extreme SSD/THOR-SR/logfiles/'

if not os.path.exists(OUT_DIR):
    OUT_DIR = 'out-files/'
    #OUT_DIR = '/media/mariana/T7 Shield/THOR-SR at Larix/out-files/'
    #OUT_DIR = '/media/mariana/Extreme SSD/THOR-SR/out-files/'

    FILE_DIR_LOG = OUT_DIR + FILE_DIR_LOG

def config():
    """Configures the acquisition parameters."""
    filename = str(input("Enter filename: "))
    actime = int(input("Enter total acquisition time in seconds: ")) 
    interval = int(input("Enter duration of each interval in seconds: "))
    bias = float(input("Enter bias voltage in volts: "))
    out_dir = OUT_DIR + filename + '/'
    filepath_log = FILE_DIR_LOG + filename + '_log.txt'
    return filename, actime, interval, bias, out_dir, filepath_log

def create_directory(file_dir):
    """Creates a directory if it doesn't exist."""
    os.makedirs(file_dir, exist_ok=True)

def create_logfile(filepath_log, filename, actime, interval, bias):
    create_directory(FILE_DIR_LOG)
    with open(filepath_log, 'w') as f:
        f.write('logfile created at ' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n\n')
        f.write(f"filename: {filename} \n")
        f.write(f"total acquisition time: {actime} seconds\n")
        f.write(f"duration of each interval: {interval} seconds\n")
        f.write(f"bias input voltage: {bias} V\n\n")
        f.write('timestamp | temperature (°C) | bias set voltage (V) | bias sense voltage (V) | bias sense current (µA) \n')

def log_temp_bias(device, file):
    temp = round(device.temperature(), 3)
    bias_set = device.bias()
    bias_v = round(device.biasVoltageSense(), 3)
    bias_c = round(device.biasCurrentSense(), 5)
    file.write(f"{round(time.time(),1)} {temp} {bias_set} {bias_v} {bias_c}\n")
    print('Temperature:', temp, '°C')
    print('Bias set voltage:', bias_set, 'V')
    print('Bias sense voltage:', bias_v, 'V')
    print('Bias sense current:', bias_c, 'µA')

def get_file_name(sequence, out_dir, filename):
    if sequence == 0:
        return out_dir + filename + FILETYPE
    else:
        return out_dir + filename + '-' + str(sequence) + FILETYPE

def startup():
    pypixet.start()
    pixet = pypixet.pixet
    pixet.refreshDevices()
    try:
        devices = pixet.devicesByType(pixet.PX_DEVTYPE_TPX3)
        dev = devices[0]
    except Exception:
        sys.exit("No TPX3 device found.")
    print('Device', dev.fullName(), 'found.')
    return pixet, dev

def end(pixet):
    pixet.exitPixet()
    pypixet.exit()

def single_acquisition(pixet, dev, sequence, time, out_dir, filename):
    print("doAdvancedAcquisition...")
    rc = dev.doAdvancedAcquisition(1, time, pixet.PX_ACQTYPE_DATADRIVEN, pixet.PX_ACQMODE_NORMAL, pixet.PX_FTYPE_AUTODETECT, 0, get_file_name(sequence, out_dir, filename))
    print(" rc", rc, "(0 is OK)")
    if rc != 0:
    	raise Exception
    

def do_single_acquisition(pixet, dev, sequence, time, filepath_log, out_dir, filename):
    with open(filepath_log, 'a') as f:
        try:
            dev.doSensorRefresh()
            log_temp_bias(dev, f)
            single_acquisition(pixet, dev, sequence, time, out_dir, filename)
        except Exception as e:
                if e == KeyboardInterrupt:
                    print("Acquisition stopped by user.")
                    end(pixet)
                    sys.exit("KeyboardInterrupt")
                else:
                    print("Exception:", e)
                    traceback.print_exc()
                    f.write(f"Exception: {e} \n")


def do_acquisition(filename, actime, interval, bias, out_dir, filepath_log):
    pixet, dev = startup()

    dev.setBias(bias)
    dev.setOperationMode(pixet.PX_TPX3_OPM_TOT_NOTOA)

    create_directory(out_dir)
    create_logfile(filepath_log, filename, actime, interval, bias)

    if actime > interval:
    
        number_of_sequences, rest_time = divmod(actime, interval)
        
        if rest_time > 0:
            number_of_sequences += 1
        else:
            rest_time = interval
        for i in range(number_of_sequences-1):
            do_single_acquisition(pixet, dev, i+1, interval, filepath_log, out_dir, filename)
            print("Sequence", i+1, "of", number_of_sequences, "done.")
        do_single_acquisition(pixet, dev, number_of_sequences, rest_time, filepath_log,
        out_dir, filename)
        print("Sequence", number_of_sequences, "of", number_of_sequences, "done.") 
  
    else:
        do_single_acquisition(pixet, dev, 0, actime, filepath_log, out_dir, filename)
        print("Sequence 1 of 1 done.")

    end(pixet)


############# PLOT OVERALL IMAGE #############

def merge_files(filepath, out_filepath):
    # Write the header to the output file
    header_written = False

    # Open the output file in write mode
    with open(out_filepath, 'w') as output_file:
        # Iterate over each input file
        for file_path in filepath:
            # Check if the file exists
            if not os.path.exists(file_path):
                print(f"File '{file_path}' does not exist. Skipping...")
                continue

            # Open each input file in read mode
            with open(file_path, 'r') as input_file:
                # Skip the header line if it has already been written
                if header_written:
                    next(input_file)

                # Copy the content from the input file to the output file
                output_file.write(input_file.read())

                # Mark that the header has been written
                header_written = True

    print(f'Merged files into {out_filepath}')

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


def get_overall_image(merged_file_filepath, img_path):
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
    plt.savefig(img_path)
    plt.show()




############# PROCESS SPECTRA RAW #############
EVENT_LENGTH = 100 # nanoseconds

if not os.path.exists(OUT_DIR):
    OUT_DIR = "out-files/"
    #OUT_DIR = '/media/mariana/T7 Shield/THOR-SR at Larix/out-files/'
    #OUT_DIR = '/media/mariana/Extreme SSD/THOR-SR/out-files/'



SPEC = {
    'Cs137': 1000,
    'Ba133': 500,
    'Na22': 2000,
    'Eu152': 2000,
    'Eu154': 2000,
    'Am241': 15000,
    #'background-night': 15000,
    #'background-day': 15000,
    'background': 15000,
    'Larix': 2000
}

def specify_fmt():
    return ' '.join(['%d' , '%.3f', '%.3f'])

def save_to_txt(filepath_out, energy, counts, counts_err):
    np.savetxt(filepath_out, np.transpose([energy, np.round(counts, 3), np.round(counts_err, 3)]), fmt = specify_fmt(), header='energy, counts per hour, error')
    
def save_spectra(dir_name, energy, counts,  counts_err, file_name):
    filepath_out = f'spectra_raw/{dir_name}/{file_name}.txt'
    try_create_dir(f'spectra_raw/{dir_name}')
    try_remove_file(filepath_out)
    save_to_txt(filepath_out, energy, counts, counts_err)
    print(f'Saved {filepath_out}')

def process_spectra(files, dir_name, spectra, event_length, upper_lim, mask):
    events, obs_time = combine_events(files, event_length, upper_lim, mask)
    energy, counts, counts_err = get_counts(events, obs_time, 'energy')
    save_spectra(dir_name, energy, counts,  counts_err, spectra)
    return spectra, obs_time

def process(filename):
    check_dir(OUT_DIR + filename)
    DIR_NAME_OUT = filename
    FILE_EXT = filename.split('-',1)[1]
    MASK = bool(int(input('Do you want to use a mask? (yes = 1, no = 0) ')))

    spec_names = []
    obs_times = []

    for key, value in SPEC.items():
        files = glob(f'{OUT_DIR + filename}/{key}*.t3pa')
        if len(files) == 0:
            continue
        else:
            print(f'Processing {key}...')
            if bool(FILE_EXT):
                key += '-' + FILE_EXT
            spec_name, obs_time = process_spectra(files, DIR_NAME_OUT, key, EVENT_LENGTH, value, mask = MASK)
            spec_names.append(spec_name)
            obs_times.append(obs_time)
            
    if len(spec_names) == 0:
        print('No files found')
    else:
        print('Total observation times:')
        for i, name in enumerate(spec_names):
            print_obs_time(name, obs_times[i])


############# PLOT SPECTRA RAW #############

def plot_raw_spectrum(ax, energy, events, key, raw_spectrum_fig, tot = False):
    """Plots the raw spectrum of the element."""
    ax.plot(energy, events, linestyle = '-', label = key, linewidth = 1.5, alpha = 0.9)

    ax.grid(linestyle='dotted')
    ax.set_yscale('log')
    if tot:
        ax.set_xlabel('TOT / ticks')
    else:
        ax.set_xlabel('energy / keV')
    ax.set_ylabel('counts per hour')
    ax.set_title('Raw Spectra, Pre-Calibrated Energy')

    plt.savefig(raw_spectrum_fig)
    plt.show()


############# RE-CALIBRATION #############
def save_calibrated_spectra(dir_name, energy, counts,  counts_err):
    filepath_out = f'spectra_calibrated/{dir_name}/{dir_name}.txt'
    try_create_dir(f'spectra_calibrated/{dir_name}')
    try_remove_file(filepath_out)
    save_to_txt(filepath_out, energy, counts, counts_err)
    print(f'Saved {filepath_out}')


def calibration(e_pre, a, b):
    """Re-Calibrates the energy through the linear calibration."""
    energy = (e_pre - b)/a
    return energy


def calibrate_spectrum(filepath, dir_name):
    """Applies the calibration to the spectrum"""
    if os.path.isfile(filepath):
        energy, counts, counts_err = np.loadtxt(filepath, skiprows=1, unpack=True)

    a = 1.0714
    b = -5.20
    cal_energy = calibration(energy, a, b)
    save_calibrated_spectra(dir_name, cal_energy, counts, counts_err)


def plot_calibrated_spectrum(ax, energy, events, key, calibrated_spectrum_fig, tot = False):
    """Plots the calibrated spectrum of the element"""
    ax.plot(energy, events, linestyle = '-', label = key, linewidth = 1.5, alpha = 0.9)

    ax.grid(linestyle='dotted')
    ax.set_yscale('log')
    #ax.set_xscale('log')
    #ax.set_yscale('linear')
    if tot:
        ax.set_xlabel('TOT / ticks')
    else:
        ax.set_xlabel('energy / keV')
    ax.set_ylabel('counts per hour')
    ax.set_title('Calibrated Spectra')

    plt.savefig(calibrated_spectrum_fig)
    plt.show()


############# PLOT EVENTS IN DELT_T INTERVAL #############
def get_toa(toa, ftoa):
    real_toa = toa * 25 - ftoa * 25/16 #ns
    
    return real_toa


def get_delta_t_image(merged_file_filepath ,delta_t):
    
    matrix_id, ToA, ToT, FToA = read_data(merged_file_filepath)

    delta_t = delta_t * 10**9

    x_values = []
    y_values = []
    counts = []
    counts_dict = defaultdict(int)
    coordinates_print = []
    energy_dict = defaultdict(int)



    for i in range(len(ToA)):
        for j in range(len(ToA)):
            if ToA[j] < 200:
                ToA[j] = ToA[j-1]
            
            elif ToA[i] < 200:
                ToA[i] = ToA[i-1]
                
            if abs(get_toa(ToA[i],FToA[i]) - get_toa(ToA[j],FToA[j])) >= delta_t:
                print("index start: ",i,"index end: ", j)
                print("Time of arrival_initial: ", get_toa(ToA[i],FToA[i])*10**-9 ,"s")
                print("Time of arrival_final: ", get_toa(ToA[j],FToA[j])*10**-9 ,"s")
                print("Delta T: ", abs(get_toa(ToA[i],FToA[i]) - get_toa(ToA[j],FToA[j]))*10**-9 ,"s")
                print("matrix_id_initial: ",matrix_id[i])
                print("matrix_id_final: ",matrix_id[j])

                for b in range(i,j):
                    counts_dict[matrix_id[b]] += 1
                    energy_dict[matrix_id[b]] += ToT[b]


                for a in range(65535):
                    #print(counts_dict)
                    counts = counts_dict[a]  # Use precomputed counts
                    energy = energy_dict[a]
                    coordinate_x = get_coordinate_x(a)
                    coordinate_y = get_coordinate_y(a)
                    coordinates_print.append([coordinate_x, coordinate_y, counts, energy])

                x_values = [data[0] for data in coordinates_print]
                y_values = [data[1] for data in coordinates_print]
                counts = [data[2] for data in coordinates_print]
                energy = [data[3] for data in coordinates_print]

                trimmed_list = counts.copy()
                trimmed_list_energy = energy.copy()

                for _ in range(3):                                  #mask 3 noisy pixels
                    max_value = max(trimmed_list)
                    max_value_energy = max(trimmed_list_energy)
                    max_index = trimmed_list.index(max_value)
                    max_index_energy = trimmed_list_energy.index(max_value_energy)
                    trimmed_list_energy[max_index_energy] = 0
                    trimmed_list[max_index] = 0


                plt.scatter(x_values, y_values, c=trimmed_list_energy, cmap='jet', s=1)
                plt.colorbar(label='kev')
                plt.xlabel('X-coordinate')
                plt.ylabel('Y-coordinate')
                plt.title('Delta T Image')
                plt.show()
                time.sleep(0.1)
                i=j
                counts_dict = defaultdict(int)
                counts = []
                coordinate_x = []
                coordinate_y = []
                coordinates_print = []
                x_values = []
                y_values = []
                trimmed_list = []
                energy_dict = defaultdict(int)
            else:
                pass

    return plt.show()



############# MAIN #############

if __name__ == "__main__":

    filename, actime, interval, bias, out_dir, filepath_log = config()

    do_acquisition(filename, actime, interval, bias, out_dir, filepath_log)
    process(filename)
    
    user_integrated_image = input('Do you want to plot the integrated image? (yes = 1, no = 0)\n')
    
    dir_all = filename
    key_all = dir_all
    #merged_file = f"spectra_raw/{dir_all}/{key_all}.txt"

    files_path = glob(f'{OUT_DIR + filename}/{key_all}*.t3pa')
    out_merged_file_path = f"{OUT_DIR + filename}/{key_all}-merged.txt"
    merge_files(files_path, out_merged_file_path)
    
    if user_integrated_image == '1':
        #integrated_fig = f"spectra_raw/{dir_all}/{key_all}.png"
        img_path = f"{OUT_DIR + filename}/{key_all}-integrated_image.png"

        if os.path.isfile(out_merged_file_path):
            get_overall_image(out_merged_file_path, img_path)
            print(f"Image Saved {img_path}")
        

    elif user_integrated_image == '0':
        pass

    else:
        raise ValueError('Please enter 1 or 0')


    user_spectrum = input('Do you want to plot the raw energy spectrum? (yes = 1, no = 0)\n')

    dir = filename
    key = dir
    file = f"spectra_raw/{dir}/{key}.txt"
    #file = f"/media/mariana/T7 Shield/THOR-SR at Larix/spectra_raw/{dir}/{key}.txt"
    #file = f"/media/mariana/Extreme SSD/THOR-SR/spectra_raw/{dir}/{key}.txt"


    
    if user_spectrum == '1':
        raw_spectrum_fig = f"spectra_raw/{dir}/{key}.png"
        #raw_spectrum_fig = f"/media/mariana/T7 Shield/THOR-SR at Larix/spectra_raw/{dir}/{key}.png"
        #raw_spectrum_fig = f"/media/mariana/Extreme SSD/THOR-SR/spectra_raw/{dir}/{key}.png"

        if os.path.isfile(file):
            fig, ax = plt.subplots()

            energy, events = np.loadtxt(file, skiprows=1, usecols=(0, 1), unpack=True)
            plot_raw_spectrum(ax, energy, events, key, raw_spectrum_fig)
            print(f"Plot Saved {raw_spectrum_fig}")
        

    elif user_spectrum == '0':
        pass

    else:
        raise ValueError('Please enter 1 or 0')
    
    calibrate_spectrum(file, dir)

    user_calibrated_spectrum = input('Do you want to plot the calibrated energy spectrum? (yes = 1, no = 0)\n')

    dir_cal = filename
    key_cal = dir_cal
    file_cal = f"spectra_calibrated/{dir_cal}/{key_cal}.txt"
    #file_cal = f"/media/mariana/T7 Shield/THOR-SR at Larix/spectra_calibrated/{dir_cal}/{key_cal}.txt"
    #file_cal = f"/media/mariana/Extreme SSD/THOR-SR/spectra_calibrated/{dir_cal}/{key_cal}.txt"

    calibrated_spectrum_fig = f"spectra_calibrated/{dir_cal}/{key_cal}.png"
    #calibrated_spectrum_fig = f"/media/mariana/T7 Shield/THOR-SR at Larix/spectra_calibrated/{dir_cal}/{key_cal}.png"
    #calibrated_spectrum_fig = f"/media/mariana/Extreme SSD/THOR-SR/spectra_calibrated/{dir_cal}/{key_cal}.png"

    if user_calibrated_spectrum == '1':
        fig_cal, ax_cal = plt.subplots()

        if os.path.isfile(file_cal):
            energy_cal, events_cal = np.loadtxt(file_cal, skiprows=1, usecols=(0, 1), unpack=True)
            plot_calibrated_spectrum(ax_cal, energy_cal, events_cal, key_cal, calibrated_spectrum_fig)
            print(f"Plot Saved {calibrated_spectrum_fig}")
            

    elif user_calibrated_spectrum == '0':
        pass

    else:
        raise ValueError('Please enter 1 or 0')



    user_delta_t_images = input('Do you want to plot events with delta_t windows? (yes = 1, no = 0)\n')

    if user_delta_t_images == '1':

        if os.path.isfile(out_merged_file_path):
            delta_t = input("What do you want the delta_t to be? ")

            get_delta_t_image(out_merged_file_path, int(delta_t))
            

    elif user_delta_t_images == '0':
        pass

    else:
        raise ValueError('Please enter 1 or 0')
        

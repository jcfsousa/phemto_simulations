import os
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

from tpx_analysis import combine_events, plot_hist, get_counts, print_obs_time
from file_handling import try_remove_file, try_create_dir, check_dir

# FILE_DIR = 'C:/Users/jonat/Documents/LIP Coimbra/PIXET/out-files/'
FILE_DIR = '/home/lipg17/Documents/Timepix3-Python-API/out-files/'
EVENT_LENGTH = 100 # nanoseconds

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

def show_hist(events, title):
    plot_hist(events, title)
    plt.show(block=False)
    plt.pause(10)
    plt.close()

def process_spectra(files, dir_name, spectra, event_length, upper_lim, mask):
    events, obs_time = combine_events(files, event_length, upper_lim, mask)
    # show_hist(events, f'{spectra} spectrum')
    energy, counts, counts_err = get_counts(events, obs_time, 'energy')
    save_spectra(dir_name, energy, counts,  counts_err, spectra)
    return spectra, obs_time

def main():
    DIR_NAME_IN = input('Enter input directory name: ')
    check_dir(FILE_DIR + DIR_NAME_IN)
    DIR_NAME_OUT = input('Enter output directory name: ')
    FILE_EXT = input('Enter spectrum extenstion name (ENTER for just the name of the isotope): ')
    MASK = bool(int(input('Do you want to use a mask? (yes = 1, no = 0) ')))

    spec_names = []
    obs_times = []

    for key, value in SPEC.items():
        files = glob(f'{FILE_DIR + DIR_NAME_IN}/{key}*.t3pa')
        if len(files) == 0:
            continue
        else:
            print(f'Processing {key}...')
            if bool(FILE_EXT):
                key += FILE_EXT
            spec_name, obs_time = process_spectra(files, DIR_NAME_OUT, key, EVENT_LENGTH, value, mask = MASK)
            spec_names.append(spec_name)
            obs_times.append(obs_time)
            
    if len(spec_names) == 0:
        print('No files found')
    else:
        print('Total observation times:')
        for i, name in enumerate(spec_names):
            print_obs_time(name, obs_times[i])


if __name__ == "__main__":
    main()

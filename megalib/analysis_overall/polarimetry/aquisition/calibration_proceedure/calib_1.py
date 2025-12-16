import numpy as np
import os

# calibration constants for factory calibration
CONSTANTS = ['a', 'b', 'c', 't']

# length of each event in ns
EVENT_LENGTH = 100

# constants for the calibration
A = 1.0682
B = -8.8411


# ------------------- energy calibration ------------------- #


def get_cals(calc_dict, x, y):
    """Returns the calibration constants for a given pixel."""
    return ([calc_dict[constant][x, y] for constant in CONSTANTS])

def get_const():
    """Loads the calibration constants from the files into a dictionary."""
    cal_const = {}
    for constant in CONSTANTS:
        cal_const[constant] = np.loadtxt(f'calibration-constants/constant-{constant}.txt', unpack=True)
    return cal_const

def calc_energy(tot, a, b, c, t):
    """Calculates the energy from the tot using the calibration constants."""
    p = (b-tot)/a - t
    q = (1/a) * (t*(tot-b) - c)
    det = (p**2)/4 - q
    return -p/2 + np.sqrt(det)

def get_energy(tot, calc_dict, x, y):
    """Calculates the energy from the tot using the calibration constants for a given pixel."""
    return calc_energy(tot, *get_cals(calc_dict, x, y))

# ------------------- read in spectra ------------------- #

def check_filepath(filepath_in, filepath_out):
    if filepath_in == filepath_out:
        raise ValueError('Input and output files must be different.')

def remove_file(filepath):
    if os.path.isfile(filepath):
        os.remove(filepath)

def remove_empty_file(filepath):
    if os.path.getsize(filepath) == 0:
        os.remove(filepath)
        print('Removed empty file:', os.path.basename(filepath))
        print('This file is probably corrupted. Please use a different file.')
        raise ValueError('Empty file.')
    
def calculate_percent_depricated(filepath_in, filepath_out):
    size_in = os.path.getsize(filepath_in)
    size_out = os.path.getsize(filepath_out)
    per = (size_in - size_out) / size_in * 100
    return round(per, 1)


def fix_spectra(filepath):
    """Removes lines with non-ascii characters from the file."""
    filepath_out = filepath.replace('.t3pa', '-fixed.t3pa')
    remove_file(filepath_out)
    with open(filepath, 'rb') as fr:
        lines = fr.readlines()
        with open(filepath_out, 'a') as fw:
            counter = 0
            for line in lines:
                try:
                    fw.write(line.decode('ascii'))
                except UnicodeDecodeError:
                    counter += 1
        print(f'Removed {counter} lines from {os.path.basename(filepath)} ({calculate_percent_depricated(filepath, filepath_out)} %)')

def load_data(filepath):
    header = np.loadtxt(filepath, max_rows=1, dtype=object)
    data = np.loadtxt(filepath, skiprows=1, dtype=np.int64, unpack=True)
    return header, data

def read_data(filepath):
    """Loads the data from the file into a numpy array."""
    try:
        return load_data(filepath)
    except (ValueError, UnicodeDecodeError):
        print('File contains non-ascii characters. Fixing file...')
        fix_spectra(filepath)
        filepath = filepath.replace('.t3pa', '-fixed.t3pa')
        return load_data(filepath)

def format_header(header):
    header = np.delete(header, 2)
    header[1] = "Matrix Index"
    header[3] = 'Energy'
    fmt = '\t'.join(['%-5s' , '%12s', '%-2s', '%-2s', '%-2s', '%-s'])
    header = fmt % tuple(header)
    return header

def save_data(filepath_out, header, data, energy):
    """Saves the data to a file."""
    data[3] = energy
    fmt = '\t'.join(['%-5d' , '%5d', '%11d', '%3d', '%2d', '%d'])
    header = format_header(header)
    np.savetxt(filepath_out, np.transpose(data), fmt=fmt, header=header, comments='')

# ------------------- calibration ------------------- #


def do_calibration(data):
    """Calculates the energy from the tot for each pixel."""
    calc_dict = get_const()
    y, x = divmod(data[1], 256)
    energy = get_energy(data[3], calc_dict, x, y)
    return energy

def main():
    """Reads in the data from the file and calibrates it."""
    filepath_in = input('Enter filepath of file to calibrate: ')
    filepath_out = input('Enter filepath of output file: ')
    check_filepath(filepath_in, filepath_out)
    remove_file(filepath_out)
    print('Loading data...')
    header, data = read_data(filepath_in)
    print('Data loaded. Calibrating energy...')
    energy = do_calibration(data)
    save_data(filepath_out, header, data, energy)
    print('Calibration complete.')

if __name__ == '__main__':
    main()

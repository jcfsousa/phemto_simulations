import os
import sys

def try_create_dir(filepath):
    """Creates a directory if it doesn't exist."""
    if not os.path.exists(filepath):
        os.makedirs(filepath)

def check_file_exists(filepath):
    """Checks if a file exists."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError('Please provide the correct path of the file you want to calibrate.')

def check_dir_exists(dir_path):
    """Checks if a directory exists."""
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory {dir_path} does not exist.")

def check_dir(filepath):
    """Checks if the directory of a file exists."""
    try:
        check_dir_exists(filepath)
    except FileNotFoundError as e:
        sys.exit(e)
        

#TODO: do I need this? currently not used
def check_filepath(filepath_in, filepath_out):
    """Checks if the input and output files are different."""
    if filepath_in == filepath_out:
        raise ValueError('Input and output files must be different.')

def try_remove_file(filepath):
    """Removes a file if it exists."""
    if os.path.isfile(filepath):
        os.remove(filepath)

def handle_empty_file(filepath, filepath_fixed):
    """Removes an empty file."""
    if os.path.getsize(filepath_fixed) == 0:
        os.remove(filepath_fixed)
        print('Fixed file is empty. Removing fixed file and renaming original file:', 'CURRUPTED_' + os.path.basename(filepath))
        # add CURRUPTED to the beginning of the basename
        os.rename(filepath, filepath.replace(os.path.basename(filepath), 'CURRUPTED_' + os.path.basename(filepath)))
    
def calculate_percent_depricated(filepath_in, filepath_out):
    """Calculates the percentage of bytes removed from the file."""
    size_in = os.path.getsize(filepath_in)
    size_out = os.path.getsize(filepath_out)
    per = (size_in - size_out) / size_in * 100
    return round(per, 1)

def fix_spectra(filepath):
    """Removes lines with non-ascii characters from the file and saves the fixed file in the same directory."""
    check_file_exists(filepath)
    filepath_out = filepath.replace('.t3pa', '_fixed.txt')
    try_remove_file(filepath_out)
    with open(filepath, 'rb') as fr:
        lines = fr.readlines()
        with open(filepath_out, 'a') as fw:
            counter = 0
            for line in lines:
                try:
                    line = line.decode('ascii')
                    if len(line.split("\t")) != 6:
                        counter += 1
                    else: 
                        fw.write(line)
                except UnicodeDecodeError:
                    counter += 1
        print(f'Removed {counter} lines from {os.path.basename(filepath)} ({calculate_percent_depricated(filepath, filepath_out)} %)')
    handle_empty_file(filepath, filepath_out)

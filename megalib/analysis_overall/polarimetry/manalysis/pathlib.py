import os
import re

def sort_files(files):
    sorted_files = sorted(files, key=extract_id_from_file)
    return sorted_files

def extract_id_from_file(filename):
    match = re.search(r'r(\d+)', filename)
    return int(match.group(1)) if match else 0

def check_dir_exists(dir):
    if not os.path.isdir(dir):
        #print(f"Directory {dir} doesnt exist")
        return False
    else:
        return True


def check_number_files_in_dir(dir, startswith='', endswith=''):
    if check_dir_exists(dir):
        check_folder = [f for f in os.listdir(dir) if f.startswith(startswith) and f.endswith(endswith)]
        return len(check_folder)
    else:
        return 0


def creat_dir(dir):
    if check_dir_exists(dir):
        return
    else:
        os.makedirs(dir, exist_ok=True)
        return


def get_list_files(dir, startswith='', endswith=''):
    if check_dir_exists(dir):
        list_files = [f for f in os.listdir(dir) if f.startswith(startswith) and f.endswith(endswith)]
        return list_files

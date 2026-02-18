import subprocess
import argparse
import manalysis.pathlib as pathlib
import manalysis.specLib as specLib
import manalysis.configlib as configlib

import os
import glob

global_config = None

class config:
    output_folder: str = None
    inputPol: str = None
    inputNonPol: str = None

def print_preprocess_ascii():
    """Print PRE-PROCESS in ASCII art style"""
    art = """
██████╗ ██████╗ ███████╗    ██████╗ ██████╗  ██████╗  ██████╗███████╗███████╗███████╗
██╔══██╗██╔══██╗██╔════╝    ██╔══██╗██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔════╝██╔════╝
██████╔╝██████╔╝█████╗      ██████╔╝██████╔╝██║   ██║██║     █████╗  ███████╗███████╗
██╔═══╝ ██╔══██╗██╔══╝      ██╔═══╝ ██╔══██╗██║   ██║██║     ██╔══╝  ╚════██║╚════██║
██║     ██║  ██║███████╗    ██║     ██║  ██║╚██████╔╝╚██████╗███████╗███████║███████║
╚═╝     ╚═╝  ╚═╝╚══════╝    ╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚══════╝╚══════╝╚══════╝
    """
    print(art)

if __name__ == '__main__':
                  

    print_preprocess_ascii()

    parser = argparse.ArgumentParser(description='This parser takes .tra.gz files from revan megalib and identifies single and double compton events in the data. It Parses the data to the PHEMTO 4x4 HED configuration and saves it into a .csv like file named *.t3pa.')
    parser.add_argument('-i', '--inputSource', required=True, help='FULL path of the location of the .t3pa Source')

    args = parser.parse_args()
    

    global_config = config()
    global_config.inputSource = args.inputSource

    input_folder = global_config.inputSource


    specLib.pre_process_source(input_folder)
    specLib.process_event_multiplicity(input_folder)

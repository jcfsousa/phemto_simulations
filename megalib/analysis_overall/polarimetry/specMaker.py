import os
import argparse
import sys
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import manalysis.specLib as specLib
from multiprocessing import Pool
from tqdm import tqdm
from matplotlib.lines import Line2D
import manalysis.configlib as configlib
import os
import manalysis.pathlib as pathlib
import manalysis.comptons as comptons

custom_specRange = False
max_spectra_range = None
min_spectra_range = 0

fontsize = 20
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (8,8)
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize + 4 
plt.rcParams['figure.titlesize'] = fontsize + 6
plt.rcParams['axes.labelsize'] = fontsize + 6
plt.rcParams['axes.titlepad'] = fontsize
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.fontsize'] = fontsize - 4 
plt.rcParams['axes.labelsize'] = fontsize + 2
plt.rcParams['xtick.labelsize'] = fontsize 
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.grid'] = True
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = fontsize*0.35
plt.rcParams['ytick.major.size'] = fontsize*0.35
plt.rcParams['xtick.minor.size'] = fontsize*0.175
plt.rcParams['ytick.minor.size'] = fontsize*0.175
plt.rcParams['xtick.major.width'] = fontsize*0.1
plt.rcParams['ytick.major.width'] = fontsize*0.1
plt.rcParams['xtick.minor.width'] = fontsize*0.1
plt.rcParams['ytick.minor.width'] = fontsize*0.1
plt.rcParams['axes.linewidth'] = fontsize/fontsize
plt.rcParams['grid.linestyle'] = 'dotted'

def compare_spectra(args):
    global custom_specRange, max_spectra_range, min_spectra_range
    output_folder_chips, event_type_folder, event_type, chip, source, defaultFigure, log = args
    
    defaultFigure = False

    spec_masked_output_folder = os.path.join(event_type_folder, 'spectra') 
    os.makedirs(spec_masked_output_folder, exist_ok = True)
    
    plt.figure(figsize=(10,6)) ##initizlize figure

    fig, ax = plt.subplots(figsize=(10, 6))  # Create a figure and axes


    binscalib, cntscalib, obs_time = specLib.get_spectra_histCalib_toPlot(event_type_folder,
                                                    event_type = event_type,
                                                    output_file_name=f'Calib_spec_hist-{event_type}_chip{chip}.txt',
                                                    chips = chip) 

    
    bins, cnts, obs_time = specLib.get_spectra_hist(event_type_folder,
                                                    event_type = event_type,
                                                    output_file_name=f'preCalib_spec_hist-{event_type}_chip{chip}.txt',
                                                    chips = chip)
    

    if custom_specRange == False:
        x, y = specLib.search_peaks(binscalib, cntscalib)
        
        if len(x) == 0:
            x = [1400]

        max_spectra_range = x[-1] + x[-1] * 0.5

    specLib.plot_energy_spectra_sources(output_folder_chips,
                                        bins,
                                        cnts,
                                        log=log,
                                        show=False,
                                        custom_name=f'{event_type}-chip{chip}_CalibComparison',
                                        defaultFigure=defaultFigure,
                                        min_spectra_range = min_spectra_range,
                                        max_spectra_range = max_spectra_range,
                                        colour = 'Orange',
                                        ax=ax,
                                        plotLabel = 'Uncalibrated')
    
    ax = specLib.plot_energy_spectra_sources(output_folder_chips,
                                        binscalib,
                                        cntscalib,
                                        log=log,
                                        show=False,
                                        custom_name=f'{event_type}-chip{chip}_CalibComparison',
                                        defaultFigure=defaultFigure,
                                        min_spectra_range = min_spectra_range,
                                        max_spectra_range = max_spectra_range,
                                        colour = 'black',
                                        ax=ax,
                                        plotLabel = 'Calibrated')
    
    if defaultFigure == True:
        plt.close()
    plt.close(fig)


def spectra_Calib(args):
    global custom_specRange, max_spectra_range, min_spectra_range
    output_folder_chips, event_type_folder, event_type, chip, source, defaultFigure, log = args

    spec_masked_output_folder = os.path.join(event_type_folder, 'spectra') 
    os.makedirs(spec_masked_output_folder, exist_ok = True)
    

    binscalib, cntscalib, obs_time = specLib.get_spectra_histCalib_toPlot(event_type_folder,
                                                    event_type = event_type,
                                                    output_file_name=f'Calib_spec_hist-{event_type}_chip{chip}.txt',
                                                    chips = chip) 

    
    if custom_specRange == False:
        x, y = specLib.search_peaks(binscalib, cntscalib)
        
        if len(x) == 0:
            x = [1400]

        max_spectra_range = x[-1] + x[-1] * 0.5
    
    specLib.plot_energy_spectra_sources(output_folder_chips, binscalib,
                                        cntscalib,
                                        log=log,
                                        show=False,
                                        custom_name=f'{event_type}-chip{chip}_Calib',
                                        defaultFigure=defaultFigure,
                                        min_spectra_range = min_spectra_range,
                                        max_spectra_range = max_spectra_range,
                                        plotLabel= "Calibrated Data")
    
    if defaultFigure == True:
        plt.close()
    plt.close()


def spectra(args):
    global custom_specRange, max_spectra_range, min_spectra_range  
    #this function requires to have the source peak on the config file, i need to change this.......

    output_folder_chips, event_type_folder, event_type, chip_id, chip_name, source, defaultFigure, log, source_energy, plt_show = args

    spec_masked_output_folder = os.path.join(event_type_folder, 'spectra') 
    os.makedirs(spec_masked_output_folder, exist_ok = True)

    if chip_id == 0: # Si
        bin_step = 0.1 #keV
    elif chip_id == 1: # CdTe
        bin_step = 0.1
    else:
        bin_step = 1

    bins, cnts, obs_time = specLib.get_spectra_hist(event_type_folder,
                                                    bin_step = bin_step,
                                                    event_type = event_type,
                                                    output_file_name=f'{source}_spec_hist-{event_type}_Det-{chip_name}.txt',
                                                    chip_id = chip_id,
                                                    chip_name = chip_name)
    
    if custom_specRange == False:
        try:
            max_spectra_range = source_energy  * 1.2
        except:
            x, y = specLib.search_peaks(bins, cnts)
            
            if len(x) == 0:
                x = [1400]

            max_spectra_range = x[-1] + x[-1] * 0.5

    specLib.plot_energy_spectra_sources(output_folder_chips, bins, cnts,
                                        log=log, show=plt_show,
                                        custom_name=f'{event_type}-Det-{chip_name}_preCalib',
                                        defaultFigure=defaultFigure,
                                        min_spectra_range = min_spectra_range,
                                        max_spectra_range = max_spectra_range,
                                        plotLabel= f"Uncalibrated Data: {event_type}") 
    if defaultFigure == True:
        plt.close()
    plt.close()


'''
def compare_chips_peaks(args):
   " This function is not updated to work with the sources_database.json sources_peaks format"

    source, output_base, sources_peaks = args
    
    input_folder = os.path.join(specLib.global_config.input_dir, source)
    source_id = os.path.basename(input_folder)
    resultFolder = os.path.join(specLib.global_config.output_folder, source_id)
    os.makedirs(resultFolder, exist_ok=True)
    resultFolder_parquet = os.path.join(resultFolder, 'parquet')    

    singles_folder = os.path.join(resultFolder_parquet,'singles')
    doubles_folder = os.path.join(resultFolder_parquet, 'doubles')
    multiples_folder = os.path.join(resultFolder_parquet, 'multiples')
    masked_folder = os.path.join(resultFolder_parquet, 'masked')

    for source_in_dict, peaks in sources_peaks.items():
        if source_in_dict == source:
            peak_energy = round(max(peaks),0)
            if peak_energy < 100:
                min_spectra_range = int(peak_energy * 0.6)
                max_spectra_range = int(peak_energy * 1.1)
            else:
                min_spectra_range = int(peak_energy *  0.9)
                max_spectra_range = int(peak_energy * 1.1)
            
            plt.figure(figsize=(10,10))

            event_type = 'singles'

            event_type_folder = singles_folder
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 
                      'yellow', 'brown', 'pink', 'lime', 'teal', 'olive', 'navy', 
                      'maroon', 'gold', 'gray', 'black', 'indigo', 'violet']


            chips = [0, 1, 2, 3]
            i = 0
            for chip in chips:

                hist_event_type_folder = os.path.join(event_type_folder, 'spectra') 

                bin_centers, cnts, _ = specLib.get_spectra_hist(event_type_folder,
                                                                event_type=event_type,
                                         output_file_name=f'spec_hist-{event_type}_chip{chip}.txt',
                                                                chips=chip)

                peak_bins, peak_indices = specLib.search_peaks(bin_centers, cnts)
                
                measured_peak = max(peak_bins) 
                specLib.plot_energy_spectra_sources(resultFolder, bin_centers, cnts,
                                                    min_spectra_range=min_spectra_range,
                                                    max_spectra_range=max_spectra_range,
                                                    show=False,
                                                    custom_name=f'{source_in_dict}_{peak_energy}keV_comparison_preCalib',
                                                    source_peaks = peaks,
                                                    measuredPeak = measured_peak,
                                                    defaultFigure = False,
                                                    colour=colors[i], plotSinglePeak=True,
                                                    log = False,
                                                    chip=chip)
                

                i+=1
        #plt.show()
        plt.close()
                
    for source_in_dict, peaks in sources_peaks.items():
        if source_in_dict == source:
            peak_energy = round(max(peaks),0)
            if peak_energy < 100:
                min_spectra_range = int(peak_energy * 0.6)
                max_spectra_range = int(peak_energy * 1.1)
            else:
                min_spectra_range = int(peak_energy *  0.9)
                max_spectra_range = int(peak_energy * 1.1)
            
            plt.figure(figsize=(10,10))

            event_type = 'singles'

            event_type_folder = singles_folder
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 
                      'yellow', 'brown', 'pink', 'lime', 'teal', 'olive', 'navy', 
                      'maroon', 'gold', 'gray', 'black', 'indigo', 'violet']


            chips = [0, 1, 2, 3]
            i = 0
            for chip in chips:

                hist_event_type_folder = os.path.join(event_type_folder, 'spectra') 

                bin_centers_calib, cnts_calib, _ = specLib.get_spectra_histCalib_toPlot(event_type_folder,
                                                                event_type=event_type,
                                         output_file_name=f'spec_hist-{event_type}_chip{chip}.txt',
                                                                                        chips=chip)

                peak_bins_calib, peak_indices_calib = specLib.search_peaks(bin_centers_calib, cnts_calib)
                
                measured_peak_calib = max(peak_bins_calib) 
                specLib.plot_energy_spectra_sources(resultFolder, bin_centers_calib, cnts_calib,
                                                    min_spectra_range=min_spectra_range,
                                                    max_spectra_range=max_spectra_range,
                                                    show=False,
                                                    custom_name=f'{source_in_dict}_{peak_energy}keV_comparison_Calib',
                                                    source_peaks = peaks,
                                                    measuredPeak = measured_peak_calib,
                                                    defaultFigure = False,
                                                    colour=colors[i], plotSinglePeak=True,
                                                    log = False,
                                                    chip=chip)
                i+=1
        plt.close()

'''
'''
    event_type = 'singles'
    event_type_folder = singles_folder
    
    for chip in chips:
        check_calib_curve = os.path.join(output_base, 'results', '1-QuadCharacterizationResults', f'chip{chip}', 'calib', f'calibCurve_chip{chip}_singles.csv')
        
        output_folder_chips = os.path.join(output_folder,f'chip-{chip}')
       
        if os.path.exists(check_calib_curve):  
    
            process_args = [
                (output_folder_chips,
                 event_type_folder,
                 check_calib_curve,
                 spectra_range,
                 event_type,
                 chip,
                 True,
                 True)
            ]

            with Pool() as pool:
                for _ in tqdm(pool.imap_unordered(spectra_Calib, process_args), total=len(process_args), desc=f"Plotting Spectra Chip-{chip}"):
                    pass

'''

'''
        spec_singles_output_folder = os.path.join(singles_folder, 'spectra') 
        os.makedirs(spec_singles_output_folder, exist_ok = True)
        print(f"    Plotting chip-{chip} single spectra....")
        single_plot_df, ignore = specLib.get_df_hist(singles_folder, spec_singles_output_folder, chip)
        single_bins, single_cnts_s = specLib.get_spectra_histogram(spec_singles_output_folder, single_plot_df,
                                                                   output_file_name=f'spec_hist-chip{chip}.txt',
                                                                   max_spectra_range=spectra_range,
                                                                   observation_time=observation_time)    
        specLib.plot_energy_spectra_sources(output_folder_chips, single_bins, single_cnts_s,
                                             min_spectra_range=0,
                                             max_spectra_range=spectra_range, log=True, show=False,
                                             custom_name=f'{source_id}-singles-chip{chip}',
                                             defaultFigure=defaultFigure) 
        if defaultFigure == True:
             plt.close()
    
       


        spec_doubles_output_folder = os.path.join(doubles_folder, 'spectra') 
        os.makedirs(spec_doubles_output_folder, exist_ok = True)
        print(f"    Plotting chip-{chip} double spectra....")
        double_plot_df, ignore= specLib.get_df_hist(doubles_folder, spec_doubles_output_folder, chip)
        double_bins, double_cnts_s = specLib.get_spectra_histogram(spec_doubles_output_folder, double_plot_df,
                                                                   output_file_name=f'spec_hist-chip{chip}.txt',
                                                                   max_spectra_range=spectra_range,
                                                                   observation_time=observation_time)
        specLib.plot_energy_spectra_sources(output_folder_chips, double_bins, double_cnts_s,
                                             min_spectra_range=0,
                                             max_spectra_range=spectra_range, log=False, show=False,
                                             custom_name=f'{source_id}-doubles-chip{chip}',
                                             defaultFigure=defaultFigure, source_peaks=sources_peaks[source_id])    
        del double_plot_df
        if defaultFigure == True:
            plt.close()


        spec_multiples_output_folder = os.path.join(multiples_folder, 'spectra') 
        os.makedirs(spec_multiples_output_folder, exist_ok = True)
        print(f"    Plotting chip-{chip} multiple spectra....")
        multiple_plot_df, ignore = specLib.get_df_hist(multiples_folder, spec_multiples_output_folder, chip)
        multiple_bins, multiple_cnts_s = specLib.get_spectra_histogram(spec_multiples_output_folder,
                                                                       multiple_plot_df,
                                                                       output_file_name=f'spec_hist-chip{chip}.txt',
                                                                      max_spectra_range=spectra_range,
                                                                       observation_time=observation_time)
        specLib.plot_energy_spectra_sources(output_folder_chips, multiple_bins, multiple_cnts_s,
                                    min_spectra_range=0, max_spectra_range=spectra_range, log=True, 
                                    show=False, custom_name=f'{source_id}-multiples-chip{chip}', 
                                    defaultFigure=defaultFigure, source_peaks=sources_peaks[source_id])    
        del multiple_plot_df
        plt.close()
        ''' 

global_config = None

class config:
    output_folder: str = None
    inputPol: str = None
    inputNonPol: str = None

def print_specmaker_ascii():
    """Print SPECMAKER in ASCII art style"""
    art = """
███████╗██████╗ ███████╗ ██████╗███╗   ███╗ █████╗ ██╗  ██╗███████╗██████╗ 
██╔════╝██╔══██╗██╔════╝██╔════╝████╗ ████║██╔══██╗██║ ██╔╝██╔════╝██╔══██╗
███████╗██████╔╝█████╗  ██║     ██╔████╔██║███████║█████╔╝ █████╗  ██████╔╝
╚════██║██╔═══╝ ██╔══╝  ██║     ██║╚██╔╝██║██╔══██║██╔═██╗ ██╔══╝  ██╔══██╗
███████║██║     ███████╗╚██████╗██║ ╚═╝ ██║██║  ██║██║  ██╗███████╗██║  ██║
╚══════╝╚═╝     ╚══════╝ ╚═════╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝
    """
    print(art)


if __name__ == '__main__':
                  

    print_specmaker_ascii()

    parser = argparse.ArgumentParser(description='This parser takes .tra.gz files from revan megalib and identifies single and double compton events in the data. It Parses the data to the PHEMTO 4x4 HED configuration and saves it into a .csv like file named *.t3pa.')
    parser.add_argument('-o', '--output', required=True, help='Base file for output of polarimetry analysis')
    parser.add_argument('-i', '--inputSource', required=True, help='FULL path of the location of the .t3pa Source')
    parser.add_argument('-s', '--show', action='store_true', help='Show plots. Code will wait untill plot is closed')
    parser.add_argument('-l', '--log', action='store_true', help='Plots are y log scale')
    parser.add_argument('-r', '--range', help='Provide spectra range ie: 0-600 . If not provided it will use the _<energy>keV_ present on name of source from 0-<energy>*1.2. If this is also not provided it will try to find a peak in the spectra and adjust to this range')

    args = parser.parse_args()
    

    global_config = config()
    global_config.output_folder = args.output
    global_config.inputSource = args.inputSource

    source = global_config.inputSource.split('/')[-1]

    pathlib.creat_dir(global_config.output_folder)

    output_folder_base = global_config.output_folder
    input_folder = global_config.inputSource


    pathlib.creat_dir(output_folder_base)
    
    
    chip_config = {0:"Si", 1:'CdTe'} #Overflow 0 on .t3pa is Si, Overflow on .t3pa is CdTe

    source_energy = comptons.get_energy_from_source_name(source)

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)

    
    if args.log:
        log = True
    else:
        log = False

    
    if args.range:
        custom_specRange = True
        try:
            spec_range = args.range.split('-')
            min_spectra_range = float(spec_range[0])
            max_spectra_range = float(spec_range[-1])
        except Exception as e:
            print(f"\033[31m ERROR: Please input spectral range in the correct format... x-y, {e}.\033[0m")
            sys.exit(1)

        if (max_spectra_range - min_spectra_range) <= 0:
            print("\033[31m ERROR: Max spectral range should be larger than Min spectral range.\033[0m")
            sys.exit(1)
    else: 
        custom_specRange = False


    source_id = source
    resultFolder = output_folder_base
    os.makedirs(resultFolder, exist_ok=True)
    resultFolder_parquet = os.path.join(resultFolder, 'parquet')    
    singles_folder = os.path.join(resultFolder_parquet,'singles')
    doubles_folder = os.path.join(resultFolder_parquet, 'doubles')
    multiples_folder = os.path.join(resultFolder_parquet, 'multiples')
    masked_folder = os.path.join(resultFolder_parquet, 'masked')

    event_type_folder = (masked_folder, singles_folder, doubles_folder, multiples_folder)
    
    defaultFigure = True
    if defaultFigure == False:
        plt.figure(figsize=(10,4))


    event_type = ('masked', 'singles', 'doubles', 'multiples')
    chips = [0,1] # 0=Si, 1=CdTe


    for chip_id, chip_name in chip_config.items():
        output_folder_chips = os.path.join(resultFolder,f'chip-{chip_name}')
        
        os.makedirs(output_folder_chips, exist_ok = True)

        process_args = [(output_folder_chips, event_type_folder, 
                        event_type, chip_id, chip_name, source, True, log, source_energy, args.show)
                        for event_type, event_type_folder in zip(event_type, event_type_folder)]

        
        #for task in tqdm(process_args,
        #                 total=len(process_args),
        #                 desc='Spectra'):
        #    spectra_preCalib(task)
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(spectra, process_args), total=len(process_args), desc=f"Plotting Spectra Chip-{chip_name}"):
                pass
            
            #with Pool() as pool:
            #    for _ in tqdm(pool.imap_unordered(spectra_Calib, process_args), total=len(process_args), desc=f'Plotting Spectra Calibrated Chip-{chip}'):
            #        pass

            #with Pool() as pool:
            #    for _ in tqdm(pool.imap_unordered(compare_spectra, process_args), total=len(process_args), desc=f'Plotting Pre/Post Calibrated Spectra Chip-{chip}'):
            #        pass
        #perform_spectra(args)
        #except Exception as e:
        #    print(f'\033[31m ERROR: {e}. Maybe try to run pre_process.py script first.\033[0m')
        #    sys.exit(1)

        #try:
        #compare_chips_peaks(args)
        #except Exception as e:
        #    print(f'\033[31m ERROR: {e}. Maybe try to run pre_process.py script first.\033[0m')




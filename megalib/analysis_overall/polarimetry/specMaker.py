import os
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

custom_specRange = False
max_spectra_range = None
min_spectra_range = 0

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


def spectra_preCalib(args):
    global custom_specRange, max_spectra_range, min_spectra_range  
    #this function requires to have the source peak on the config file, i need to change this.......

    output_folder_chips, event_type_folder, event_type, chip, source, defaultFigure, log = args

    spec_masked_output_folder = os.path.join(event_type_folder, 'spectra') 
    os.makedirs(spec_masked_output_folder, exist_ok = True)
    

    bins, cnts, obs_time = specLib.get_spectra_hist(event_type_folder,
                                                    event_type = event_type,
                                                    output_file_name=f'preCalib_spec_hist-{event_type}_chip{chip}.txt',
                                                    chips = chip)
    
    if custom_specRange == False:
        x, y = specLib.search_peaks(bins, cnts)
        
        if len(x) == 0:
            x = [1400]

        max_spectra_range = x[-1] + x[-1] * 0.5

    specLib.plot_energy_spectra_sources(output_folder_chips, bins, cnts,
                                        log=log, show=False,
                                        custom_name=f'{event_type}-chip{chip}_preCalib',
                                        defaultFigure=defaultFigure,
                                        min_spectra_range = min_spectra_range,
                                        max_spectra_range = max_spectra_range,
                                        plotLabel= f"Uncalibrated Data: {event_type}") 
    plt.show() 
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



if __name__ == '__main__':
                    
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)

    selected_config = configlib.initialconfig(parent_dir)
    
    specLib.global_config = specLib.Config(selected_config)              
    
    print('Chip configuration loaded:')
    with open(specLib.global_config.config_chips) as f:
        text = f.readlines()
        print(text)

    sources = specLib.global_config.sources
    sources_peaks = specLib.global_config.sources_peaks
    output_folder_base = specLib.global_config.output_folder
    config_chips = specLib.global_config.config_chips
    
    
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
    
    log_input = input("Logarithm scale (y/n): ")

    if log_input == 'y':
        log = True
    elif log_input == 'n':
        log = False
    else:
        print("\033[31m ERROR: Please provide valid answer.... \033[0m")
        sys.exit(1)

    
    custom_ranges = input("Automatic energy range (y/n): ")

    if custom_ranges == 'n':
        custom_specRange = True
        try:
            min_spectra_range = int(input("Min spectra range (int): "))
        except Exception as e:
            print(f"\033[31m ERROR: Please input an interger, {e}.\033[0m")
        try:
            max_spectra_range = int(input("Max spectra range (int): "))
        except Exception as e:
            print(f"\033[31m ERROR: Please input an interger, {e}.\033[0m")

        if (max_spectra_range - min_spectra_range) <= 0:
            print("\033[31m ERROR: Max spectral range should be larger than Min spectral range.\033[0m")
            sys.exit(1)
    elif custom_ranges == 'y':
        custom_specRange = False
    else: 
        print("\033[31m Please provide valid answer....\033[0m")
        sys.exit(1)


    for source in sources:
        #args = (source, output_folder_base, sources_peaks)
        #try: 
        source_id = source
        input_folder = os.path.join(specLib.global_config.input_dir, source)
        resultFolder = os.path.join(specLib.global_config.output_folder, source_id)
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
        chips = [0,1]


        for chip in chips:
            output_folder_chips = os.path.join(resultFolder,f'chip-{chip}')
            
            os.makedirs(output_folder_chips, exist_ok = True)

            process_args = [(output_folder_chips, event_type_folder, 
                            event_type, chip, source, True, log)
                            for event_type, event_type_folder in zip(event_type, event_type_folder)]

            
            #for task in tqdm(process_args,
            #                 total=len(process_args),
            #                 desc='Spectra'):
            #    spectra_preCalib(task)
            with Pool() as pool:
                for _ in tqdm(pool.imap_unordered(spectra_preCalib, process_args), total=len(process_args), desc=f"Plotting Spectra Chip-{chip}"):
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




import os
import sys
from numpy._core.defchararray import splitlines
import manalysis.pathlib as pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import gc
import random
import json
import manalysis.identify_events as dataflux
from tqdm import tqdm
import manalysis.comptons as compton
from matplotlib.ticker import ScalarFormatter

global_config = None

class Config:
    def __init__(self, config_file):

        with open(config_file, 'r') as file:
            config_data = json.load(file)
            file.close()

        self.config_chips = config_data["config_chips"]
        self.sources = config_data["sources"]
        self.input_dir = config_data["input_dir"]
        self.output_folder = config_data["output_folder"]
        self.chip_dict = {}
        
        source_database = f"{os.path.dirname(config_file)}/sources_database.json"
        with open(source_database, "r") as file:
            source_db = json.load(file)
            file.close()
        self.sources_peaks = source_db

        try:
            with open(self.config_chips, 'r') as file:
                chip_config = file.read().splitlines()
                for line in chip_config:
                    split_line = line.split('=')
                   
                    chip_id = split_line[0]
                    chip = split_line[1]
                    self.chip_dict[int(chip_id)] = chip
        except Exception as e:
            print(f"\033[31m Critical ERROR, chip config did not load. ERROR: {e}\033[0m")
            sys.exit(1)
        
            
    def __str__(self):
        return json.dumps({
            "config_chips": self.config_chips,
            "sources": self.sources,
            "sources_peaks": self.sources_peaks,
            "input_dir": self.input_dir,
            "output_folder": self.output_folder,
        }, indent=4)



def pre_process_source(source):
    '''
    Multi threading

    # Step 1: Identify events by time interval

    # Step 2: Load and process the data

    # Step 3: Cluster the events

    # Step 4: Save the processed data

    '''
    input_folder = os.path.join(global_config.input_dir, source)
    source_id = source 

    resultFolder = os.path.join(global_config.output_folder, source_id)
    os.makedirs(resultFolder, exist_ok=True)
    resultFolder_parquet = os.path.join(resultFolder, 'parquet')    
    
    print(f"Analysing '{source}' radioactive source....")

    if pathlib.check_dir_exists(resultFolder_parquet): #check if identify all already ran by comparing to number of files on input directory
        n_files_output_folder = pathlib.check_number_files_in_dir(resultFolder_parquet, endswith='.parquet')
        n_files_input_folder = pathlib.check_number_files_in_dir(input_folder, endswith='.t3pa')
        
        if n_files_output_folder >=  n_files_input_folder :
            print('     identify_all() already run.... ')
        else:
            print(f'     identify_all() didnt fully ran, [{n_files_output_folder}/{n_files_input_folder}] files ran. Calling identify_all() again....')
            dataflux.identify_all(input_folder, resultFolder_parquet, event_length=1000)
    else:
        pathlib.creat_dir(resultFolder_parquet) 
        dataflux.identify_all(input_folder, resultFolder_parquet, event_length=1000)

    gc.collect() # collect garbage.........


def process_event_multiplicity(source):

    input_folder = os.path.join(global_config.input_dir, source)
    source_id = source 

    resultFolder = os.path.join(global_config.output_folder, source_id)
    os.makedirs(resultFolder, exist_ok=True)
    resultFolder_parquet = os.path.join(resultFolder, 'parquet')    

    observation_time = 0

    single_events_df = pd.DataFrame()
    double_events_df = pd.DataFrame()
    multiple_events_df = pd.DataFrame()
   
    singles_folder = os.path.join(resultFolder_parquet,'singles')
    doubles_folder = os.path.join(resultFolder_parquet, 'doubles')
    multiples_folder = os.path.join(resultFolder_parquet, 'multiples')
    masked_folder = os.path.join(resultFolder_parquet, 'masked')

    n_files_singles = pathlib.check_number_files_in_dir(singles_folder, endswith='.parquet')
    n_files_doubles = pathlib.check_number_files_in_dir(doubles_folder, endswith='.parquet')
    n_files_multiples = pathlib.check_number_files_in_dir(multiples_folder, endswith='.parquet')
    n_files_masked = pathlib.check_number_files_in_dir(masked_folder, endswith='.parquet')
    n_files_toProcess = pathlib.check_number_files_in_dir(resultFolder_parquet, endswith='.parquet')

    if n_files_toProcess == max(n_files_singles, n_files_doubles, n_files_multiples, n_files_masked):
            print(f'    singles, doubles and multiples already identified... folder:{resultFolder_parquet} ')

    else:
        n_files = pathlib.check_number_files_in_dir(resultFolder_parquet, startswith='df_all_data_df_', endswith='.parquet')
        if n_files == 0:
            print(f"    No .parquet files found on {resultFolder_parquet}.")
        else:
            pathlib.creat_dir(singles_folder)
            pathlib.creat_dir(doubles_folder)
            pathlib.creat_dir(multiples_folder)
            pathlib.creat_dir(masked_folder)
          
            parquet_files = pathlib.get_list_files(resultFolder_parquet, startswith='df_all_data_df_', endswith='.parquet')
            i = 1
            for i, parquet_file in enumerate(tqdm(parquet_files, total=n_files,
                                                  desc=f"Computing, single, double and multiple, {source_id}",
                                                  unit="file"), start=1):

                all_data_df_abc = pd.read_parquet(f"{resultFolder_parquet}/{parquet_file}", 
                                                  columns = ['Matrix Index', 'ToT (keV)', 'Ns',
                                                             'Overflow', 'Event', 'Cluster', 'X', 'Y'])
                
                #matrix_indices_to_remove = [135964, 186265, 255633, 201903,
                #                            245595, 203846, 151849, 201207,
                #                            190134, 122651, 245594]

                #all_data_df_abc_filtered = all_data_df_abc[~all_data_df_abc['Matrix Index'].isin(matrix_indices_to_remove)]
                all_data_df_abc_filtered = all_data_df_abc

                #input()
                #single_cluster_events, double_cluster_events, multiple_cluster_events = compton.get_multiplicity_events(all_data_df_abc_filtered)

                single_events_df, double_events_df, multiple_events_df = compton.get_multiplicity_events(all_data_df_abc_filtered) 
                #single_events_df = all_data_df_abc_filtered[all_data_df_abc_filtered['Event'].isin(single_cluster_events)]
                #double_events_df = all_data_df_abc_filtered[all_data_df_abc_filtered['Event'].isin(double_cluster_events)]
                #multiple_events_df = all_data_df_abc_filtered[all_data_df_abc_filtered['Event'].isin(multiple_cluster_events)]


                single_events_df.to_parquet(f"{singles_folder}/singles_{parquet_file}")
                double_events_df.to_parquet(f"{doubles_folder}/doubles_{parquet_file}")
                multiple_events_df.to_parquet(f"{multiples_folder}/multiples_{parquet_file}")
                all_data_df_abc_filtered.to_parquet(f"{masked_folder}/masked_{parquet_file}")

                i=i+1

                del single_events_df
                del double_events_df
                del multiple_events_df
                del all_data_df_abc
                gc.collect()

def df_parameterize(df):
    column_types = {
        'ToT': 'int8',
        'FToA': 'int8',
        'X': 'int8',
        'Y': 'int8',
        'XO': 'int8',
        'YO': 'int8',
        'ToT (keV)': 'int8',
        'Matrix Index': 'int16',
        'Event': 'int32',
        'Overflow': 'int8',
        'Cluster': 'int32',
    }

    valid_columns = {col: dtype for col, dtype in column_types.items() if col in df.columns}
    #print(valid_columns)
    df = df.fillna(0).replace([float('inf'), float('-inf')], 0)

    return df.astype(valid_columns)




def get_spectra_histCalib_toPlot(parquet_folder,  event_type='singles', output_file_name = 'spec_histogram.txt', min_spectra_range = 0, max_spectra_range = 1600, bin_step = 1, counts_normalized = False, chips = 'all'):
    '''
    To get the correct histogram for data analysis run the 'get_spectra_histCalib()' function. If you want to perform a plot of the histogram use this function since it includes a smoothening of the histogram spectra. 
   
    parquet_folder: place where the .parquet folders are to analyse
    output_folder: place to put the spectra .txt
    counts_normalize: not operational atm......
    '''
    parquet_files = [f for f in os.listdir(parquet_folder) if f.endswith('.parquet')]
    
    ns_cumulative = 0
   
    bins_shared = np.arange(min_spectra_range, max_spectra_range, bin_step)
    cumulative_counts = np.zeros(len(bins_shared)-1)

    bin_centers = (bins_shared[:-1] + bins_shared[1:]) / 2

    for parquet_file in parquet_files:
        df = pd.read_parquet(f"{parquet_folder}/{parquet_file}", columns=['Event', 'Cluster', 'ToT (keV)', 'Overflow', 'Ns'])
        df.fillna(0)

        if df.empty:
            continue

        if isinstance(chips, int):
            df = df[df['Overflow'] == chips]

        calib_df_plot = pd.DataFrame()
        calib_df_plot = cluster_energy_calib_toPlot(df) # this adds a random number (0 to 1) to each event to smoothen the curve.v
        if calib_df_plot is None or calib_df_plot.empty:
            continue 

        cnts_calib_plot, _ = np.histogram(calib_df_plot['EnergyCalib (keV)'],
                                     bins=bins_shared,
                                     range=(min_spectra_range, max_spectra_range))
        
        cumulative_counts += cnts_calib_plot

        bin_centers = (bins_shared[:-1] + bins_shared[1:]) / 2   
    
    observation_time = ns_cumulative
   
    return bin_centers, cumulative_counts, observation_time



def apply_calib_curve(energy, a, b):
    return (a * energy + b)


def apply_calib_curve_by_overflow(row):
    #a, b = get_calib_constants(row['Overflow'])  # Retrieve constants based on Overflow 
    #PHEMTO simulation
    a = 1 
    b = 0
    #print(f'a={a}, b={b}')
    return a * row['ToT (keV)'] + b

def cluster_energy_calib(df, show_individual_clusters = False):
    '''
    Accepts dataframe relative to an Event, with collumns 'Event', 'Cluster', 'ToT (keV)', 'Overflow'.
    It calibrates the cluster depending on the chip the interaction appened, then it summs the calibrated energy per cluster to get the total energy of the event.

    '''
    cluster_preCalib = pd.DataFrame()
    #print(df)
    cluster_preCalib = df.groupby(['Event', 'Cluster', 'Overflow']).agg({'ToT (keV)': 'sum'}).reset_index()
    cluster_preCalib['ToT (keV)'] = abs(cluster_preCalib['ToT (keV)'])

    cluster_preCalib['EnergyCalib (keV)'] = cluster_preCalib.apply(apply_calib_curve_by_overflow, axis=1) # apply calib (a,b) depending on overflow
    
    cluster_preCalib['EnergyCalib (keV)'] = cluster_preCalib['EnergyCalib (keV)'].astype(float)

    if show_individual_clusters == True:
        cluster_calib = cluster_preCalib.groupby(['Event', 'Cluster', 'Overflow'])['EnergyCalib (keV)'].sum() 
        return cluster_calib

    events_calib = pd.DataFrame()
    events_calib = cluster_preCalib.groupby('Event', as_index=False, group_keys=False).agg({'EnergyCalib (keV)': 'sum'}).reset_index()
    events_calib.drop(columns=['index'], inplace=True)
    return events_calib


def apply_calib_curve_toPlot_by_overflow(row):
    # known binning issue solution, at this stage gonna leave it like that...... dec2024
    coin_toss = random.randint(0,1)
    if coin_toss == 1:
        a, b = get_calib_constants(row['Overflow'])  # Retrieve constants based on Overflow 
        #print(f'a={a}, b={b}')
        energy_calib = (a * row['ToT (keV)'] + b) + (random.uniform(0,1))
        return energy_calib 
    else:
        a, b = get_calib_constants(row['Overflow'])  # Retrieve constants based on Overflow 
        #print(f'a={a}, b={b}')
        energy_calib = (a * row['ToT (keV)'] + b) - (random.uniform(0,1))
        return energy_calib 

def cluster_energy_calib_toPlot(df):
    '''
    The function 'cluster_energy_curve()' should be used to calibrate the events and not this one.
    This function is to be used to calibrate the clusters to perform the plot of the calibrated Histogram. 
    Accepts dataframe with collumns 'Event', 'Cluster', 'ToT (keV)'
    '''
    cluster_preCalib = pd.DataFrame()
    
    #cluster_preCalib = df.groupby(['Event', 'Cluster'])['ToT (keV)'].sum().reset_index()
    cluster_preCalib = df.groupby(['Event', 'Cluster']).agg({'ToT (keV)': 'sum', 'Overflow': 'first'}).reset_index()

    cluster_preCalib['ToT (keV)'] = abs(cluster_preCalib['ToT (keV)'])

    if cluster_preCalib.empty:
        return None
    
    cluster_preCalib['EnergyCalib (keV)'] = cluster_preCalib.apply(apply_calib_curve_toPlot_by_overflow, axis=1) # apply calib (a,b) depending on overflow

    cluster_preCalib['EnergyCalib (keV)'] = cluster_preCalib['EnergyCalib (keV)'].astype(int)
    events_calib = pd.DataFrame()
    events_calib = cluster_preCalib.groupby('Event', as_index=False, group_keys=False).agg({'EnergyCalib (keV)': 'sum'}).reset_index()

    events_calib.drop(columns=['index'], inplace=True)
    
    return events_calib

def get_calib_constants(chip_id):
    '''
    This function uses the singles for the calibration.
    Returns the a,b calib constants (y=ax+b) for a given chip_id.
    '''
    
    chip_id = int(chip_id)
    calib_dict = global_config.calib_dict
    
    a = calib_dict[chip_id][0]
    b = calib_dict[chip_id][1]

    return a,b 

def get_res_constants():
    '''
    This function grabs the a,b,c constants of the resolution curve, given the path on the {conf}.json file.
    Returns the a,b,c constants for the percentual value of the resolution, 
    R(%)=\sqrt{a^2 E^{-2} + b^2 E^{-1} + c^2}, E is the cluster energy.
    '''

    chip_id = int(chip_id)
    res_dict = global_config.res_dict
    
    a = res_dict[chip_id][0]
    b = res_dict[chip_id][1]
    c = res_dict[chip_id][2]

    return a,b,c






def get_spectra_histCalib(parquet_folder, event_type='singles', output_file_name = 'spec_histogram.txt', min_spectra_range = 0, max_spectra_range = 1600, bin_step = 1, counts_normalized = False, chips = 'all'):
    '''
    parquet_folder: place where the .parquet folders are to analyse
    output_folder: place to put the spectra .txt
    counts_normalize: not operational atm.....
    '''
    parquet_files = [f for f in os.listdir(parquet_folder) if f.endswith('.parquet')]
    
    output_folder = os.path.join(parquet_folder, 'spectra')   
    os.makedirs(output_folder, exist_ok = True)


    ns_cumulative = 0
   
    bins_shared = np.arange(min_spectra_range, max_spectra_range, bin_step)
    cumulative_counts = np.zeros(len(bins_shared)-1)

    bin_centers = (bins_shared[:-1] + bins_shared[1:]) / 2

    for parquet_file in parquet_files:
        df = pd.read_parquet(f"{parquet_folder}/{parquet_file}", columns=['Event', 'Cluster', 'ToT (keV)', 'Overflow', 'Ns'])
        df.fillna(0)

        if isinstance(chips, int):
            df = df[df['Overflow'] == chips]
        
        calib_df = pd.DataFrame()
        calib_df = cluster_energy_calib(df) 

        cnts_calib, _ = np.histogram(calib_df['EnergyCalib (keV)'],
                                     bins=bins_shared,
                                     range=(min_spectra_range, max_spectra_range))
        
        cumulative_counts += cnts_calib

        bin_centers = (bins_shared[:-1] + bins_shared[1:]) / 2   
    
    observation_time = ns_cumulative
    output_file_Calib = os.path.join(output_folder, f'Calib_{output_file_name}')
    os.makedirs(output_folder, exist_ok=True)      
    data_to_save_Calib = np.column_stack((bin_centers, cumulative_counts))
    np.savetxt(output_file_Calib, data_to_save_Calib,
               header=f'ObsTime(ns)\t{observation_time}\nBin(keV)\tcnts(1/s)', delimiter='\t') 
    
    return bin_centers, cumulative_counts, observation_time

def get_spectra_hist(parquet_folder, event_type='singles', output_file_name = 'spec_histogram.txt', min_spectra_range = 0, max_spectra_range = 1600, bin_step = 1, counts_normalized = False, chips = 'all'):
    '''
    parquet_folder: place where the .parquet folders are to analyse
    output_folder: place to put the spectra .txt
    counts_normalize: not operational atm......
    '''
    parquet_files = [f for f in os.listdir(parquet_folder) if f.endswith('.parquet')]
    n_files = len(parquet_files)
    
    output_folder = os.path.join(parquet_folder, 'spectra')   
    os.makedirs(output_folder, exist_ok = True)
    check_spect_file = [file for file in os.listdir(output_folder) if file.startswith('spec_hist') and file.endswith(f'chip{chips}.txt')]

    if len(check_spect_file) == 200:
    #histrogram already on the file path, go read the histogram instead of using the df to get new histogram 
        hist_data_file_path = os.path.join(output_folder, output_file_name)
         #print('I already have a spectra file.........')
        hist_data = np.loadtxt(hist_data_file_path)
        bin_centers = hist_data[:, 0]
        counts_per_second = hist_data[:, 1]  
        return bin_centers, counts_per_second, 1
 
    else:
        if n_files == 0:
            print(f'No .parquet files found on {parquet_folder}.')
            return
        else:
            ns_cumulative = 0
           
            n_bins = np.arange(min_spectra_range, max_spectra_range, bin_step)  # Define bin edges
            cumulative_counts = np.zeros(len(n_bins) - 1)
            bin_centers = (n_bins[:-1] + n_bins[1:]) / 2

            for parquet_file in parquet_files:
                
                df = pd.read_parquet(f"{parquet_folder}/{parquet_file}", columns = ['Event', 'Cluster', 'ToT (keV)','Overflow','Ns'])
                max_ns_in_file = df['Ns'].max()
                ns_cumulative = ns_cumulative + max_ns_in_file 
                
                if isinstance(chips, int):
                    df = df[df['Overflow'] == chips]
                #masking chips
                df.fillna(0) ## in case NA appears in the data, idk why................
    
                events = pd.DataFrame()
                events = df[['Event', 'ToT (keV)']].groupby('Event').sum()
                
                cnts, _ = np.histogram(events['ToT (keV)'], bins = n_bins, range=(min_spectra_range,max_spectra_range))
                cumulative_counts += cnts   

        observation_time = ns_cumulative

        output_file_preCalib = os.path.join(output_folder, f'preCalib_{output_file_name}')
        os.makedirs(output_folder, exist_ok=True)      
        data_to_save_preCalib = np.column_stack((bin_centers, cumulative_counts))
        np.savetxt(output_file_preCalib, data_to_save_preCalib,
                   header=f'ObsTimei(ns)\t{observation_time}\nBin(keV)\tcnts(1/s)', delimiter='\t') #pre Calib

        return bin_centers, cumulative_counts, observation_time

def plot_energy_spectra_sources(outputFolder, bin_centers, counts, 
                                min_spectra_range=0, max_spectra_range=1600, 
                                log=True, show=True, custom_name='', 
                                defaultFigure=True, source_peaks=[], 
                                colour='black', chip='', 
                                plotSinglePeak=False, measuredPeak=0, ax=None,
                                plotLabel = None, event_type = None):
    """
    Plots the energy spectrum from the given data.

    Parameters:
        ax (matplotlib.axes.Axes): Axes object to use for plotting. 
                                   If None, uses the current axis or creates a new one.
        [other parameters remain the same as before]
    """
    mask = (bin_centers >= min_spectra_range) & (bin_centers <= max_spectra_range)
    bin_centers = bin_centers[mask]
    counts = counts[mask]
    
    if ax is None:
        if defaultFigure:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            ax = plt.gca()

    ax.plot(bin_centers, counts, c=colour, linewidth = 2, drawstyle='steps-mid', label=plotLabel)
    ax.set_xlim(min_spectra_range, max_spectra_range)
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Counts')
    ax.tick_params(axis='both', which='both', top=True, bottom=True, right=True)
    ax.grid(False)
    #ax.legend(loc='upper right')
    #ax.set_title(f'Energy Spectrum {custom_name}')
    ax.minorticks_on()

    colors = ['red', 'blue', 'green', 'magenta', 'purple', 'cyan', 'orange',
              'yellow', 'brown', 'pink', 'lime', 'teal', 'olive', 'navy',
              'maroon', 'gold', 'gray', 'black', 'indigo', 'violet']
    
    #if plotSinglePeak:
      
    #    ax.axvline(measuredPeak, ymin=0, ymax=1, c=colour, ls='dashed',
    #              label=f'chip-{chip}: {round(measuredPeak)} keV')
    #elif isinstance(source_peaks, (float, int)):
    #    ax.axvline(source_peaks, ymin=0, ymax=1, color='red', ls='dashed',
    #               label=f'Theoretical: {source_peaks} keV')
    #elif source_peaks:
    #    for i, peak in enumerate(source_peaks):
    #        ax.axvline(peak, ymin=0, ymax=1, color=colors[i % len(colors)],
    #                   ls='dashed', label=f'{peak} keV')

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        unique_labels = dict(zip(labels, handles))  # Remove duplicates
        ax.legend(unique_labels.values(), unique_labels.keys(), loc='best')

    if log:
        try:
            ax.set_yscale('log')
            ax.set_ylabel('Counts')
        except Exception as e:
            print(f"\033[33m WARNING: Could not use log scale, will use linear, {e}.\033[0m")
            ax.set_yscale('linear')
            ax.set_ylabel('Counts')

    ax.figure.savefig(f'{outputFolder}/Energy_Spectrum_{custom_name}.png')
    if show:
        ax.figure.show()

    return ax

'''
def plot_energy_spectra_sources(outputFolder, bin_centers, counts_per_second, min_spectra_range=0, max_spectra_range=1600, log=True, show=True, custom_name='', defaultFigure=True, source_peaks = [], colour = 'black', chip = '', plotSinglePeak = False, measuredPeak = 0, ax=None):
    """
    Plots the energy spectrum from the given data.

    Parameters:
        outputFolder (str): Path to save the output plot.
        total_observation_time (float): Total observation time in seconds to normalize counts.
        spectra_range (float): Maximum energy range for the histogram.
        log (bool): If True, use a log scale for the y-axis.
        show (bool): If True, display the plot.
    """
    mask = (bin_centers >= min_spectra_range) & (bin_centers <= max_spectra_range)
    bin_centers = bin_centers[mask]
    counts_per_second = counts_per_second[mask]
    
    errors = np.sqrt(counts_per_second)
    if defaultFigure == True: #its true if we want to creat a figure outside this function
        plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, counts_per_second, c=colour, drawstyle = 'steps-mid')  # Plot as a line with markers
    plt.xlim(min_spectra_range, max_spectra_range)
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    #plt.yticks(np.arange(0, max(counts_per_second)+1, max(counts_per_second)/10))
    #plt.title(f'Energy Spectrum - {energy_input_filename}')
    plt.title(f'Energy Spectrum {custom_name}')
    plt.grid(True)

    colors = ['red', 'blue', 'green', 'magenta', 'purple', 'cyan', 'orange', 
              'yellow', 'brown', 'pink', 'lime', 'teal', 'olive', 'navy', 
              'maroon', 'gold', 'gray', 'black', 'indigo', 'violet']
    if plotSinglePeak:
        plt.axvline(measuredPeak,ymin=0, ymax=1, c=colour , ls='dashed', label=f'chip-{chip}: {round(measuredPeak)} keV')

    elif type(source_peaks) == float or type(source_peaks) == int:
        #print(f'peaks = {source_peaks}')
        plt.axvline(source_peaks,ymin=0, ymax=1,  color = 'red', ls='dashed', label=f'Theoretical: {source_peaks} keV')
    elif len(source_peaks) == 0:
        pass
    else:
        for i in range(len(source_peaks)):
            #print(f'peaks = {source_peaks[i]}')
            plt.axvline(source_peaks[i], ymin=0, ymax=1,
                        color = f'{colors[i]}', ls='dashed',
                        label=f'{source_peaks[i]} keV')

    handles, lables = plt.gca().get_legend_handles_labels()

    if lables:
        unique_lables = dict(zip(lables,handles))
        plt.legend(unique_lables.values(), unique_lables.keys())

    if log:
        plt.yscale('log')
        plt.ylabel('Counts per Second')


    plt.savefig(f'{outputFolder}/Energy_Spectrum_{custom_name}.png')
    
    if show:
        plt.show()

    return 
'''


def search_peaks(bins, cnts, peak_width = 10):
    '''
    This function accepts np.arrays and checks for peaks on the data. It was initially designed to 
    search peaks on radiactive spectra.

    Parameters:
        bins -> x axis 

        cnts -> y axis

        peak_width -> expected peak width

    Output example:
        peak_bins: [ 21.5 187.5 433.5 684.5] 
        peak_indices: [ 20 186 432 683]
    '''
    cnts_filtered = np.array(cnts)

    peak_indices, properties = scipy.signal.find_peaks(
            cnts_filtered,
            width = peak_width            # Minimum width of peaks
            )
    
    peak_bins = bins[peak_indices]

    return peak_bins, peak_indices


def whereEvent(event_df, chip_id):
    '''
    Checks if a given event if fully inside a determined chip. Returns True if yes. Returns False if the event if on multiple chips (ie:. compton, high energy proton/electron).
    '''
    return (event_df['Overflow'] == chip_id).any()


def get_chip_id(chip):
    for chip_id, chip_dict in global_config.chip_dict.items():
        if chip_dict == chip:
            return int(chip_id)
    raise ValueError(f" {chip} not fount in chip_dict: {global_config.chip_dict}")

import os, psutil
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.optimize import curve_fit
from calibration.calibration import Calibration
from alive_progress import alive_bar
from concurrent.futures import ProcessPoolExecutor, as_completed
import psutil
import time
from multiprocessing import Pool
from tqdm import tqdm
import gc

def identify_events_time(inputFile, event_length, max_event_index, outputFolder_csv, *kwargs):
    """
    Performs coincidence time analysis of ToA+FToA. 
    Identifies events with a coincidence time of <event_length>.This function can be called numerouse times, (ie: for loop reading several .t3pa) and it will append the event_index given the user provides as input the last event index used, <max_event_index>. 
    
    Parameters:
    - inputFile: .t3pa file path to be analysed.
    - event_length: event coincidence window.
    - max_event_index: reference event ID, next events will index this reference number
    - calibration_data: is a np array of the calibration constants (a,b,c,t)
    - inputFolder: Path to the input folder containing data files.

    """
    calib = Calibration('','')
    
    df_sorted = calib.sort_df(inputFile) # sorts the data by ns.

    event = 1
    event_index = np.zeros(len(df_sorted['Ns']))

    current_event_start = df_sorted['Ns'].iloc[0]
    for i in range(len(df_sorted['Ns'])):
        if df_sorted['Ns'].iloc[i] - current_event_start < event_length:
            event_index[i] = event
        else:
            current_event_start = df_sorted['Ns'].iloc[i]
            event += 1
            event_index[i] = event

    df_sorted['Event'] = event_index + max_event_index
    
    # Update the max_event_index for the next file
    max_event_index = df_sorted['Event'].max()
    total_observation_time = calib.calculate_observation_time(df_sorted)
    abc = os.path.basename(inputFile).split('.')[0]
    df_sorted.to_parquet(f'{outputFolder_csv}/df_sorted_{abc}.parquet')
    
    del df_sorted, total_observation_time

    return max_event_index

# Function to find neighbors using NumPy
def find_neighbors(arr, index):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    x, y = arr[index]
    # Finding all neighbors within Manhattan distance of 1
    neighbors = np.where((np.abs(arr[:, 0] - x) <= 1) & (np.abs(arr[:, 1] - y) <= 1))[0]
    neighbors = neighbors[neighbors != index]  # Exclude self
    
    return neighbors

# Function to perform flood fill/BFS
def bfs(arr, start_index, visited):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    queue = deque([start_index])
    cluster = []
    
    while queue:
        current = queue.popleft()
        if current not in visited:
            visited.add(current)
            cluster.append(current)
            neighbors = find_neighbors(arr, current)
            queue.extend([n for n in neighbors if n not in visited])
    
    return cluster
  

def cluster(data, global_cluster_id=1):
    """Cluster data points for each event and assign globally unique cluster IDs."""
    # Dictionary to store cluster assignments directly
    cluster_dict = {}

    # Group data by 'Event' column and cluster each event
    for event_id, event_group in data.groupby('Event'):
        for idx_value, group in event_group.groupby('Overflow'): #to check for different detectors
            arr = group[['X', 'Y']].to_numpy()
            visited = set()
        
        # Process clusters using BFS (or a more efficient algorithm if possible)
            for i in range(len(group)):
                if i not in visited:
                    # Run BFS and get indices of the cluster
                    cluster = bfs(arr, i, visited)
                    
                    # Store indices directly in cluster_dict with the global cluster ID
                    indices = group.index[cluster]
                    for index in indices:
                        cluster_dict[index] = global_cluster_id
                    
                    # Increment global ID for the next cluster
                    global_cluster_id += 1

    return cluster_dict, global_cluster_id




def cluster_original(data, global_cluster_id=1):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    # First loop: process each event separately to generate clusters
    all_clusters = []
    for event_id, group in data.groupby('Event'):
        arr = group[['X0', 'Y0']].to_numpy()  
        visited = set()                     
        clusters = []
        
        # Find clusters using BFS (assumed to be a predefined function)
        for i in range(len(group)):
            if i not in visited:
                cluster = bfs(arr, i, visited)
                clusters.append((event_id, cluster))
        
        # Collect cluster information for the current group
        all_clusters.extend(clusters)
    
    # Second loop: assign globally unique cluster IDs without resetting
    cluster_dict = {}
    for event_id, cluster in all_clusters:
        for index in cluster:
            cluster_dict[data.loc[data['Event'] == event_id].index[index]] = global_cluster_id
        global_cluster_id += 1  # Increment global ID for the next cluster
    

    n_events = data['Event'].nunique()

    return cluster_dict, global_cluster_id


def plot_energy_spectra(outputFolder, data, total_observation_time, spectra_range, log=True, show=True, custom_name='', newFigure=True):
    """
    Plots the energy spectrum from the given data.

    Parameters:
        inputFolder (str): Path to the input folder (not used in this function but might be needed for context).
        outputFolder (str): Path to save the output plot.
        data (pd.DataFrame): Data containing 'Event' and 'ToT (keV)' columns.
        total_observation_time (float): Total observation time in seconds to normalize counts.
        spectra_range (float): Maximum energy range for the histogram.
        log (bool): If True, use a log scale for the y-axis.
        show (bool): If True, display the plot.
    """
    # Group the data by 'Event' and sum the 'ToT (keV)'
    events = data[['Event', 'ToT (keV)']].groupby('Event').sum()

    # Calculate the histogram values
    counts, bin_edges = np.histogram(events['ToT (keV)'], bins=round(spectra_range/2), range=(0, spectra_range))

    # Convert counts to counts per second
    counts_per_second = counts / total_observation_time

    # Plot the histogram values as a line
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers for the x values

    # Create a DataFrame for the histogram values
    histogram_df = pd.DataFrame({
        'Energy (keV)': bin_centers,
        'Counts per Second': counts_per_second
    })

    if newFigure:
        plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, counts_per_second, linestyle='-')  # Plot as a line with markers
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts per Second')
    plt.yticks(np.arange(0, max(counts_per_second)+1, max(counts_per_second)/10))
    #plt.title(f'Energy Spectrum - {energy_input_filename}')
    plt.title(f'Energy Spectrum {custom_name}')
    plt.grid(True)

    if log:
        plt.yscale('log')
        plt.ylabel('Counts per Second (Log)')

    plt.savefig(f'{outputFolder}/Energy_Spectrum_{custom_name}.png')
    plt.close()
    
    if show:
        plt.show()

    del data


def current_milli_time():
    return round(time.time() * 1000)


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

    return df.astype(valid_columns)


def parallelize_identifyEvent_Cluster(args):
    inputFile, event_length, outputFolder_csv = args
    
    global_cluster_id = 1
    # Step 1: Identify events by time interval
    identify_events_time(inputFile, event_length, 0, outputFolder_csv)

    # Step 2: Load and process the data
    abc = os.path.basename(inputFile).split('.')[0]
    data = pd.read_parquet(f'{outputFolder_csv}/df_sorted_{abc}.parquet')

    # Step 3: Cluster the events
    cluster_dict, global_cluster_id = cluster(data, global_cluster_id) #spatial cluster, ADD: check different detectors (z pos)
    data['Cluster'] = data.index.map(cluster_dict)
    #data['Cluster'] = data['Event'] #time cluster

    # Setp 4: Parameterize DF
    #data = df_parameterize(data)
    #print(data)
    # Step 5: Save the processed data
    output_path = f'{outputFolder_csv}/df_all_data_df_{abc}.parquet'

    data.to_parquet(output_path)

    # Cleanup the temporary sorted file
    os.remove(f'{outputFolder_csv}/df_sorted_{abc}.parquet')
    
    gc.collect() # collect garbage ...........



def identify_all(inputFolder, outputFolder, event_length, show=False):
    """
    Identifies events (within a coincidence time), performs clustering (identify how namy clusters an event has) from multiple input files while displaying a progress bar monitoring RAM usage.
    
    Parameters:
    - inputFolder (str): Path to the input folder containing data files.
    - event_length (int): Length of time intervals to identify events.
    
    Returns:
    - all_data_df (pd.DataFrame): DataFrame containing all processed data with clusters assigned.
    """
    
    calib = Calibration('','')
    input_files = calib.check_input_folders(inputFolder)
    
    #calibration_data = calib.load_calibration_constants(calibFolder)
    
    
    args = [(inputFile, event_length, outputFolder)
    for inputFile in input_files]

    with Pool() as pool:
        for _ in tqdm(pool.imap_unordered(parallelize_identifyEvent_Cluster,args), total=len(args), desc="Event ID & Clustering"):
            pass


def merge_parquet(input_file, output_file):
    """
    Merges a single .parquet file into an existing merged .parquet file.

    Args:
        input_file (str): Path to the .parquet file to merge.
        output_file (str): Path to the existing or new merged .parquet file.
    """
    if os.path.exists(output_file):
        # Load the existing merged file
        merged_df = pd.read_parquet(output_file)
    else:
        # Initialize an empty DataFrame if the output file doesn't exist
        merged_df = pd.DataFrame()

    # Load the new file and append to the merged DataFrame
    new_df = pd.read_parquet(input_file)
    merged_df = pd.concat([merged_df, new_df], ignore_index=True)

    # Save the updated merged file
    merged_df.to_parquet(output_file, index=False)


def check_flux(data, total_observation_time):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    number_of_events = data['Event'].iloc[-1]
    number_of_events_per_seconds = number_of_events/total_observation_time #total cnts/s
    counts_per_pixel = data['Matrix Index'].value_counts()
    counts_per_pixel_per_sec = counts_per_pixel/total_observation_time  #cnt/s

    return counts_per_pixel_per_sec.max()




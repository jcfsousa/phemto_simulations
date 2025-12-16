import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
from scipy.optimize import curve_fit
import calibration.calibration as calib
from alive_progress import alive_bar

from agnpy.utils.plot import load_mpl_rc
from concurrent.futures import ProcessPoolExecutor, as_completed

load_mpl_rc()

calib = calib.Calibration()

def identify_events_time(inputFile, event_length, max_event_index, calibration_data, *kwargs):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    df_sorted = calib.sort_df(inputFile, calibration_data)

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

    n_events = df_sorted['Event'].nunique()

    return df_sorted, max_event_index

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
    queue = [start_index]
    cluster = []
    
    while queue:
        current = queue.pop(0)
        if current not in visited:
            visited.add(current)
            cluster.append(current)
            neighbors = find_neighbors(arr, current)
            queue.extend([n for n in neighbors if n not in visited])
    
    return cluster
    
def cluster(data, global_cluster_id=1):
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
        arr = group[['X', 'Y']].to_numpy()  
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

def process_file(inputFile, event_length, max_event_index, calibration_data, global_cluster_id):
    """Process a single file and return the data with clusters."""
    # Identify events by time interval
    data, max_event_index = identify_events_time(inputFile, event_length, max_event_index, calibration_data)
    # Cluster the events using the modified cluster function
    cluster_dict, global_cluster_id = cluster(data, global_cluster_id)
    
    # Map the global cluster IDs back to the data
    data['Cluster'] = data.index.map(cluster_dict)

    return data, max_event_index, global_cluster_id

def identify_all(inputFolder, event_length, ProgressBar=True):
    """
    Identifies and clusters events from multiple input files while displaying a progress bar 
    and monitoring RAM usage.
    
    Parameters:
    - inputFolder (str): Path to the input folder containing data files.
    - event_length (int): Length of time intervals to identify events.
    
    Returns:
    - all_data_df (pd.DataFrame): DataFrame containing all processed data with clusters assigned.
    """
    all_data_df = pd.DataFrame()
    max_event_index = 0
    global_cluster_id = 1  # Initialize global cluster ID

    input_files = calib.check_input_folders(inputFolder)
    calibration_data = calib.load_calibration_constants()

    with alive_bar(len(input_files), bar='bubbles', spinner='arrows') as bar:
        # Create a list to hold futures
        futures = []
        results = [None] * len(input_files)  # Preallocate list for results

        with ProcessPoolExecutor() as executor:
            for index, inputFile in enumerate(input_files):
                future = executor.submit(process_file, inputFile, event_length, max_event_index, calibration_data, global_cluster_id)
                futures.append((future, index))

            # Collect results in the order of the input files
            for future, index in futures:
                try:
                    data, max_event_index, global_cluster_id = future.result()
                    results[index] = data
                except Exception as e:
                    print(f"Error processing file {input_files[index]}: {e}")
                    results[index] = pd.DataFrame()  # Handle error by returning an empty DataFrame

                bar()  # Update the progress bar

        # Concatenate results in the original order
        all_data_df = pd.concat(results, ignore_index=True)

    return all_data_df


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


def get_multiplicity_events(data, do_print=False):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    cluster_counts = data.groupby('Event')['Cluster'].nunique()

    single_events = cluster_counts[cluster_counts == 1].index.tolist()
    double_events = cluster_counts[cluster_counts == 2].index.tolist()
    multiple_events = cluster_counts[cluster_counts > 2].index.tolist()
    
    if do_print:
        return print(f"# Single Events: {len(single_events)}\n# Double Events: {len(double_events)}\n# Multiple Events: {len(multiple_events)}")
    else:
        return single_events, double_events, multiple_events

def plot_energy_spectra(data, total_observation_time):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    # Group the data by 'Event' and sum the 'ToT (keV)'
    events = data[['Event', 'ToT (keV)']].groupby('Event').sum()

    # Calculate the histogram values
    counts, bin_edges = np.histogram(events['ToT (keV)'], bins=100, range=(0, 800))

    # Convert counts to counts per second
    counts_per_second = counts / total_observation_time

    # Plot the histogram values as a line
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Calculate bin centers for the x values

    # Create a DataFrame for the histogram values
    histogram_df = pd.DataFrame({
        'Energy (keV)': bin_centers,
        'Counts per Second': counts_per_second
    })


    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, counts_per_second, linestyle='-')  # Plot as a line with markers
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts per Second')
    plt.yticks(np.arange(0, max(counts_per_second)+1, max(counts_per_second)/10))
    #plt.title(f'Energy Spectrum - {energy_input_filename}')
    plt.title(f'Energy Spectrum')
    plt.grid(True)

    plt.show()

    return

def counts_in_energy_peak(data, energy, do_print=False):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    # ------- Singles -------------
    single_cluster_events = get_multiplicity_events(data)[0]
    # Group by 'Event' and sum the 'ToT (keV)' values
    single_event_tot_sums = data[data['Event'].isin(single_cluster_events)].groupby('Event')['ToT (keV)'].sum()
    peak = (single_event_tot_sums > energy * (1 - calib.resolution(energy))) & (single_event_tot_sums < energy * (1 + calib.resolution(energy)))
    single_events_in_peak = single_event_tot_sums[peak].index.tolist()

    # ------- Doubles -------------
    double_cluster_events = get_multiplicity_events(data)[1]
    # Group by 'Event' and sum the 'ToT (keV)' values
    double_event_tot_sums = data[data['Event'].isin(double_cluster_events)].groupby('Event')['ToT (keV)'].sum()
    peak = (double_event_tot_sums > energy * (1 - calib.resolution(energy))) & (double_event_tot_sums < energy * (1 + calib.resolution(energy)))
    double_events_in_peak = double_event_tot_sums[peak].index.tolist()

    # ------- Multiples -------------
    multiple_cluster_events = get_multiplicity_events(data)[2]
    multiple_event_tot_sums = data[data['Event'].isin(multiple_cluster_events)].groupby('Event')['ToT (keV)'].sum()
    peak = (multiple_event_tot_sums > energy * (1 - calib.resolution(energy))) & (multiple_event_tot_sums < energy * (1 + calib.resolution(energy)))
    multiple_events_in_peak = multiple_event_tot_sums[peak].index.tolist()

    if do_print:
        return print(f"# Single cluster events in the peak: {len(single_events_in_peak)}\n# Double cluster events in the peak: {len(multiple_events_in_peak)}\n# Multiple events in the peak: {len(multiple_events_in_peak)}")
    else:
        return single_events_in_peak, double_events_in_peak, multiple_events_in_peak

def energies_data_frames_or_dict(data, dict = False):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """

    # Initialize dictionaries to store the E1 and E2 for each event
    E1 = {}
    E2 = {}

    # Iterate over each event
    for event, group_df in data:
        # Group by 'Cluster' and sum the 'Log_energy' for each cluster
        cluster_energy_sums = group_df.groupby('Cluster')['ToT (keV)'].sum()
        
        # Identify the highest and lowest summed energy clusters
        highest_energy_cluster = cluster_energy_sums.idxmax()
        lowest_energy_cluster = cluster_energy_sums.idxmin()

        highest_energy = cluster_energy_sums.max()
        lowest_energy = cluster_energy_sums.min()

        # Get the most energetic hit's coordinates for the highest and lowest energy clusters
        highest_energy_central_hit = group_df[group_df['Cluster'] == highest_energy_cluster].sort_values('ToT (keV)', ascending=False).iloc[0]
        lowest_energy_central_hit = group_df[group_df['Cluster'] == lowest_energy_cluster].sort_values('ToT (keV)', ascending=False).iloc[0]
        
        # Save these values in the dictionaries along with upper and lower limits
        E1[event] = {
            'E1': highest_energy,
            'upper': highest_energy * (1 + calib.resolution(highest_energy)),
            'lower': highest_energy * (1 - calib.resolution(highest_energy)),
            'resolution': calib.resolution(highest_energy),
            'x_peak': highest_energy_central_hit['X'],
            'y_peak': highest_energy_central_hit['Y']
        }

        E2[event] = {
            'E2': lowest_energy,
            'upper': lowest_energy * (1 + calib.resolution(lowest_energy)),
            'lower': lowest_energy * (1 - calib.resolution(lowest_energy)),
            'resolution': calib.resolution(lowest_energy),
            'x_peak': lowest_energy_central_hit['X'],
            'y_peak': lowest_energy_central_hit['Y']
        }

    # Convert dictionaries to dataframes for better readability if needed
    E1_df = pd.DataFrame.from_dict(E1, orient='index').reset_index().rename(columns={'index': 'Event'})
    E2_df = pd.DataFrame.from_dict(E2, orient='index').reset_index().rename(columns={'index': 'Event'})

    if dict:
        return E1, E2
    else:
        return E1_df, E2_df

def compton_photon(E0, theta):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    me = 511.0 #keV
    theta = np.radians(theta)
    cos_theta = np.cos(theta)
    E1 = E0 / (1 + (E0/me)*(1-cos_theta))
    return E1

def energy_electron(E0, theta):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    me = 511.0 # keV
    theta = np.radians(theta)
    num = (E0/me) * (1 - np.cos(theta))
    den = 1 + ((E0/me)*(1 - np.cos(theta)))
    elect_energy = E0 * num / den
    return elect_energy

def compton_data_frame(E1, E2, n_compton_events = False):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """    
    m0_c2 = 511.0 # keV

    # Initialize a dictionary to store theta for each event
    E_in = {}
    Compton_photon = {}
    Electron = {}

    # Iterate over each event to calculate theta
    for event in E1.keys():
        # Calculate the incident energy
        E_in[event] = E1[event]['E1'] + E2[event]['E2']
        Compton_photon[event] = compton_photon(E_in[event], 180)
        Electron[event] = energy_electron(E_in[event], 180)

    # Convert theta dictionary to a DataFrame for better readability
    energies_df = pd.DataFrame(list(E_in.items()), columns=['Event', 'E_in'])

    # Add the E1, E2 and E_in columns to the DataFrame
    energies_df['E1'] = energies_df['Event'].map(lambda event: E1[event]['E1'])
    energies_df['E1_upper'] = energies_df['Event'].map(lambda event: E1[event]['upper'])
    energies_df['E1_lower'] = energies_df['Event'].map(lambda event: E1[event]['lower'])
    energies_df['E1_res'] = energies_df['Event'].map(lambda event: E1[event]['resolution'])
    energies_df['E1_X'] = energies_df['Event'].map(lambda event: E1[event]['x_peak'])
    energies_df['E1_Y'] = energies_df['Event'].map(lambda event: E1[event]['y_peak'])


    energies_df['E2'] = energies_df['Event'].map(lambda event: E2[event]['E2'])
    energies_df['E2_upper'] = energies_df['Event'].map(lambda event: E2[event]['upper'])
    energies_df['E2_lower'] = energies_df['Event'].map(lambda event: E2[event]['lower'])
    energies_df['E2_res'] = energies_df['Event'].map(lambda event: E2[event]['resolution'])
    energies_df['E2_X'] = energies_df['Event'].map(lambda event: E2[event]['x_peak'])
    energies_df['E2_Y'] = energies_df['Event'].map(lambda event: E2[event]['y_peak'])


    energies_df['E_in'] = energies_df['Event'].map(E_in)
    energies_df['E_Compton_Photon'] = energies_df['Event'].map(Compton_photon)
    energies_df['E_Electron'] = energies_df['Event'].map(Electron)

    # Initialize a list to store results for all events and theta values
    comptons = []

    # Iterate over each event and theta to find matching conditions
    for event in E1.keys():
        for theta in range(181):
            E_photon = compton_photon(E_in[event], theta)
            E_elect = energy_electron(E_in[event], theta)

            photon_equals_E1 = 'no'
            electron_equals_E2 = 'no'
            photon_equals_E2 = 'no'
            electron_equals_E1 = 'no'

            if E_photon >= E1[event]['lower'] and E_photon <= E1[event]['upper']:
                photon_equals_E1 = 'yes'
                if E_elect >= E2[event]['lower'] and E_elect <= E2[event]['upper']:
                    electron_equals_E2 = 'yes'
            
            if E_photon >= E2[event]['lower'] and E_photon <= E2[event]['upper']:
                photon_equals_E2 = 'yes'
                if E_elect >= E1[event]['lower'] and E_elect <= E1[event]['upper']:
                    electron_equals_E1 = 'yes'
            
            # Append the result to the list only if conditions are met
            if (photon_equals_E1 == 'yes' and electron_equals_E2 == 'yes') or (photon_equals_E2 == 'yes' and electron_equals_E1 == 'yes'):
                if E2[event]['upper'] > E1[event]['lower']:
                    continue
                else:
                    comptons.append({
                        'Event': event,
                        'E_photon': E_photon,
                        'E_elect': E_elect,
                        'theta': theta,
                        'E_photon=E1': photon_equals_E1,
                        'E_elect=E2': electron_equals_E2,
                        'E_photon=E2': photon_equals_E2,
                        'E_elect=E1': electron_equals_E1,
                        'E1_upper': E1[event]['upper'],
                        'E1_lower': E1[event]['lower'],
                        'E2_upper': E2[event]['upper'],
                        'E2_lower': E2[event]['lower']
                    })

    # Convert the results list to a DataFrame
    comptons_df = pd.DataFrame(comptons)

    if n_compton_events: 
        return print("# Compton events in the peak:", comptons_df['Event'].nunique())
    else:
        return energies_df, comptons_df

def filter_circunference(data):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """    
    
    #no_center = data
    x_central_pixel = 686
    y_central_pixel = 92

    # Define the range conditions to exclude central pixels
    range = 0
    x_range = (x_central_pixel - range, x_central_pixel + range)
    y_range = (y_central_pixel - range, y_central_pixel + range)


    # Define the range to create circumference
    radius = 256 - x_central_pixel
    radius_squared = radius ** 2

    within_circumference = ((data['X'] - x_central_pixel) ** 2 + 
                            (data['Y'] - y_central_pixel) ** 2 > radius_squared)
            
    within_center = (data['X'] >= x_range[0]) & (data['X'] <= x_range[1]) & \
                    (data['Y'] >= y_range[0]) & (data['Y'] <= y_range[1])

        
    # Debug prints to check masks
    # print("Radius squared:", radius_squared)
    # print("Within circumference mask sum:", within_circumference.sum())
    # print("Within center mask sum:", within_center.sum())


    # Combine the masks to exclude rows where either condition is met
    final_mask = ~within_circumference & ~within_center

    # Apply the mask to filter the DataFrame
    filtered_circumference_df = data[final_mask]

    # Step 1: Count the number of clusters for each event
    cluster_counts = filtered_circumference_df.groupby('Event')['Cluster'].nunique()

    # Step 2: Identify events with only one cluster
    events_with_one_cluster = cluster_counts[cluster_counts == 1].index

    # Step 3: Filter out rows corresponding to these events
    filtered_df = filtered_circumference_df[~filtered_circumference_df['Event'].isin(events_with_one_cluster)]

    return filtered_df   

def no_duplicates(data1, data2):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """    
    comptons_event_index = data2['Event'].unique()
    compton_events_df = data1[data1['Event'].isin(comptons_event_index)]
    compton_events_df = compton_events_df.drop(columns=['E1_upper', 'E2_upper', 'E1_lower', 'E2_lower', 'E_Compton_Photon', 'E_Electron'])

    # Restrict all data to Compton Events
    #circ_comptons_all_data_df = all_data_df[0][all_data_df[0]['Event'].isin(comptons_event_index)]

    no_duplicates_comptons_df = data2.drop_duplicates(subset=['Event'], keep='first')

    compton_events_df['E_photon=E1'] = no_duplicates_comptons_df['E_photon=E1'].to_list()
    compton_events_df['E_elect=E2'] = no_duplicates_comptons_df['E_elect=E2'].to_list()
    compton_events_df['E_photon=E2'] = no_duplicates_comptons_df['E_photon=E2'].to_list()
    compton_events_df['E_elect=E1'] = no_duplicates_comptons_df['E_elect=E1'].to_list()

    return compton_events_df

def identify_compton(data, energy, filter = False):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    double_events_in_peak = counts_in_energy_peak(data, energy)[1]
    double_events_in_peak_df = data[data['Event'].isin(double_events_in_peak)]
    grouped_double_events_in_peak_df = double_events_in_peak_df.groupby('Event')

    E1, E2 = energies_data_frames_or_dict(grouped_double_events_in_peak_df, dict=True)[0], energies_data_frames_or_dict(grouped_double_events_in_peak_df, dict=True)[1]

    energies_df = compton_data_frame(E1, E2)[0]

    comptons_event_index = energies_df['Event'].unique()
    compton_events_df = energies_df[energies_df['Event'].isin(comptons_event_index)]
    compton_events_df = compton_events_df.drop(columns=['E1_upper', 'E2_upper', 'E1_lower', 'E2_lower', 'E_Compton_Photon', 'E_Electron'])

    # Restrict all data to Compton Events
    #comptons_all_data_df = data[data['Event'].isin(comptons_event_index)]

    final_compton_events_df = no_duplicates(energies_df, compton_events_df)
    
    if filter:
        
        grouped_filtered_by_circumference = filter_circunference(comptons_all_data_df).groupby('Event')

        E1, E2 = energies_data_frames_or_dict(grouped_filtered_by_circumference, dict=True)[0], energies_data_frames_or_dict(grouped_filtered_by_circumference, dict=True)[1]

        energies_df = compton_data_frame(E1, E2)[0]
        comptons_df = compton_data_frame(E1, E2)[1]

        final_compton_events_df = no_duplicates(energies_df, comptons_df)

        return final_compton_events_df
    else:
        return final_compton_events_df

def restrict_all_phi(data, x_center_pixel, y_center_pixel):    
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """    
    all_phi = []

    #central pixel

    for event, line in data.iterrows():
        if line['E_photon=E1'] == 'yes' and line['E_elect=E2'] == 'yes':

            if (line['E1_X'] - x_center_pixel) == 0 and (line['E1_Y'] - y_center_pixel) > 0:
                phi = math.pi/2 * (180/math.pi) 

                all_phi.append(phi)
                continue
            
            elif (line['E1_X'] - x_center_pixel) == 0 and (line['E1_Y'] - y_center_pixel) < 0:
                phi = 3 * (math.pi/2) * (180/math.pi)

                all_phi.append(phi)
                continue
            
            else:            
                #print('event: ', event)
                #arg = (line['E1_Y0'] - line['E2_Y0']) / (line['E1_X0'] - line['E2_X0'])
                phi = math.atan2(line['E1_Y'] - y_center_pixel , line['E1_X'] - x_center_pixel)

                phi_deg = math.degrees(phi)

                if phi_deg < 0:
                    phi_deg += 360
            
                all_phi.append(round(phi_deg,0))
        
        elif line['E_photon=E2'] == 'yes' and line['E_elect=E1'] == 'yes':

            if (line['E2_X'] - x_center_pixel) == 0 and (line['E2_Y'] - y_center_pixel) > 0:
                phi = math.pi/2 * (180/math.pi)

                all_phi.append(phi)
            
            elif (line['E2_X'] - x_center_pixel) == 0 and (line['E2_Y'] - y_center_pixel) < 0:
                phi = 3 * (math.pi/2) * (180/math.pi)

                all_phi.append(phi)

            else:
                #arg = (line['E2_Y0'] - line['E1_Y0']) / (line['E2_X0'] - line['E1_X0'])

                phi = math.atan2(line['E2_Y'] - y_center_pixel, line['E2_X'] - x_center_pixel) 

                phi_deg = math.degrees(phi)

                if phi_deg < 0:
                    phi_deg += 360
                    
                all_phi.append(round(phi_deg,0))
        
    data['Phi'] = all_phi

    return data

def convert_to_mm(data, distance, x_center_pixel, y_center_pixel, show = False):    
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """		
    # Convert pixel coordinates to millimeters
    data['E1_X_mm'] = data['E1_X'] * 0.055
    data['E1_Y_mm'] = data['E1_Y'] * 0.055
    data['E2_X_mm'] = data['E2_X'] * 0.055
    data['E2_Y_mm'] = data['E2_Y'] * 0.055

    # Convert central pixel coordinates to millimeters
    x_center_mm = x_center_pixel * 0.055
    y_center_mm = y_center_pixel * 0.055

    # Determine which coordinates correspond to the photon
    data['photon_X_mm'] = np.where(
        data['E_photon=E1'] == 'yes', data['E1_X_mm'], data['E2_X_mm']
    )
    data['photon_Y_mm'] = np.where(
        data['E_photon=E1'] == 'yes', data['E1_Y_mm'], data['E2_Y_mm']
    )

    # Calculate Euclidean distance between the central pixel and the photon cluster coordinates
    data['distance_mm'] = np.sqrt(
        (data['photon_X_mm'] - x_center_mm)**2 +
        (data['photon_Y_mm'] - y_center_mm)**2
    )
        
    final_filtered_df = data[data['distance_mm'] >= distance]
    #print(final_filtered_df)

    if show:
        plt.figure(figsize=(10, 10))

        # Plot all points
        plt.scatter(data['photon_X_mm'], data['photon_Y_mm'], color='grey')

        # Highlight points that are filtered based on distance
        plt.scatter(final_filtered_df['photon_X_mm'], final_filtered_df['photon_Y_mm'], color='green', label=f'Accepted events (distance >= {distance} mm)')

        # Plot the circle for reference
        circle = plt.Circle((x_center_mm, y_center_mm), radius=distance, color='red', fill=False, linestyle='--', label=f'Exclusion Radius ({distance} mm)')
        plt.gca().add_patch(circle)

        plt.xlabel('X-coordinate (mm)')
        plt.ylabel('Y-coordinate (mm)')
        plt.title('Compton Events Data with Filter Applied')
        plt.legend()
        plt.grid(True)

    return final_filtered_df

def bin_phi(data, x_center_pixel, y_center_pixel, distance):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """    
    data = restrict_all_phi(data, x_center_pixel, y_center_pixel)
    final_filtered_df = convert_to_mm(data, distance, x_center_pixel, y_center_pixel, show = False)
    phi_counts_binned_df = function_to_idk(final_filtered_df, energy=250, show=False)

    return phi_counts_binned_df

def save_df_to_csv(data):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """
    matrix_index_corrected = data['Matrix Index'] - 2*65536

    new_all_data = data
    new_all_data['Matrix Index'] = matrix_index_corrected

    columns_to_save = ['Matrix Index', 'ToA', 'ToT', 'FToA', 'Overflow']
    df_t3pa = new_all_data[columns_to_save]

    return df_t3pa.to_csv('300kev_10umx10um_F12_5s_0deg.t3pa', sep='\t')
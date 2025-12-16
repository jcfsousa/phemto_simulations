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

def no_duplicates(data1, data2, compton_events_df):
    """
    Cluster data points for each event and assign globally unique cluster IDs.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing the data to be clustered.
    - global_cluster_id (int): Initial global cluster ID to start with.

    Returns:
    - cluster_dict (dict): A dictionary mapping index to global cluster ID.
    - global_cluster_id (int): Updated global cluster ID after clustering.
    """  
    # Restrict all data to Compton Events
    #circ_comptons_all_data_df = all_data_df[0][all_data_df[0]['Event'].isin(comptons_event_index)]

    no_duplicates_comptons_df = data2.drop_duplicates(subset=['Event'], keep='first')

    compton_events_df['E_photon=E1'] = no_duplicates_comptons_df['E_photon=E1'].to_list()
    compton_events_df['E_elect=E2'] = no_duplicates_comptons_df['E_elect=E2'].to_list()
    compton_events_df['E_photon=E2'] = no_duplicates_comptons_df['E_photon=E2'].to_list()
    compton_events_df['E_elect=E1'] = no_duplicates_comptons_df['E_elect=E1'].to_list()
    
    return compton_events_df

def identify_compton(data, energy):
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
    comptons_df = compton_data_frame(E1, E2)[1]
    
    comptons_event_index = comptons_df['Event'].unique()
    compton_events_df = energies_df[energies_df['Event'].isin(comptons_event_index)]
    compton_events_df = compton_events_df.drop(columns=['E1_upper', 'E2_upper', 'E1_lower', 'E2_lower', 'E_Compton_Photon', 'E_Electron'])

    # Restrict all data to Compton Events
    comptons_all_data_df = data[data['Event'].isin(comptons_event_index)]

    final_compton_events_df = no_duplicates(energies_df, comptons_df, compton_events_df)    
    final_compton_events_df = compton_events_df
    
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

def function_to_idk(data, energy, show=False):
    to_beam_ref_frame = 90

    # Apply a 90-degree shift to the Phi values
    data['Phi_Beam_Ref'] = (data['Phi'] + to_beam_ref_frame) % 360

    angle_bin = 10

    # Bin the Phi values into 10-degree intervals
    bins = np.arange(-5, 356, angle_bin)  # Adjust bin edges for 10-degree intervals

    # Create labels for the bins as the center points
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Bin the Phi values and use bin centers as labels
    data['Phi_bin'] = pd.cut(data['Phi_Beam_Ref'], bins, right=False, labels=bin_centers)

    # Count the occurrences in each bin
    phi_counts_binned = data['Phi_bin'].value_counts().sort_index()

    # Convert to DataFrame and reset index
    phi_counts_binned_df = phi_counts_binned.reset_index()
    phi_counts_binned_df.columns = ['Phi_bin', 'Counts']

    # Get the mean of Counts
    total_counts_binned = phi_counts_binned_df['Counts'].sum()
    n_phis_binned = phi_counts_binned_df['Phi_bin'].nunique()
    mean_binned = total_counts_binned / n_phis_binned

    # Normalize the counts
    phi_counts_binned_df['Counts_Norm'] = phi_counts_binned_df['Counts'] / mean_binned

    # Calculate the errors for the normalized counts
    phi_counts_binned_df['Error'] = np.sqrt(phi_counts_binned_df['Counts']) / mean_binned

    if show:
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.errorbar(phi_counts_binned_df['Phi_bin'], phi_counts_binned_df['Counts_Norm'], yerr=phi_counts_binned_df['Error'], fmt='o', color='red', ecolor='black', capsize=3)
        plt.xlabel('Phi (Bin Center)')
        plt.ylabel('Counts Norm to Mean Counts')
        plt.title(f'{energy}kV Events Radial Bin Count ({angle_bin}° BinsCounts)')
        plt.xticks(rotation=0)
        plt.grid(True)
        plt.show()

    return phi_counts_binned_df

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
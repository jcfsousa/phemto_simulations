import matplotlib
import sys
import os, psutil
from matplotlib.ticker import AutoMinorLocator, LogLocator

import random
from os.path import split
import colorcet as cc
import re
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
from scipy.optimize import curve_fit
from aquisition.tpx_analysis3 import linear
from calibration.calibration import Calibration
from matplotlib.colors import LogNorm
from alive_progress import alive_bar
import manalysis.specLib as specLib
import manalysis.pathlib as pathlib
from multiprocessing import Pool
from tqdm import tqdm
import manalysis.polarizationfits as fits
from scipy.interpolate import interp1d
import gc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import ScalarFormatter
from pathlib import Path


m_ec2 = 511.0  # keV

def compton_photon_energy(E_in, theta_rad):
    """Scattered photon energy from Compton formula."""
    return E_in / (1.0 + (E_in / m_ec2) * (1.0 - np.cos(theta_rad)))

def compton_theta_from_energies(E_in, E_gamma):
    """Scatter angle (radians) from initial and scattered photon energies."""
    cos_theta = 1.0 - m_ec2 * (1.0 / E_gamma - 1.0 / E_in)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def compton_theta_derivative(E_in, E_gamma):
    """
    d(theta)/dE_gamma for error propagation
    """
    cos_theta = 1.0 - m_ec2 * (1.0 / E_gamma - 1.0 / E_in)
    if cos_theta <= -1.0 or cos_theta >= 1.0:
        return 0.0
    denom = E_gamma**2 * np.sqrt(1 - cos_theta**2)
    if denom == 0:
        return 0.0
    return m_ec2 / denom

def chi2_energy_compton(E_in, E1_meas, E2_meas,
                        sigma1, sigma2,
                        theta_geom, photon_is_E1=True):
    """
    Energy chi^2: compare measured E1/E2 with Compton prediction at theta_geom.
    """
    if photon_is_E1:
        E_gamma_meas = E1_meas
        E_e_meas     = E2_meas
        sig_gamma    = sigma1
        sig_e        = sigma2
    else:
        E_gamma_meas = E2_meas
        E_e_meas     = E1_meas
        sig_gamma    = sigma2
        sig_e        = sigma1

    # Predicted photon/electron energies
    # TODO: need to add the sum of simas, E_in also has a sigma
    E_gamma_pred = compton_photon_energy(E_in, theta_geom)
    E_e_pred     = E_in - E_gamma_pred

    chi2_gamma = ((E_gamma_meas - E_gamma_pred) / sig_gamma) ** 2
    chi2_e     = ((E_e_meas     - E_e_pred)     / sig_e    ) ** 2

    return chi2_gamma + chi2_e


def chi2_geom(E_in, E1_meas, E2_meas,
              sigma1, sigma2,
              theta_geom, sigma_theta_deg,
              photon_is_E1=True):
    """
    Geometric chi^2: compare theta from energies vs geometry.
    """
    if photon_is_E1:
        E_gamma = E1_meas
        sigma_E_gamma = sigma1
    else:
        E_gamma = E2_meas
        sigma_E_gamma = sigma2

    theta_kin = compton_theta_from_energies(E_in, E_gamma)      # rad
    theta_kin_deg   = np.degrees(theta_kin)

    #Error prop of theta computation with measured Energies
    dtheta_dE = compton_theta_derivative(E_in, E_gamma)
    sigma_theta_kin = abs(dtheta_dE) * sigma_E_gamma  # rad

    sigma_theta_total = np.degrees(np.sqrt(sigma_theta_kin**2 + np.radians(sigma_theta_deg)**2))
    theta_geom_deg  = np.degrees(theta_geom)

    delta = theta_kin_deg - theta_geom_deg
    return (delta / sigma_theta_total) ** 2


def fom_compton(E_in, E1_meas, E2_meas,
                sigma1, sigma2,
                theta_geom, sigma_theta_deg,
                photon_is_E1=True):
    """
    Total FOM = chi2_energy + chi2_geom.
    """
    chi2_E = chi2_energy_compton(E_in, E1_meas, E2_meas,
                                 sigma1, sigma2,
                                 theta_geom, photon_is_E1)
    chi2_G = chi2_geom(E_in, E1_meas, E2_meas,
                       sigma1, sigma2,
                       theta_geom, sigma_theta_deg,
                       photon_is_E1)

    return chi2_E + chi2_G

def select_compton_events(df, sigma_theta_deg, fom_max=9.0):
    selected = []

    for _, row in df.iterrows():
        E_in   = row['E1'] + row['E2']
        E1_meas = row['E1']
        E2_meas = row['E2']
        theta_geom = row['theta_geom']
        sigma1 = row['E1_sigma']
        sigma2 = row['E2_sigma']

        # Hypothesis 1: E1 = photon
        fom_h1 = fom_compton(E_in, E1_meas, E2_meas,
                             sigma1, sigma2,
                             theta_geom, sigma_theta_deg,
                             photon_is_E1=True)

        # Hypothesis 2: E2 = photon
        fom_h2 = fom_compton(E_in, E1_meas, E2_meas,
                             sigma1, sigma2,
                             theta_geom, sigma_theta_deg,
                             photon_is_E1=False)

        if fom_h1 < fom_h2:
            fom_best = fom_h1
            photon_is_E1 = True
        else:
            fom_best = fom_h2
            photon_is_E1 = False

        if fom_best > fom_max:
            continue

        if photon_is_E1:
            E_photon = E1_meas
            E_photon_X0 = row['E1_x_peak']
            E_photon_Y0 = row['E1_y_peak']
            E_photon_Overflow = row['E1_Overflow']
            E_photon_sigma = row['E1_sigma']
            E_elect  = E2_meas
            E_elect_X0 = row['E2_x_peak']
            E_elect_Y0 = row['E2_y_peak']
            E_elect_Overflow = row['E2_Overflow']
            E_elect_sigma = row['E2_sigma']
        else:
            E_photon = E2_meas
            E_photon_X0 = row['E2_x_peak']
            E_photon_Y0 = row['E2_y_peak']
            E_photon_Overflow = row['E2_Overflow']
            E_photon_sigma = row['E2_sigma']
            E_elect  = E1_meas
            E_elect_X0 = row['E1_x_peak']
            E_elect_Y0 = row['E1_y_peak']
            E_elect_Overflow = row['E1_Overflow']
            E_elect_sigma = row['E1_sigma']

        theta_kin = compton_theta_from_energies(E_in, E_photon)

        selected.append({
            'Event': row['Event'],
            'E_in': E_in,
            'E_photon': E_photon,
            'E_photon_sigma': E_photon_sigma,
            'E_photon_X0': E_photon_X0,
            'E_photon_Y0': E_photon_Y0,
            'E_photon_overflow': E_photon_Overflow,
            'E_elect': E_elect,
            'E_elect_sigma': E_elect_sigma,
            'E_elect_X0': E_elect_X0,
            'E_elect_Y0': E_elect_Y0,
            'E_elect_overflow': E_elect_Overflow,
            'theta_geom': theta_geom,
            'theta_kin': theta_kin,
            'FOM': fom_best
        })

    return pd.DataFrame(selected)



def debug_plots(df_sel, outputFolder, custom_name):
    """
    df_sel is the output of select_compton_events().
    Produces several sanity‑check plots.
    """
    # Convert angles
    theta_geom_deg = np.degrees(df_sel['theta_geom'].values)
    theta_kin_deg  = np.degrees(df_sel['theta_kin'].values)
    delta_theta    = theta_kin_deg - theta_geom_deg

    FOM = df_sel['FOM'].values

    plt.figure(figsize=(12, 10))

    # 1. FOM distribution
    ax1 = plt.subplot(2, 2, 1)
    ax1.hist(FOM, bins=100, histtype='step', color='k')
    ax1.set_xlabel('FOM = chi2_energy + chi2_geom')
    ax1.set_ylabel('Counts')
    ax1.set_title('FOM distribution')

    # 2. θ_kin vs θ_geom
    try:
        ax2 = plt.subplot(2, 2, 2)
        ax2.hist2d(theta_geom_deg, theta_kin_deg,
                   bins=80, cmap='jet')
        ax2.plot([0, 180], [0, 180], 'r--', lw=1)
        ax2.set_xlabel('theta_geom (deg)')
        ax2.set_ylabel('theta_kin (deg)')
        ax2.set_title('theta from geometry vs theta from energy')
        plt.colorbar(ax2.collections[0], ax=ax2, label='Counts')
    except Exception as e:
        print('ERROR in Compton debug plot, ERROR: {e}')

    # 3. Δθ = θ_kin − θ_geom
    ax3 = plt.subplot(2, 2, 3)
    ax3.hist(delta_theta, bins=100, histtype='step', color='b')
    ax3.set_xlabel('delta theta = theta_kin − theta_geom (deg)')
    ax3.set_ylabel('Counts')
    ax3.set_title('Angular residual')

    # 4. E_photon vs θ_geom
    ax4 = plt.subplot(2, 2, 4)
    sc = ax4.scatter(theta_kin_deg, df_sel['E_photon'].values,
                     c=FOM, s=5, cmap='plasma', alpha=0.7)
    ax4.set_xlabel('theta_kin (deg)')
    ax4.set_ylabel('E_photon (keV)')
    ax4.set_title('E_photon vs theta_kin (colored by FOM)')
    plt.colorbar(sc, ax=ax4, label='FOM')

    plt.tight_layout()
    plt.savefig(f"{outputFolder}/compton_goodness_plot_{custom_name}.png")
    plt.close()







def detect_sudden_changes(data, threshold=2.0):
    """
    Detects sudden changes in a 1D array using the first derivative.
    
    Parameters:
        data (array-like): 1D array containing energy counts (e.g., x_energy_deposited).
        threshold (float): Threshold for detecting a significant change (in standard deviations).
        
    Returns:
        List of indices where sudden changes occur.
    """
    # Calculate the first derivative (rate of change)
    diff = np.diff(data)
    #print(diff)

    std_dev = np.std(diff)
    #print(std_dev)

    normalized_diff = diff / std_dev if std_dev != 0 else diff
    #print(normalized_diff)

    sudden_changes = np.where(np.abs(normalized_diff) > threshold)[0]


    return sudden_changes.tolist()

def get_multiplicity_events(data):
    """

    """
    #print(data)

    counts = data.groupby('Event')['Cluster'].nunique()
    #print(counts)
    single_event_ids = counts[counts == 1].index
    double_event_ids = counts[counts == 2].index
    multiple_event_ids = counts[counts > 2].index

    mask_singles = data['Event'].isin(single_event_ids)
    data_singles = data[mask_singles]
    mask_double = data['Event'].isin(double_event_ids)
    data_double = data[mask_double]
    mask_multiple = data['Event'].isin(multiple_event_ids)
    data_multiple = data[mask_multiple]

    n_events_singles = data_singles['Event'].nunique()
    n_events_double = data_double['Event'].nunique()
    n_events_multiple = data_multiple['Event'].nunique()
    

    #hits_per_event = data.groupby('Event').size()  # Series: index=Event, value=Nhits
    #single_events  = hits_per_event[hits_per_event == 1].index.tolist()
    #double_events  = hits_per_event[hits_per_event == 2].index.tolist()
    #multiple_events  = hits_per_event[hits_per_event > 2].index.tolist()


    #cluster_counts = data.groupby('Event')['Cluster'].nunique()

    #single_events = cluster_counts[cluster_counts == 1].index.tolist()
    #double_events = cluster_counts[cluster_counts == 2].index.tolist()
    #multiple_events = cluster_counts[cluster_counts > 2].index.tolist()
    
    #with open(f'{outputFolder}/Events.txt', 'w') as f:
    #    f.write(f"# Single Events: {len(single_events)}\n# Double Events: {len(double_events)}\n# Multiple Events: {len(multiple_events)}")    
    #print(f"# Single Events: {n_events_singles}\n# Double Events: {n_events_double}\n# Multiple Events: {n_events_multiple}")
    return data_singles, data_double, data_multiple

def counts_in_energy_peak(data, energy):
    """
    Identifies the multiplicity of events within the energy peak given by the user. It returns the event ID of each type of event (single, double and multiple).

    """
    calib = Calibration('', '')
    
    # ------- Singles -------------
    single_cluster_events = get_multiplicity_events(data)[0]
    
    # Group by 'Event' and sum the 'ToT (keV)' values
    single_event_tot_sums = single_cluster_events.groupby('Event')['ToT (keV)'].sum()
    peak = (single_event_tot_sums > energy * (1 - calib.resolution(energy))) & (single_event_tot_sums < energy * (1 + calib.resolution(energy)))
    single_events_in_peak = single_event_tot_sums[peak].index.tolist()

    # ------- Doubles -------------
    double_cluster_events = get_multiplicity_events(data)[1]
    # Group by 'Event' and sum the 'ToT (keV)' values
    double_event_tot_sums = double_cluster_events.groupby('Event')['ToT (keV)'].sum()

    ### to use with the Ba133 point source to check the residual modulation over the energy range 81, 356kev
    #peak = (double_event_tot_sums > 81 * (1 - calib.resolution(81))) & (double_event_tot_sums < 356 * (1 + calib.resolution(356)))

    peak = (double_event_tot_sums > energy * (1 - calib.resolution(energy))) & (double_event_tot_sums < energy * (1 + calib.resolution(energy)))
    double_events_in_peak = double_event_tot_sums[peak].index.tolist()


    # ------- Multiples -------------
    multiple_cluster_events = get_multiplicity_events(data)[2]
    multiple_event_tot_sums = multiple_cluster_events.groupby('Event')['ToT (keV)'].sum()
    peak = (multiple_event_tot_sums > energy * (1 - calib.resolution(energy))) & (multiple_event_tot_sums < energy * (1 + calib.resolution(energy)))
    multiple_events_in_peak = multiple_event_tot_sums[peak].index.tolist()

    #return single_events_in_peak, double_events_in_peak, multiple_events_in_peak
    return single_events_in_peak, double_events_in_peak, multiple_events_in_peak


def barycenter(cluster_df, overflow, cdte_detSize, cdte_pixSize, si_detSize, si_pixSize):
    cluster_energy = cluster_df['ToT (keV)'].sum()   
 
    n_collumns_df = len(cluster_df.axes[1])
    
    cluster_df.insert(n_collumns_df, 'X_wheight', (cluster_df['X']*cluster_df['ToT (keV)'])/cluster_energy)
    cluster_df.insert(n_collumns_df, 'Y_wheight', (cluster_df['Y']*cluster_df['ToT (keV)'])/cluster_energy)

    x_barycenter = cluster_df['X_wheight'].sum()
    y_barycenter = cluster_df['Y_wheight'].sum()
    
    # TODO: change this, read from config, get global variables of the detectors
    if overflow == 0: #Si detector
        x_barycenter = ((x_barycenter+1) * si_pixSize) - (si_detSize/2)
        y_barycenter = ((y_barycenter+1) * si_pixSize) - (si_detSize/2)
    elif overflow == 1: #CdTe detector
        x_barycenter = ((x_barycenter+1) * cdte_pixSize) - (cdte_detSize/2)
        y_barycenter = ((y_barycenter+1) * cdte_pixSize) - (cdte_detSize/2)
    else:
        x_barycenter = ((x_barycenter+1) * 0.0055)
        y_barycenter = ((y_barycenter+1) * 0.0055)
        return x_barycenter, y_barycenter
    return x_barycenter, y_barycenter

def apply_barycenter_formatData(data, z_cdte, cdte_detSize, cdte_pixSize, z_si, si_detSize, si_pixSize):
    """
    This function determines the energy of each cluster, E1 and E2. It uses the calibration curves y=ax+b to calibrate the clusters.
        E1 - Most energetic cluster 
        E2 - Least energenic cluster  
    The function also tries to determine the pixel where the interaction happened via barycenter using energy.

    """
    
    calib = Calibration('', '')
    # Initialize dictionaries to store the E1 and E2 for each event
    E1 = {}
    E2 = {}
    
    #print(data)
    # Iterate over each event
    for event, group_df in data:
        # Group by 'Cluster' and sum the 'Log_energy' for each cluster

        #cluster_energy_sums = group_df.groupby('Cluster')['ToT (keV)'].sum()
        #print(event)
        #print(group_df)
        #to calibrate the clusters
        cluster_energy_sums = specLib.cluster_energy_calib(group_df, show_individual_clusters = True)
        

        def extract_cluster_id(cluster_idx):
            if isinstance(cluster_idx, tuple):
                return int(cluster_idx[1])  # adjust [1] to match the position of 'Cluster' in your MultiIndex
            return int(cluster_idx)

        def extract_cluster_overflow(cluster_idx):
            if isinstance(cluster_idx, tuple):
                return int(cluster_idx[-1])  # adjust [1] to match the position of 'Cluster' in your MultiIndex
            return int(cluster_idx)
        
        # Identify the highest and lowest summed energy clusters
        highest_energy_cluster = cluster_energy_sums.idxmax()
        lowest_energy_cluster = cluster_energy_sums.idxmin()

        highest_id = extract_cluster_id(highest_energy_cluster)
        lowest_id  = extract_cluster_id(lowest_energy_cluster)

        highest_overflow = extract_cluster_overflow(highest_energy_cluster)
        lowest_overflow = extract_cluster_overflow(lowest_energy_cluster)

        highest_energy = cluster_energy_sums.max()
        lowest_energy = cluster_energy_sums.min()
    
        # Use the barycenter keV method to estimate the origin of the elctron (compton, photoelectric)
        data_cluster_high_energy = group_df[group_df['Cluster'] == highest_id]
        x_highest_energy_cluster, y_highest_energy_cluster = barycenter(data_cluster_high_energy, highest_overflow, cdte_detSize, cdte_pixSize, si_detSize, si_pixSize) #tranforms to cm

        data_cluster_low__energy = group_df[group_df['Cluster'] == lowest_id]
        x_lowest_energy_cluster, y_lowest_energy_cluster = barycenter(data_cluster_low__energy, lowest_overflow, cdte_detSize, cdte_pixSize, si_detSize, si_pixSize) #trnsforms to cm

        
        # Save these values in the dictionaries along with upper and lower limits
        E1[event] = {
            'E1': highest_energy,
            'upper': highest_energy * (1 + calib.resolution(highest_energy)),
            'lower': highest_energy * (1 - calib.resolution(highest_energy)),
            'resolution': calib.resolution(highest_energy),
            'x_peak': round(x_highest_energy_cluster,3),
            'y_peak': round(y_highest_energy_cluster,3),
            'Overflow': highest_overflow
        }

        E2[event] = {
            'E2': lowest_energy,
            'upper': lowest_energy * (1 + calib.resolution(lowest_energy)),
            'lower': lowest_energy * (1 - calib.resolution(lowest_energy)),
            'resolution': calib.resolution(lowest_energy),
            'x_peak': round(x_lowest_energy_cluster, 3),
            'y_peak': round(y_lowest_energy_cluster, 3),
            'Overflow': lowest_overflow
        }
        
    rows = []
    for event, e1_info in E1.items():
        e2_info = E2[event]
        row = {
            'Event': event,
            # E1 info
            'E1':          e1_info['E1'],
            'E1_upper':    e1_info['upper'],
            'E1_lower':    e1_info['lower'],
            'E1_resolution_fwhm': e1_info['resolution'],
            'E1_sigma':    e1_info['E1'] * e1_info['resolution'] / 2.355,  # if resolution is relative
            'E1_x_peak':   e1_info['x_peak'],
            'E1_y_peak':   e1_info['y_peak'],
            'E1_Overflow': e1_info['Overflow'],
            # E2 info
            'E2':          e2_info['E2'],
            'E2_upper':    e2_info['upper'],
            'E2_lower':    e2_info['lower'],
            'E2_resolution_fwhm': e2_info['resolution'],
            'E2_sigma':    e2_info['E2'] * e2_info['resolution'] / 2.355,  # same assumption
            'E2_x_peak':   e2_info['x_peak'],
            'E2_y_peak':   e2_info['y_peak'],
            'E2_Overflow': e2_info['Overflow'],
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    
    # TODO: give compute_theta_geom_from_row() the z_cdte and z_si according to config file
    df['theta_geom'] = df.apply(compute_theta_geom_from_row,
                                args=(z_cdte, z_si), 
                                axis=1)
    
    return df

def compton_photon(E0, theta):
    """
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

def compton_data_frame(E1, E2, filter = False):
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
        #Compton_photon[event] = compton_photon(E_in[event], 180)
        #Electron[event] = energy_electron(E_in[event], 180)

    # Convert theta dictionary to a DataFrame for better readability
    energies_df = pd.DataFrame(list(E_in.items()), columns=['Event', 'E_in'])

    # Add the E1, E2 and E_in columns to the DataFrame
    energies_df['E1'] = energies_df['Event'].map(lambda event: E1[event]['E1'])
    energies_df['E1_upper'] = energies_df['Event'].map(lambda event: E1[event]['upper'])
    energies_df['E1_lower'] = energies_df['Event'].map(lambda event: E1[event]['lower'])
    energies_df['E1_res'] = energies_df['Event'].map(lambda event: E1[event]['resolution'])
    energies_df['E1_X0'] = energies_df['Event'].map(lambda event: E1[event]['x_peak'])
    energies_df['E1_Y0'] = energies_df['Event'].map(lambda event: E1[event]['y_peak'])
    energies_df['E1_Overflow'] = energies_df['Event'].map(lambda event: E1[event]['Overflow'])


    energies_df['E2'] = energies_df['Event'].map(lambda event: E2[event]['E2'])
    energies_df['E2_upper'] = energies_df['Event'].map(lambda event: E2[event]['upper'])
    energies_df['E2_lower'] = energies_df['Event'].map(lambda event: E2[event]['lower'])
    energies_df['E2_res'] = energies_df['Event'].map(lambda event: E2[event]['resolution'])
    energies_df['E2_X0'] = energies_df['Event'].map(lambda event: E2[event]['x_peak'])
    energies_df['E2_Y0'] = energies_df['Event'].map(lambda event: E2[event]['y_peak'])
    energies_df['E2_Overflow'] = energies_df['Event'].map(lambda event: E2[event]['Overflow'])


    energies_df['E_in'] = energies_df['Event'].map(E_in) 
    #energies_df['E_Compton_Photon'] = energies_df['Event'].map(Compton_photon) #this can be removed (ze)
    #energies_df['E_Electron'] = energies_df['Event'].map(Electron) #this can be removed (ze)

    #if filter:
    #    energies_df.to_parquet(f'{outputFolder}/parquet/energies_df_filtered_{abc}.parquet')
    #else:
    #    energies_df.to_parquet(f'{outputFolder}/parquet/energies_df_{abc}.parquet')

    # Initialize a list to store for all events and theta values
    comptons = []
    theta_computed_dict = {}

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
            ## Here we should apply the filter of the circumference. Reject comptons with an hypotnuse less than x mm (to reject the adjacent pixels to the hit pixels) and reject comptons with an hypotnuse distance greater than y mm. This y mm should be the shorter distance between the compton electrons and any adjacent physical limit of the detector. 

                if E2[event]['upper'] > E1[event]['lower']:
                    continue
 
                if (photon_equals_E1 == 'yes' and electron_equals_E2 == 'yes'):
                    theta_computed = compute_theta_compton(E1[event]['E1'], E2[event]['E2'])
                    theta_computed_dict[event] = theta_computed
                    #compute the theta with formula and error propagation, E1 is photon and E2 is electron
                if (photon_equals_E2 == 'yes' and electron_equals_E1 == 'yes'):
                    theta_computed = compute_theta_compton(E2[event]['E2'], E1[event]['E1'])
                    theta_computed_dict[event] = theta_computed
                    #compute the theta with formula and error, E1 is electron and E2 is photon
                
                comptons.append({
                    'Event': event,
                    'E_photon': E_photon,
                    'E_elect': E_elect,
                    'theta_computed': theta_computed,
                    'E_photon=E1': photon_equals_E1,
                    'E_elect=E2': electron_equals_E2,
                    'E_photon=E2': photon_equals_E2,
                    'E_elect=E1': electron_equals_E1,
                    'E1_upper': E1[event]['upper'],
                    'E1_lower': E1[event]['lower'],
                    'E2_upper': E2[event]['upper'],
                    'E2_lower': E2[event]['lower']
                })

                continue

    energies_df['theta_computed'] = energies_df['Event'].map(theta_computed_dict)

    # Convert the list to a DataFrame
    comptons_df = pd.DataFrame(comptons)

    #if filter:
    #    comptons_df.to_parquet(f'{outputFolder}/parquet/comptons_df_filtered_{abc}.parquet')
    
    #else:
    #    comptons_df.to_parquet(f'{outputFolder}/parquet/comptons_df_{abc}.parquet')
    

    return comptons_df, energies_df

def compute_theta_compton(E_ph, E_elec):
    mc2 = 511 #kev/c²
    try:
        theta = math.acos((-mc2/(E_ph + E_elec))*(((E_ph + E_elec)/E_ph)-1) + 1)
    except:
        theta = 0
    #need to do error propagation for Eph and Eelec to be fully correct, 14/08/2025

    return theta

def filter_circumference(data, outputFile, outputFolder, x_center_pixel, y_center_pixel, max_dist):

    # Define the range to create circumference
    radius = 246 - x_center_pixel
    #radius = 256 - x_center_pixel
    radius_squared = radius ** 2

    within_circumference = ((data['X0'] - x_center_pixel) ** 2 + 
                            (data['Y0'] - y_center_pixel) ** 2 > radius_squared)

    final_mask = ~within_circumference 

    # Apply the mask to filter the DataFrame
    filtered_circumference_df = data[final_mask]
    
    # Step 1: Count the number of clusters for each event
    cluster_counts = filtered_circumference_df.groupby('Event')['Cluster'].nunique()

    # Step 2: Identify events with only one cluster
    events_with_one_cluster = cluster_counts[cluster_counts == 1].index

    # Step 3: Filter out rows corresponding to these events
    filtered_df = filtered_circumference_df[~filtered_circumference_df['Event'].isin(events_with_one_cluster)]

    abc = os.path.basename(outputFile).split('.')[0]
    #filtered_df.to_parquet(f'{outputFolder}/parquet/filtered_circumference_df_{abc}.parquet', index = False)
    
   
    return filtered_df

def plot_filter_circumference(outputFolder, x_center_pixel, y_center_pixel):
    
    df_list = [pd.read_parquet(os.path.join(f'{outputFolder}/parquet/', filename), columns = ['Event','X0','Y0']) for filename in os.listdir(f'{outputFolder}/parquet/') if filename.startswith('comptons_') and filename.endswith(".parquet")]
    
    # Concatenate all DataFrames in the list into a single DataFrame
    data = pd.concat(df_list, ignore_index=True)

    #data = concatenated_df.groupby('Phi_bin', as_index=False).sum()
    
    # Define the range to create circumference
    radius = 246 - x_center_pixel
    #radius = 256 - x_center_pixel
    radius_squared = radius ** 2

    within_circumference = ((data['X0'] - x_center_pixel) ** 2 + 
                            (data['Y0'] - y_center_pixel) ** 2 > radius_squared)
            

    # Plot the points to visualize the masks
    plt.figure(figsize=(10, 10))

    # Plot all points
    plt.scatter(data['X0'], data['Y0'], color='grey', label='All points')

    # Highlight points within the circumference
    plt.scatter(data[within_circumference]['X0'], 
                data[within_circumference]['Y0'], 
                color='blue', label='Events outside circumference')

    final_mask = ~within_circumference 

    # Apply the mask to filter the DataFrame
    filtered_circumference_df = data[final_mask]

    # Highlight filtered points
    plt.scatter(filtered_circumference_df['X0'], filtered_circumference_df['Y0'], 
                color='green', label='Accepted events')

    # Plot the circle for reference
    circle = plt.Circle((x_center_pixel, y_center_pixel), radius, color='red', fill=False, linestyle='--', label = 'Circumference')
    plt.gca().add_patch(circle)

    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title('Filter Applied to Compton Events Data')
    plt.legend()
    plt.savefig(f'{outputFolder}/filter_circumference')
    plt.close()

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


def compute_theta_geom_from_row(row, z_cdte=-5.0, z_si=0.0):
    """
    Compute geometric Compton scatter angle theta_geom (radians)
    from a DataFrame row with E1/E2 peak positions and Overflow flags.

    z_cdte:  z coordinate (cm) when Overflow == 1
    z_si:    z coordinate (cm) when Overflow == 0
    """

    # E1 position
    z1 = z_cdte if row['E1_Overflow'] == 1 else z_si
    r1 = np.array([row['E1_x_peak'], row['E1_y_peak'], z1], dtype=float)

    # E2 position
    z2 = z_cdte if row['E2_Overflow'] == 1 else z_si
    r2 = np.array([row['E2_x_peak'], row['E2_y_peak'], z2], dtype=float)

    # Define incoming photon direction.
    # Here: photon comes along +z from "infinity" toward the first interaction,
    # so direction of incoming photon at E1 is approx (0, 0, 1).
    k_in = np.array([0.0, 0.0, 1.0])

    # Scattered photon direction: from first interaction to second interaction.
    # Need to know which is first: here assume E1 is first, E2 is second.
    k_sc = r2 - r1
    norm_sc = np.linalg.norm(k_sc)
    if norm_sc == 0:
        return np.nan  # degenerate case
    k_sc /= norm_sc

    # Angle between k_in and k_sc: cos(theta) = k_in · k_sc
    cos_theta = np.dot(k_in, k_sc)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta_geom = np.arccos(cos_theta)

    return theta_geom

def process_comptons(args):
    outputFolder, peak_energy, outputFile, z_cdte, cdte_detSize, cdte_pixSize, z_si, si_detSize, si_pixSize  = args
    
    try:
        hold = os.path.basename(outputFile).split('.')[0]
        mybeer = hold.split('__')
        abc = mybeer[1]
    except:
        hold = os.path.basename(outputFile).split('.')[0]
        mybeer = hold.split('_')
        abc = mybeer[-1]

    outputFolder_parquet = os.path.join(outputFolder, 'parquet')
    
    try:
        data = pd.read_parquet(f'{outputFolder_parquet}/df_all_data_df__{abc}.parquet', columns = ['Event', 'Cluster', 'ToT (keV)', 'X', 'Y','Overflow'])
    except:
        data = pd.read_parquet(f'{outputFolder_parquet}/df_all_data_df_{abc}.parquet', columns = ['Event', 'Cluster', 'ToT (keV)', 'X', 'Y','Overflow'])
    #chip_id = specLib.get_chip_id(chip) 
    

    ## Getting only Double Events
    all_double_events_dfs = []
    energy = float(peak_energy)
    double_events_in_peak = counts_in_energy_peak(data, energy)[1]
    double_events_in_peak_df = data[data['Event'].isin(double_events_in_peak)]
    all_double_events_dfs.append(double_events_in_peak_df)

    # Saving double events in directory
    double_events_in_peak_df = pd.concat(all_double_events_dfs, ignore_index=True)
    outputFolder_parquet_doubles = os.path.join(outputFolder_parquet, f'doubles')
    outputFolder_parquet_doublesPeak = os.path.join(outputFolder_parquet_doubles, 'inPeak')
    pathlib.creat_dir(outputFolder_parquet_doublesPeak)
    double_events_in_peak_df.to_parquet(f'{outputFolder_parquet_doublesPeak}/double_events_in_peak_df_{abc}.parquet')

    grouped_double_events_in_peak_df = double_events_in_peak_df.groupby('Event')
    del double_events_in_peak_df

    # Computing x,y barycenter, Assign E1 the event with max energy, E2 the event with the min energy
    df = apply_barycenter_formatData(grouped_double_events_in_peak_df, z_cdte, cdte_detSize, cdte_pixSize, z_si, si_detSize, si_pixSize)

    ######### Compton Selection part #########
    sigma_theta_deg = 1.0   # geometrical x,y sigma position resolution
    # Can use very large fom_max to take into consideration all comptons
    fom_max = 9     # ch2_geometry + chi2_energy,  >9 is 3sigma disagreement.... 

    df_comptons = select_compton_events(df,
                                        sigma_theta_deg=sigma_theta_deg,
                                        fom_max=fom_max)

    #print("Selected events:", len(df_comptons), "out of", len(df))
    #print(df_comptons)

    # Saving compton events in directory for later use
    outputFolder_parquet_doublesPeakCompton = os.path.join(outputFolder_parquet_doublesPeak, 'comptons')
    pathlib.creat_dir(outputFolder_parquet_doublesPeakCompton)
    outputFolder_comptonPlots = f'{outputFolder}/compton_goodness'
    pathlib.creat_dir(outputFolder_comptonPlots)

    # Ploting Diagnostics
    debug_plots(df_comptons, outputFolder_comptonPlots, custom_name=abc)
    df_comptons.to_parquet(f'{outputFolder_parquet_doublesPeakCompton}/comptons__{abc}.parquet')

    del df_comptons
    
    


def identify_compton(source, peak_energy, z_cdte, cdte_detSize, cdte_pixSize, z_si, si_detSize, si_pixSize, show=False, filter=True):
    """
    This function read the .parquet data that contains the data already sorted in time, coincidence event time identification, number of clusters per event and the ToT already tranformed to keV.

    """
     
    outputFolder = f"{specLib.global_config.output_folder}/{source}"
    outputFolder_parquet = os.path.join(outputFolder, 'parquet')
    outputFolder_comptons = os.path.join(outputFolder_parquet, 'doubles', 'inPeak', 'comptons')

    list_parquet_files = pathlib.get_list_files(outputFolder_parquet, startswith='df_all_data', endswith='.parquet')

    process_args = [(outputFolder, peak_energy, outputFile, z_cdte, cdte_detSize, cdte_pixSize, z_si, si_detSize, si_pixSize) for outputFile in list_parquet_files]
    
    
    if pathlib.check_dir_exists(outputFolder_comptons):
        n_files_output_folder = pathlib.check_number_files_in_dir(outputFolder_comptons, startswith='comptons', endswith='.parquet')
        n_files_input_folder = pathlib.check_number_files_in_dir(outputFolder_parquet, startswith='df_all_data', endswith='.parquet')

        if n_files_output_folder >= n_files_input_folder:
            print('     process_comptons() already run....')
        else:
            print(f'     process_comptons() didnt fully ran, [{n_files_output_folder}/{n_files_input_folder}] files ran. Calling progress_comptons() again....')

            with Pool() as pool:
                for _ in tqdm(pool.imap_unordered(process_comptons, process_args), total=len(list_parquet_files), desc=f'Identifying Comptons: {peak_energy}keV'):
                              pass

    else:
        #for task in tqdm(process_args,
        #                 total=len(process_args),
        #                 desc='Identify Comptons'):
        #    process_comptons(task)
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(process_comptons, process_args), total=len(list_parquet_files), desc=f'Identifying Comptons: {peak_energy}keV'):
                          pass




def plot_small_circ(inputFolder, distance, x_center_pixel, y_center_pixel):

    df_list = [pd.read_parquet(os.path.join(f'{inputFolder}/parquet/', filename), columns = ['E1_X0','E1_Y0','E2_X0','E2_Y0', 'E_photon=E1']) for filename in os.listdir(f'{inputFolder}/parquet/') if filename.startswith('restricted_phi_df') and filename.endswith(".parquet")]
    
    # Concatenate all DataFrames in the list into a single DataFrame
    data = pd.concat(df_list, ignore_index=True)

    # Convert pixel coordinates to millimeters
    data['E1_X0_mm'] = data['E1_X0'] * 0.055
    data['E1_Y0_mm'] = data['E1_Y0'] * 0.055
    data['E2_X0_mm'] = data['E2_X0'] * 0.055
    data['E2_Y0_mm'] = data['E2_Y0'] * 0.055

    # Convert center pixel coordinates to millimeters
    x_center_mm = x_center_pixel * 0.055
    y_center_mm = y_center_pixel * 0.055

    # Determine which coordinates correspond to the photon
    data['photon_X0_mm'] = np.where(
        data['E_photon=E1'] == 'yes', data['E1_X0_mm'], data['E2_X0_mm']
    )
    data['photon_Y0_mm'] = np.where(
        data['E_photon=E1'] == 'yes', data['E1_Y0_mm'], data['E2_Y0_mm']
    )

    # Calculate Euclidean distance between the center pixel and the photon cluster coordinates
    data['distance_mm'] = np.sqrt(
        (data['photon_X0_mm'] - x_center_mm)**2 +
        (data['photon_Y0_mm'] - y_center_mm)**2
    )
        
    final_filtered_df = data[data['distance_mm'] >= distance]
        
    plt.figure(figsize=(10, 10))
    # Plot all points
    plt.scatter(data['photon_X0_mm'], data['photon_Y0_mm'], color='grey')
    # Highlight points that are filtered based on distance
    plt.scatter(final_filtered_df['photon_X0_mm'], final_filtered_df['photon_Y0_mm'], color='green', label=f'Accepted events (distance >= {distance} mm)')
    # Plot the circle for reference
    circle = plt.Circle((x_center_mm, y_center_mm), radius=distance, color='red', fill=False, linestyle='--', label=f'Exclusion Radius ({distance} mm)')
    plt.gca().add_patch(circle)
    plt.xlabel('X-coordinate (mm)')
    plt.ylabel('Y-coordinate (mm)')
    plt.title('Compton Events Data with Filter Applied')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{inputFolder}/Compton_Events_Data_with_Filter_Applied.png')
    plt.close()

def binning_polarimetry(data_pol, data_Nonpol, energy, angle_bin):
    
    to_beam_ref_frame = 0

    data_pol['Phi_Beam_Ref'] = (data_pol['Phi'] + to_beam_ref_frame) % 360
    data_Nonpol['Phi_Beam_Ref'] = (data_Nonpol['Phi'] + to_beam_ref_frame) % 360

    bins_pol = np.arange(0, 360 + angle_bin, angle_bin)
    bins_Nonpol = np.arange(0, 360 + angle_bin, angle_bin)
    bin_pol_centers = (bins_pol[:-1] + bins_pol[1:]) / 2
    bin_Nonpol_centers = (bins_Nonpol[:-1] + bins_Nonpol[1:]) / 2

    # Bin the Phi values and use bin centers as labels
    data_pol['Phi_bin'] = pd.cut(data_pol['Phi_Beam_Ref'], bins_pol, right=False, labels=bin_pol_centers)
    data_Nonpol['Phi_bin'] = pd.cut(data_Nonpol['Phi_Beam_Ref'], bins_Nonpol, right=False, labels=bin_Nonpol_centers)
    
    # Count the occurrences in each bin
    phi_counts_binned_pol= data_pol['Phi_bin'].value_counts().sort_index()
    phi_counts_binned_Nonpol= data_Nonpol['Phi_bin'].value_counts().sort_index()

    phi_counts_binned_df_pol = phi_counts_binned_pol.reset_index()
    phi_counts_binned_df_pol.columns = ['Phi_bin', 'Counts']
    phi_counts_binned_df_Nonpol = phi_counts_binned_Nonpol.reset_index()
    phi_counts_binned_df_Nonpol.columns = ['Phi_bin', 'Counts']
    
    
    # Normalize to the mean the Pol counts
    total_phi_counts_binned_df_pol = phi_counts_binned_df_pol['Counts'].sum()
    n = phi_counts_binned_df_pol['Phi_bin'].nunique()
    mean_binned_pol = float(total_phi_counts_binned_df_pol / n)
    phi_counts_binned_df_pol['Counts_Norm'] = phi_counts_binned_df_pol['Counts']/ mean_binned_pol
    phi_counts_binned_df_pol['Error_Norm'] = np.sqrt(phi_counts_binned_df_pol['Counts'])/mean_binned_pol

    # Normalize to the mean the NonPol counts
    total_phi_counts_binned_df_Nonpol = phi_counts_binned_df_Nonpol['Counts'].sum()
    n = phi_counts_binned_df_Nonpol['Phi_bin'].nunique()
    mean_binned_Nonpol = float(total_phi_counts_binned_df_Nonpol / n)
    phi_counts_binned_df_Nonpol['Counts_Norm'] = phi_counts_binned_df_Nonpol['Counts']/ mean_binned_Nonpol
    phi_counts_binned_df_Nonpol['Error_Norm'] = np.sqrt(phi_counts_binned_df_Nonpol['Counts'])/mean_binned_Nonpol


    # Apply correction of Non-Polarized reponce, to the normalized counts
    # N_true = (N_pol/N_nonpol)* max(N_nonpol)
    phi_counts_corrected = phi_counts_binned_df_pol['Counts']* (phi_counts_binned_df_Nonpol['Counts'].max() / phi_counts_binned_df_Nonpol['Counts'])  

    phi_counts_corrected_df = pd.DataFrame({
        'Phi_bin': phi_counts_binned_df_pol['Phi_bin'],  # Use pol bins, non-pol bins is the same
        'Counts_corrected': phi_counts_corrected,
        'Error' : np.sqrt(phi_counts_corrected)
    })

    # Normalizing the counts to the mean
    total_counts_binned = phi_counts_corrected_df['Counts_corrected'].sum()
    n_phis_binned = phi_counts_corrected_df['Phi_bin'].nunique()
    mean_binned = float(total_counts_binned / n_phis_binned)
    phi_counts_corrected_df['Counts_Norm'] = phi_counts_corrected_df['Counts_corrected'] / mean_binned
    phi_counts_corrected_df['Error_Norm'] = np.sqrt(phi_counts_corrected_df['Counts_corrected'])/mean_binned


    return phi_counts_corrected_df, phi_counts_binned_df_pol, phi_counts_binned_df_Nonpol, bin_pol_centers


def count_nEvents_allTypes(outputFolder, energy, chip, show=False, filter=True):
    """
    This function read the .parquet data that contains the data already sorted in time, coincidence event time identification, number of clusters per event and the ToT already tranformed to keV.

    """
    outputFolder_parquet = os.path.join(outputFolder, 'parquet')
    
    count_singles = 0
    count_doubles = 0
    count_multi = 0
    
    count_singles_peak = 0
    count_doubles_peak = 0
    count_multi_peak = 0

    total_events_peak = 0

    
    list_parquet_files = pathlib.get_list_files(outputFolder_parquet, startswith='df_all_data', endswith='.parquet')


    for outputFile in list_parquet_files:
        hold = os.path.basename(outputFile).split('.')[0]
        
        try:
            mybeer = hold.split('__')
            abc = mybeer[1]
            data = pd.read_parquet(f'{outputFolder_parquet}/df_all_data_df__{abc}.parquet', columns = ['Event', 'Cluster', 'ToT (keV)', 'Overflow'])
        except:
            mybeer = hold.split('_')
            abc = mybeer[-1]
            data = pd.read_parquet(f'{outputFolder_parquet}/df_all_data_df_{abc}.parquet', columns = ['Event', 'Cluster', 'ToT (keV)', 'Overflow'])

        

        count_singles_peak_hold = len(counts_in_energy_peak(data, energy)[0])
        count_doubles_peak_hold = len(counts_in_energy_peak(data, energy)[1])
        count_multi_peak_hold = len(counts_in_energy_peak(data, energy)[2])
        
        count_singles_peak += count_singles_peak_hold
        count_doubles_peak += count_doubles_peak_hold
        count_multi_peak += count_multi_peak_hold

        total_events_peak += count_singles_peak_hold + count_doubles_peak_hold + count_multi_peak_hold

    #print(f"# Total Events in Peak: {total_events_peak}\n# Single Events in Peak: {count_singles_peak}\n# Double Events in Peak: {count_doubles_peak}\n# Multiple Events in Peak: {count_multi_peak}")
    with open(f'{outputFolder}/AllEventsCount.txt', 'w') as f:
        f.write(f"# Total Events in Peak: {total_events_peak}\n# Single Events in Peak: {count_singles_peak}\n# Double Events in Peak: {count_doubles_peak}\n# Multiple Events in Peak: {count_multi_peak}")


def count_finalComptons(folder_input_polarimetry_pol, folder_input_polarimetry_Nonpol, result_polarimetry, concat_df, min_dist, max_dist, angle_bin, residual=False, pol_both = False):
    n_comptons = 0
    
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
    pathlib.creat_dir(folder_result_polarimetry)
    
    #final_comptons_parquet = os.path.join(outputFolder, 'parquet', 'doubles', 'inPeak', 'comptons', 'Phi')
    #list_parquet_files = specLib.get_list_files(final_comptons_parquet, startswith='final_comptons_limitDists', endswith='.parquet')

    #for file in list_parquet_files:

    #    hold = os.path.basename(file).split('.')[0]
    #    mybeer = hold.split('__')
    #    abc = mybeer[1]

    #    data = pd.read_parquet(f'{final_comptons_parquet}/final_comptons_limitDists__{abc}.parquet')
       
    #    n_comptons += len(data)
    
    #print(f'this is n_comptons {n_comptons}')
    
    n_comptons = len(concat_df)

    angle_bin = str(angle_bin).replace('.', '_')
    min_dist = str(min_dist).replace('.','-')

    with open(f'{folder_result_polarimetry}/ComptonsEventsCount.txt', 'w') as f:
        f.write(f"# Compton Events Used: {n_comptons}")

    
    file_path_allEvents = f'{folder_input_polarimetry_pol}/AllEventsCount.txt' # go get AllEventsCount that its on Polarized source!!!!

    with open(file_path_allEvents, 'r') as file:
        for line in file:
            if "Total Events in Peak" in line:
                # Extract the number using string splitting
                total_events = int(line.split(":")[1].strip())
                #print(f"Double Events in Peak: {total_events}")
                break

    compton_eff = n_comptons/total_events

    with open(f'{folder_result_polarimetry}/ComptonsEventsCount.txt', 'a') as f:
        f.write(f"\n# Compton eff: {compton_eff}")
   
    file_path_fits = f'{folder_result_polarimetry}/Fit_Values.txt'

    with open(file_path_fits, 'r') as file:
        for line in file:
            if line.startswith("Q ="):
                # Extract the part after "Q ="
                q_value = line.split("Q =")[1].strip()
                # Split the value and its uncertainty
                value, uncertainty = q_value.split("±")
                Q = float(value.strip())  # Convert to float
                Q_uncertainty = float(uncertainty.strip())  # Convert to float
                break

    merit_figure = (Q**2)*compton_eff

    with open(f'{folder_result_polarimetry}/ComptonsEventsCount.txt', 'a') as f:
        f.write(f"\n# Merit Figure: {merit_figure}")

    del concat_df
    gc.collect()

def plot_figureMeritMap(outputFolder, min_dist_list, angle_bin_list, max_dist_list, abs = False):

    merit_matrix = np.zeros((len(min_dist_list), len(angle_bin_list)))
    sigma_merit_matrix = np.zeros((len(min_dist_list), len(angle_bin_list)))

    energy_source = get_energy_from_source_name(outputFolder)

    for k, max_dist in enumerate(max_dist_list):
        for i, min_dist in enumerate(min_dist_list):
            for j, angle_bin in enumerate(angle_bin_list):

                if abs == False:
                    merit_value = get_MeritFigure(outputFolder, min_dist, angle_bin, max_dist)
                    merit_matrix[i, j] = merit_value * 10**3
                if abs == True:
                    merit_value, sigma = get_AbsMeritFigure(outputFolder, min_dist, angle_bin, max_dist)

                    '''
                    VERY IMPORTANT
                    FOR THE PAPER WE ARE GOING TO USE THE FIGURE OF MERIT IS THE SQRT OF
                    THE FIGURE OF MERIT PREVIOUSELY CALCULATED, GOING TO FORCE THIS NOW 
                    BUT UPDATE THE WHOLE CODE AFTER!!!
                    '''

                    merit_value = np.sqrt(merit_value)
                    sigma = np.sqrt(merit_value)

                    merit_matrix[i, j] = merit_value * 10 **2 
                    sigma_merit_matrix[i,j] = sigma * 10 **2

        
        max_idx = np.unravel_index(np.argmax(merit_matrix), merit_matrix.shape)
        max_x = angle_bin_list[max_idx[1]]
        max_y = min_dist_list[max_idx[0]]

        colors = plt.cm.jet(np.linspace(0, 1, 256))  # Start with the jet colormap
        colors[-1] = [89/255, 0, 0, 1]  # Set the last color to black (RGBA: 0, 0, 0, 1)
        custom_cmap = LinearSegmentedColormap.from_list('custom_jet', colors)

        plt.figure(figsize=(10, 8))
        X, Y = np.meshgrid(angle_bin_list, min_dist_list)
        #plt.pcolormesh(X, Y, merit_matrix, shading='auto', cmap='Greys', norm=LogNorm(vmin=np.min(merit_matrix),vmax=np.max(merit_matrix)))
        mesh = plt.pcolormesh(X, Y, merit_matrix, shading='auto', cmap=custom_cmap ,alpha=1, vmin=np.min(merit_matrix), vmax=np.max(merit_matrix), label=f'{energy_source}')

        plt.colorbar(label=r"Merit Figure, $F$ ($\times 10^{-2}$)")

        plt.scatter(max_x, max_y, marker='*', s=200, color='yellow', edgecolors='black', zorder=10, label='Max Value')
        proxy = plt.Line2D([0], [0], linestyle='None', color='gray', label=f'{energy_source} keV')
        plt.legend(handles=[proxy], loc='upper right', handlelength=0, handletextpad=0)

        plt.tick_params(axis='both', direction='out', length=6, width=1.5)
        
        #plt.xticks(angle_bin_list, labels=[f"{x:.0f}" for x in angle_bin_list])
        #plt.yticks(min_dist_list, labels=[f"{y:.3f}" for y in min_dist_list])

        #xticks = np.linspace(angle_bin_list[0], angle_bin_list[-1], num=10)
        #yticks = np.linspace(min_dist_list[0], min_dist_list[-1], num=5)

        #plt.xticks(xticks, labels=[f"{x:.1f}" for x in xticks])
        #plt.yticks(yticks, labels=[f"{y:.3f}" for y in yticks])

        print(min_dist_list)
        print(len(min_dist_list))
        xticks = [angle_bin_list[i] for i in range(0, len(angle_bin_list), len(angle_bin_list)//2)]  # pick every nth element
        yticks = [min_dist_list[i] for i in range(0, len(min_dist_list), len(min_dist_list)//2)]  # pick every nth element

        # Ensure that xticks and yticks have the correct number of labels
        plt.xticks(xticks, labels=[f"{x:.1f}" for x in xticks])
        plt.yticks(yticks, labels=[f"{y:.3f}" for y in yticks])


        
        plt.xlabel(r"Angle Bin ($^{\circ}$)")
        plt.ylabel("Min Distance (mm)")

        if abs:
            #plt.title(f"2D Color Map of Merit Figure Abs max_dist{max_dist}")
            #plt.show()
            plt.savefig(f'{outputFolder}/MeritFigureAbs_maxd{max_dist}.png')
            plt.close()
        else:
            plt.title(f"2D Color Map of Merit Figure max_dist{max_dist}")
            plt.savefig(f'{outputFolder}/MeritFigure_maxd{max_dist}.png')
            plt.close()

    return


def get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist, abs = True):
    # For a fixed max_dist this funtion gives the r_min and angle_bin that maximized the figure of merit
    # the default baheviour is to calculate the best conditions for rot=0 then for other rotations it uses the values for rot = 0
    '''
    return max_merit, best_min_dist, best_angle_bin, sigma_max_merit
    '''
    source = source_folder.split('/')[-1]
    base_path = os.path.dirname(source_folder)
    #grenoble_general_conclusion_path = f'{base_path}/3-GrenobleGeneralConclusions'
    grenoble_general_conclusion_path = source_folder

    rot = get_rot_from_source_name(source)
    source_energy = get_energy_from_source_name(source)


    merit_matrix = np.zeros((len(min_dist_list), len(angle_bin_list)))
    sigma_merit_matrix = np.zeros((len(min_dist_list), len(angle_bin_list)))

    
    for i, min_dist in enumerate(min_dist_list):
        for j, angle_bin in enumerate(angle_bin_list):

            if abs == False:
                merit_value = get_MeritFigure(source_folder, min_dist, angle_bin, max_dist)
                merit_matrix[i, j] = merit_value
                break
            if abs == True:
                #print(source_folder, min_dist, angle_bin, max_dist)
                merit_value, sigma = get_AbsMeritFigure(source_folder, min_dist, angle_bin, max_dist)
                merit_matrix[i, j] = merit_value
                sigma_merit_matrix[i,j] = sigma
                break
    
    # Find the (i, j) indices of the maximum value in the merit_matrix
    max_index = np.argmax(merit_matrix)  # Flattened index of the maximum value
    i_max, j_max = np.unravel_index(max_index, merit_matrix.shape)  # Convert to 2D indices
    
    # Get the corresponding min_dist and angle_bin values
    best_min_dist = min_dist_list[i_max]
    best_angle_bin = angle_bin_list[j_max]
    max_merit = merit_matrix[i_max, j_max]
    sigma_max_merit = sigma_merit_matrix[i_max, j_max]
    
    with open(f'{grenoble_general_conclusion_path}/{source_energy}kev_best_polarimetry_variables.txt', 'w') as f:
        f.write(f'# best_min_dist: {best_min_dist}\n# best_angle_bin: {best_angle_bin} \n# max_dist: {max_dist}')

    return max_merit, best_min_dist, best_angle_bin, sigma_max_merit


def plot_QvrsRadius_combined_absEffvrsRadius(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist, energies_overlap = None):
    
    dict_markers ={'100':'o','150': '^','200': 's','250': 'X','300':'d'}
    dict_colors = {'100':'k', '150': 'blue', '200': 'red', '250': 'orange', '300': 'green'}
    

    y_q_dict = {}
    y_q_uncert_dict = {}
    y_eff_dict = {}
    y_eff_uncert_dict = {}

    energies_all_sources = []

    conclusions_folder = f'{output_folder_base}/3-GrenobleGeneralConclusions'
    conclusions_folder = output_folder_base
    for source in sources:
        sourceFolder = f'{output_folder_base}/{source}'

        source_energy = get_energy_from_source_name(source)

        if source_energy not in energies_all_sources:
            energies_all_sources.append(source_energy)
            
        x = []
        y_q = []
        y_q_uncert = []
        y_eff = []
        y_eff_uncert = [] 
        #print(source)
        
        merit, best_min_dist, best_angle_bin, sigma = get_bestPolarimetryConditions(sourceFolder, min_dist_list, angle_bin_list, max_dist, abs = False)

        for min_dist in min_dist_list:
            q_value, q_uncertanty = get_Q(sourceFolder, min_dist, best_angle_bin, max_dist)
            

            y_q.append(q_value)
            y_q_uncert.append(q_uncertanty)


            eff, sigma_eff = get_absoluteComptonEff(sourceFolder, min_dist, best_angle_bin, max_dist)

            y_eff.append(eff)
            y_eff_uncert.append(sigma_eff)

            x.append(min_dist)

        y_q_dict[(source_energy, rot_source)] = y_q
        y_q_uncert_dict[(source_energy, rot_source)] = y_q_uncert
        y_eff_dict[(source_energy, rot_source)] = y_eff
        y_eff_uncert_dict[(source_energy, rot_source)] = y_eff_uncert
   
    if energies_overlap:

        plot_qeff_mindist_subplot(conclusions_folder, x, y_q_dict, y_q_uncert_dict, y_eff_dict, y_eff_uncert_dict, energies_overlap)
        check = 1
    else:
        check = check + 1
        plot_qeff_mindist_subplot(sourceFolder, x, y_q_dict, y_q_uncert_dict, y_eff_dict, y_eff_uncert_dict, energies_overlap)



def plot_qeff_mindist_subplot(output_folder, x, y_q_dict, y_q_uncert_dict, y_eff_dict, y_eff_uncert_dict, energies_overlap):
    
    fig, ax1 = plt.subplots(figsize=(8,7))
    ax2 = ax1.twinx()

    dict_markers ={'100':'o','150': '^','200': 's','250': 'X','300':'d'}
    dict_colors = {'100':'k', '150': 'blue', '200': 'red', '250': 'orange', '300': 'green'}
    
    check = 0

    for energy_plot in energies_overlap:
        y_q = []
        y_q_uncert = []
        y_eff = []
        y_eff_uncert = []

        for energy, rot in y_q_dict:
            if energy == energy_plot:
                y_q = (y_q_dict[(energy, rot)])
                y_q_uncert = (y_q_uncert_dict[(energy, rot)])
                y_eff = (y_eff_dict[(energy, rot)])
                y_eff_uncert = (y_eff_uncert_dict[(energy, rot)])

        

        marker = dict_markers[f'{energy_plot}']
        color = dict_colors[f'{energy_plot}']
        
        
        ax1.errorbar(x, y_q, yerr=y_q_uncert, fmt=marker, color=color, ecolor=color , markersize=7, capsize=5, label=f' {energy_plot} keV')

        x_new = np.linspace(np.min(x), np.max(x), 500)
        f = interp1d(x, y_q, kind='quadratic')
        y_smooth = f(x_new)
        ax1.plot(x_new, y_smooth, linestyle='--', c=color, alpha=0.3)

        #ax2.errorbar(x, y_eff, yerr = y_eff_uncert, fmt=marker, color=color , ecolor=color, markerfacecolor='none', markeredgecolor=color, label=r'$\epsilon_{cp}$, ' + f'{energy_plot} keV')
        ax2.errorbar(x, y_eff, yerr = y_eff_uncert, fmt=marker, color=color , ecolor=color, markersize=8, markerfacecolor='none', markeredgecolor=color)
        x_new_eff = np.linspace(np.min(x), np.max(x), 500)
        f = interp1d(x, y_eff, kind='cubic')
        y_smooth_eff = f(x_new_eff)
        ax2.plot(x_new_eff, y_smooth_eff, linestyle='--', c=color, alpha=0.3)

        
        ax2.set_ylim(0.0001, 1)
        ax2.set_yscale('log')

        ax2.tick_params(axis='y', labelcolor='k')
        ax1.tick_params(axis='y', labelcolor='k')
        
        y1_ticks = np.arange(0, 1.1, 0.1)
        ax1.set_yticks(y1_ticks)
        ax1.minorticks_on()
        
        ax2.minorticks_on()

        
        ax1.grid(False)
        ax2.grid(False)

    ax1.set_xlabel(r'Minimum Distance, \textit{r$_{{min}}$}(mm)')
    ax1.set_ylabel(r'Modulation Factor, \textit{Q$_{{100}}$}', color='k')

    # Smooth the Q curve

    # Create a second y-axis for Compton efficiency
    ax2.set_ylabel(r'Compton Efficiency, $\epsilon_{cp}$', color='k')

    # Smooth the eff curve
    
    # Scale the right y-axis to match the relative scale of the left y-axis
    
    x_minDist_max = 2

    plt.xlim(0,x_minDist_max)
    #eff_min, eff_max = ax2.get_ylim()

    #ten = len(y1_ticks)-1

    #min_ylim = (eff_min // 10) * 10
    #min_ylim = eff_min
    #max_ylim = eff_max 
    
    #n = int(round(2/0.055,0))
    #min_eff = y_eff[n+1]
    #min_eff = np.min(y_eff)
    #max_eff = np.max(y_eff)


    #if min_eff > 0:  # Ensure min_eff is positive for log scale
    #    min_ylim = 10 ** np.floor(np.log10(min_eff))
    #else:
    #    min_ylim = min_eff  # Fallback for non-positive values

# Round max_eff up to the next decade on the log scale
    #if max_eff > 0:  # Ensure max_eff is positive for log scale
    #    max_ylim = 10 ** np.ceil(np.log10(max_eff))
    #else:
    #    max_ylim = max_eff  # Fallback for non-positive values

# Set the y-axis limits for ax2
    #ax2.set_ylim(min_ylim, max_ylim)

    #x2_ticks = np.arange(0, eff_max + (eff_max/ten), eff_max/ten)
    #ax2.set_yticks(x2_ticks)
    


    # Add title and grid
    #plt.title(rf'Q$_{{100}}$ and Compton Efficiency vs Radius with Binning = {angle_bin}deg')
    #plt.grid(True)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, ncol=2, loc='best', title_fontsize=17, fontsize=15)

    plt.tight_layout()

    ax1.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, pad = 11)
    ax2.tick_params(direction='in', axis='both', which='both')

    # Save and close the figure
    plt.savefig(f'{output_folder}/Q_and_Eff_vs_Radius{energies_overlap}.png', bbox_inches='tight')
    #plt.show()
    #plt.close()

def plot_QvrsRadius(outputFolder, min_dist_list, angle_bin_list, max_dist):

    merit_matrix = np.zeros((len(min_dist_list), len(angle_bin_list)))

    x = []
    y = []
    y_uncert = []
   
    for i, min_dist in enumerate(min_dist_list):
        for j, angle_bin in enumerate(angle_bin_list):
            q_value, q_uncertanty = get_Q(outputfolder, min_dist, angle_bin, max_dist)
            
            x.append(min_dist)
            y.append(q_value)
            y_uncert.append(q_uncertanty)
    
    plt.figure(figsize=(10, 6))
    x_new = np.linspace(np.min(x), np.max(x),500)

    f = interp1d(x, y, kind='quadratic')
    y_smooth=f(x_new)
    plt.plot (x_new,y_smooth, linestyle = '--', c = 'black', alpha = 0.3)
    plt.errorbar(x, y, yerr=y_uncert, fmt='o', color='k', ecolor='k', capsize=5, label=r'Q$_{{100}}$')
    plt.xlabel('Minimum Distance (mm)')
    plt.ylabel(r'Q$_{{100}}$ Value')
    plt.title(rf'Q$_{{100}}$ vs Radius with fixed Binning = {angle_bin}deg')
    plt.grid(True)
    plt.legend()
    plt.minorticks_on()
    plt.savefig(f'{outputFolder}/QvrsRadius_bin{angle_bin}.png')
    plt.close()
    #plt.show()
    return 

def plot_QvrsBin(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist, min_dist_definition, energies_overlap = None ):
#j just perfomr analysis for rot=0
    bin = []
    q_energies_dict = {}

    energies_all_sources = []
    
    #conclusions_folder = f'{output_folder_base}/3-GrenobleGeneralConclusions'

    conclusions_folder = output_folder_base

    for source in sources:
        sourceFolder = f'{output_folder_base}/{source}'

        source_energy = get_energy_from_source_name(source)

        if source_energy not in energies_all_sources:
            energies_all_sources.append(source_energy)
            
        bin = []
        y_q = []
        y_q_uncert = []
        y_eff = []
        y_eff_uncert = [] 
        #print(source)
        
        for min_dist in min_dist_list:
            if min_dist == min_dist_definition:
                for angle_bin in angle_bin_list:

                    q_value, q_uncertanty = get_Q(sourceFolder, min_dist, angle_bin, max_dist)
            
                    y_q.append(q_value)
                    y_q_uncert.append(q_uncertanty)

                    bin.append(angle_bin)

        q_energies_dict[(source_energy)] = (y_q, y_q_uncert)

    plot_QvrsBin_subplot(conclusions_folder, bin, q_energies_dict, energies_overlap, min_dist_definition)

def plot_QvrsBin_subplot(output_folder, bin, q_energies_dict, energies_overlap, min_dist): 
    
    print('PLOT BIN SUBPLOT')
    fig, ax1 = plt.subplots(figsize=(7,7))
    
    x = bin

    dict_markers ={'100':'o','200': '^','300': 's','400': 'X','500':'d'}
    dict_colors = {'100':'k', '200': 'blue', '300': 'red', '400': 'orange', '500': 'green'}

    for energy_plot in energies_overlap:
        y = []
        y_err = []
        for energy in q_energies_dict:
            if energy == energy_plot:
                y = q_energies_dict[(energy)][0]
                y_err = q_energies_dict[(energy)][1]

        marker = dict_markers[f'{energy_plot}']
        color = dict_colors[f'{energy_plot}']

        ax1.errorbar(x, y, yerr=y_err, fmt=marker, color=color, ecolor=color, markersize = 7,
             capsize=5, label=fr'$Q_{{100}}$, {energy_plot} keV')
    

    ax1.legend(loc = "upper right", ncol=2)

    ax1.set_ylabel(r'Modulation Factor, Q$_{{100}}$')
    #ax1.set_ylim(0.45, 0.7)
    plt.xlabel(r'Angle Bin, $\Delta \phi$($^\circ$)')
    plt.title(fr'$r_{{min}}$ = {round(min_dist,3)} mm')
    plt.minorticks_on()
    plt.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True)
    plt.grid(False)
    plt.tight_layout()
    #plt.savefig(f'{output_folder}/QvrsBinning_md{round(min_dist,3)}.png')
    plt.show()
    plt.close()
    return 

def compute_uncert_q_normalized(q, q_uncer, q_i, q_i_uncert):
    uncert = np.sqrt((q_uncer/q_i)**2 + (-((q*q_i_uncert)/q_i**2))**2)
    #print(uncert)
    return uncert

def plot_AbsEffvrsMaxdist(outputFolder, min_dist, angle_bin, max_dist_list, multiple_energies = False, normalized_q = False, ax1 = None, ax2=None):

    x = []
    y = []
    y_uncert = []

    x_eff = []
    x_eff_sigma = []

    x_fm = []
    x_fm_sigma = []

    q_initial = 1
    q_initial_uncertanty = 0
    q_initial_check = False

    source_energy = get_energy_from_source_name(outputFolder)

    for k, max_dist in enumerate(max_dist_list):
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')

        folder = f'{outputFolder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
        file_merit = f'{folder}/Fit_Values.txt'

        if os.path.exists(file_merit):
            with open(file_merit, 'r') as f:
                for line in f:
                    if "Q = " in line:
                        q_line = line.split('=')[-1].strip()
                        q_value = round(float(q_line.split('±')[0].strip()),4)
                        q_uncertanty = round(float(q_line.split('±')[-1].strip()),6)

                        if not q_initial_check:
                            q_initial = q_value
                            q_initial_uncertanty = q_uncertanty
                            q_initial_check = True
                        x.append(max_dist)
                        if normalized_q:
                            # if you want not normalized just change the bellow variable q_value with 1
                            y.append(q_value/q_initial)
                            y_uncert_normalized = compute_uncert_q_normalized(q_value, q_uncertanty, q_initial, q_initial_uncertanty)
                            y_uncert.append(y_uncert_normalized)
                        else:
                            y.append(q_value)
                            y_uncert.append(q_uncertanty)
                        break

        eff, sigma_eff = get_absoluteComptonEff(outputFolder, min_dist, angle_bin, max_dist)
        x_eff.append(eff*100)
        x_eff_sigma.append(sigma_eff)

        fm, sigma_fm = get_AbsMeritFigure(outputFolder, min_dist, angle_bin, max_dist)
        x_fm.append(fm)
        x_fm_sigma.append(sigma_eff)
        
    dict_markers ={'100':'o','150': '^','200': 's','250': 'X','300':'d'}
    dict_colors = {'100':'k', '150': 'blue', '200': 'red', '250': 'orange', '300': 'green'}

    marker = dict_markers[f'{source_energy}']
    color = dict_colors[f'{source_energy}']


    try:
        if multiple_energies:
            print(f'energy: {source_energy}')
            print('performing rmax multiple images')
        else:
            plt.figure(figsize=(10, 6))

        if normalized_q and multiple_energies: 
            x_new = np.linspace(np.min(x), np.max(x),500)
            f = interp1d(x, y, kind='quadratic')
            y_smooth=f(x_new)
            
            # USE for Q vs Max_dist
            #plt.plot (x_new,y_smooth, linestyle = '--', c = 'black', alpha = 0.3)
            #ax1.scatter(x,y, marker=marker, color=color, label=f'{source_energy} keV')
            #ax1.set_ylabel(r'Modulation Factor, $Q_{{100}}$')
                
            #USE BELLOW FOR EFF VRS MAX_DIST
            ax1.errorbar(x, x_eff, yerr=x_eff_sigma, fmt=marker, color = color, ecolor=color, label=f'{source_energy} keV', capsize=3)
            ax1.set_ylabel(r'Compton Efficiency, $\epsilon_{cp}$ ($\times 10^{-2}$)', color='k')
            ax1.set_ylim(0.2,2.5)
            ax1.tick_params(axis='y', labelcolor='k')

            ax1.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True)
            ax1.minorticks_on()

        else:
            plt.errorbar(x, y, yerr=y_uncert, fmt=marker, color='k', ecolor='k', capsize=5, label=r'Q$_{{100}}$')
            plt.ylabel(r'\epsilon$_{{cp}}$ Value')

        #ax2 = plt.gca().twinx()
        #ax2.errorbar(x, x_eff, yerr=x_eff_sigma, fmt='^', color='b', ecolor='b', capsize=5, label='Efficiency')
        #ax2.set_ylabel(r'Absolute Compton Efficiency, $\epsilon_{compton}$', color='b')
        #ax2.tick_params(axis='y', labelcolor='b')


        plt.xlabel(r'Maximum Distance, $r_{max}$ (mm)')
        #plt.title(rf'Q$_{{100}}$ vs Max Distance with fixed Binning = {angle_bin}deg, fixes min dist = {min_dist}')
        plt.grid(False)

        plt.legend(loc = 'upper left')
        plt.minorticks_on()
        if multiple_energies:
            return ax1, ax2
        else:
            plt.savefig(f'{outputFolder}/AbseffvrsRadius_bin{angle_bin}_md{min_dist}.png')
            plt.close()
    except (ValueError):
        pass 

    #plt.show()
    return 

def plot_QvrsMaxdist(outputFolder, min_dist, angle_bin, max_dist_list, multiple_energies = False, normalized_q = False, ax1 = None, ax2=None):

    x = []
    y = []
    y_uncert = []
    y_uncert_normalized_list = []

    x_eff = []
    x_eff_sigma = []

    x_fm = []
    x_fm_sigma = []

    q_initial = 1
    q_initial_uncertanty = 0
    q_initial_check = False

    source_energy = get_energy_from_source_name(outputFolder)

    for k, max_dist in enumerate(max_dist_list):
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')

        folder = f'{outputFolder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
        file_merit = f'{folder}/Fit_Values.txt'

        if os.path.exists(file_merit):
            with open(file_merit, 'r') as f:
                for line in f:
                    if "Q = " in line:
                        q_line = line.split('=')[-1].strip()
                        q_value = round(float(q_line.split('±')[0].strip()),4)
                        q_uncertanty = round(float(q_line.split('±')[-1].strip()),6)

                        if not q_initial_check:
                            q_initial = q_value
                            q_initial_uncertanty = q_uncertanty
                            q_initial_check = True
                        x.append(max_dist)
                        if normalized_q:
                            # if you want not normalized just change the bellow variable q_value with 1 otherwise use Q_initial
                            y.append(q_value/q_initial)
                            y_uncert_normalized = compute_uncert_q_normalized(q_value, q_uncertanty, q_initial, q_initial_uncertanty)
                            y_uncert_normalized_list.append(y_uncert_normalized)
                        else:
                            y.append(q_value)
                            y_uncert.append(q_uncertanty)
                        break
        eff, sigma_eff = get_absoluteComptonEff(outputFolder, min_dist, angle_bin, max_dist)
        x_eff.append(eff)
        x_eff_sigma.append(sigma_eff)

        fm, sigma_fm = get_AbsMeritFigure(outputFolder, min_dist, angle_bin, max_dist)
        x_fm.append(fm)
        x_fm_sigma.append(sigma_eff)
        
    dict_markers ={'100':'o','150': '^','200': 's','250': 'X','300':'d'}
    dict_colors = {'100':'k', '150': 'blue', '200': 'red', '250': 'orange', '300': 'green'}

    marker = dict_markers[f'{source_energy}']
    color = dict_colors[f'{source_energy}']


    try:
        if multiple_energies:
            print(f'energy: {source_energy}')
            print('performing rmax multiple images')
        else:
            plt.figure(figsize=(10, 6))

        if normalized_q and multiple_energies: 
            x_new = np.linspace(np.min(x), np.max(x),500)
            f = interp1d(x, y, kind='quadratic')
            y_smooth=f(x_new)
            
            # USE for Q vs Max_dist
            plt.plot (x_new,y_smooth, linestyle = '--', c = 'black', alpha = 0.3)
            ax1.errorbar(x,y, yerr=y_uncert, marker=marker, color=color, label=f'{source_energy} keV', capsize=3)
            ax1.set_ylabel(r'Modulation Factor, $Q_{{100}}$')
                
            #USE BELLOW FOR EFF VRS MAX_DIST
            #ax1.errorbar(x, x_eff, fmt=marker, color = color, ecolor='b', label=f'{source_energy} keV')
            #ax1.set_ylabel(r'Absolute Compton Efficiency, $\epsilon_{compton}$', color='k')
            #ax1.set_ylim(0.002,0.025)
            #ax1.tick_params(axis='y', labelcolor='k')

            
            ax1.set_ylim(0,1)
            ax1.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True)
            ax1.minorticks_on()

        else:
            x_new = np.linspace(np.min(x), np.max(x),500)
            f = interp1d(x, y, kind='quadratic')
            y_smooth=f(x_new)
            plt.plot (x_new,y_smooth, linestyle = '--', c = 'black', alpha = 0.3)
            ax1.errorbar(x,y, yerr=y_uncert, marker=marker, color=color, label=f'{source_energy} keV', capsize=3)
            ax1.set_ylabel(r'Modulation Factor, $Q_{{100}}$')
            ax1.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True)
    #        ax1.set_ylim(0,1)

        #ax2 = plt.gca().twinx()
        #ax2.errorbar(x, x_eff, yerr=x_eff_sigma, fmt='^', color='b', ecolor='b', capsize=5, label='Efficiency')
        #ax2.set_ylabel(r'Absolute Compton Efficiency, $\epsilon_{compton}$', color='b')
        #ax2.tick_params(axis='y', labelcolor='b')


        plt.xlabel(r'Maximum Distance, $r_{max}$ (mm)')
        #plt.title(rf'Q$_{{100}}$ vs Max Distance with fixed Binning = {angle_bin}deg, fixes min dist = {min_dist}')
        plt.grid(False)

        plt.legend(loc = 'upper left')
        plt.minorticks_on()
        if multiple_energies:
            return ax1, ax2
        else:
            plt.savefig(f'{outputFolder}/QvrsRadius_bin{angle_bin}_md{min_dist}.png')
            plt.close()
    except (ValueError):
        pass 

    #plt.show()
    return 

def plot_EffvrsRadius(outputFolder, min_dist_list, angle_bin_list, max_dist):

    merit_matrix = np.zeros((len(min_dist_list), len(angle_bin_list)))

    x = []
    y = []
    y_uncert = []
    
    for i, min_dist in enumerate(min_dist_list):
        for j, angle_bin in enumerate(angle_bin_list):
            angle_bin_str = str(angle_bin).replace('.','-')
            min_dist_str = str(min_dist).replace('.','-')
            max_dist_str = str(max_dist).replace('.','-')

            folder = f'{outputFolder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
            file_merit = f'{folder}/ComptonsEventsCount.txt'
     
            if os.path.exists(file_merit):
                with open(file_merit, 'r') as f:
                    for line in f:
                        if "Compton eff:" in line:
                            eff = round(float(line.split(':')[-1].strip()),4)
                            x.append(min_dist)
                            y.append(eff)
                            break
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(x, y, fmt='o', color='b', ecolor='r', capsize=5, label='Compton Efficiency vs Min Dist')
    plt.xlabel('Min dist (mm)')
    plt.ylabel('Compton detection Efficiency')
    plt.ylim()
    plt.title(f'Compton relative detection Efficiency vs min distance used for a given binning={angle_bin}deg')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{outputFolder}/QvrsBinning_bin{min_dist}.png')
    plt.close()
    #plt.show()
    return 


#def get_energy_from_source_name(source):
#    # only accpets source names that have the energy explicitly on the name.
#    # example: monochromaticbeam_200kev_collimated
#
#    split_source = source.split('_')
#
#    for item in split_source:
#        if 'kev' in item.lower():
#            source_energy = int(item.split('kev')[0])
#            return source_energy

def get_energy_from_source_name(filename: str) -> int:
    """
    Extract energy (keV) from names like:
    CollimatedBeamPol100keV_config4x4_0.5cm
    """
    name = Path(filename).name  # strip path
    m = re.search(r'(\d+)keV', name)
    if not m:
        raise ValueError(f"No energy found in filename: {filename}")
    return int(m.group(1))

def get_pol_type_from_source_name(filename: str) -> str:
    """
    Extract polarization type ('Pol' or 'NonPol') from names like:
    CollimatedBeamPol100keV_config4x4_0.5cm
    CollimatedBeamNonPol100keV_config4x4_0.5cm
    """
    name = Path(filename).name  # strip path
    m = re.search(r'(Pol|NonPol)', name)
    if not m:
        raise ValueError(f"No polarization type found in filename: {filename}")
    return m.group(1)

def get_rot_from_source_name(source):
    # only accpets source names that have the energy explicitly on the name.
    # example: monochromaticbeam_200kev_collimated

    split_source = source.split('_')

    for item in split_source:
        if 'deg' in item.lower():
            rot = float(item.split('deg')[0].replace('-', '.'))
            return rot

def get_Q(source_folder):
    '''
    Give path to results polarimetry, returns Q on Fit_Values.txt file
    '''
    file= f'{source_folder}/Fit_Values.txt'
    try:
        with(open(file, 'r')) as f:
             for line in f:
                if "Q =" in line:
                    q_whole = line.split('=')[-1].strip()
                    q = abs(round(float(q_whole.split('±')[0].strip()),4))
                    q_uncertanty = abs(round(float(q_whole.split('±')[-1].strip()),4))
                    return q, q_uncertanty
    except Exception as e: 
        print(f"ERROR: {source_folder}/Fit_Values.txt doesnt exist - Try run polarimetry.py first - {e}")
        sys.exit(1)

def get_PA(source_folder, min_dist, angle_bin, max_dist):
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    folder = f'{source_folder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
    file= f'{folder}/Fit_Values.txt'

    if os.path.exists(file):
        with(open(file, 'r')) as f:
             for line in f:
                if "PA =" in line:
                    pa_whole = line.split('=')[-1].strip()
                    pa = round(float(pa_whole.split('±')[0].strip()),4)  * -1 # to use the symetrical
                    pa_uncertanty = round(float(pa_whole.split('±')[-1].strip()),4)
                    return pa, pa_uncertanty
    else:
        print(f"ERROR: {source_folder}/Fit_Values.txt doesnt exist - Try run polarimetry.py first")
        sys.exit(1)

def get_ComptonEvents_used(source_folder, min_dist, angle_bin, max_dist):
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    folder = f'{source_folder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
    file= f'{folder}/ComptonsEventsCount.txt'

    if os.path.exists(file):
        with(open(file, 'r')) as f:
             for line in f:
                 if "# Compton Events Used:" in line:
                    n_comptons = float(line.split(' ')[-1])
                    sigma = np.sqrt(n_comptons)
                    return n_comptons, sigma


def get_MeritFigure(source_folder, min_dist, angle_bin, max_dist):
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    folder = f'{source_folder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
    file= f'{folder}/ComptonsEventsCount.txt'

    if os.path.exists(file):
        with(open(file, 'r')) as f:
             for line in f:
                 if "# Merit Figure:" in line:
                    merit_figure = float(line.split(' ')[-1])
                    return merit_figure

def get_AbsMeritFigure(source_folder):
    '''
    Give path to result_polarimetry, returns ABSOLUTE merit figure and simga from ComptonsEventsCount.txt
    '''

    file= f'{source_folder}/ComptonsEventsCount.txt'
    
    try:
        with(open(file, 'r')) as f:
            for line in f:
                if "# Merit Figure Abs:" in line:
                    merit_figure = float(line.split(' ')[-1])
                if "# Sigma Merit Figure Abs:" in line:
                    sigma = float(line.split(' ')[-1])
            return merit_figure, sigma

    except Exception as e:
        print(f"ERROR: {source_folder}/Fit_Values.txt doesnt exist - Try run polarimetry.py first - {e}")
        sys.exit(1)


def get_relativeComptonEff(source_folder):
    '''
    Give path to result_polarimetry, returns RELATIVE compton_eff and sigma
    '''
    file= f'{source_folder}/ComptonsEventsCount.txt'

    try:
        with(open(file, 'r')) as f:
             for line in f:
                 if "# Compton eff:" in line:
                    compton_eff = float(line.split(' ')[-1])
                    return compton_eff 

    except Exception as e: 
        print(f"ERROR: {source_folder}/Fit_Values.txt doesnt exist - Try run polarimetry.py first - {e}")
        sys.exit(1)

def get_absoluteComptonEff(source_folder):
    '''
    Give path to result polarimetry, returns ABSOLUTE compton_eff and sigma
    '''
    file= f'{source_folder}/ComptonsEventsCount.txt'

    try:
        with(open(file, 'r')) as f:
            for line in f:
                if "# Compton Abs eff:" in line:
                    compton_eff = float(line.split(' ')[-1])
                if "# Sigma Compton Abs eff:" in line:
                    sigma = float(line.split(' ')[-1])
            return compton_eff, sigma
    except Exception as e:
        print(f"ERROR: {source_folder}/ComptonsEventsCount.txt doesnt exist - Try run polarimetry.py first - {e}")
        sys.exit(1)


def plot_MeritFigurevrsEnergy(outputFolder_base, sources, min_dist_list, angle_bin_list, max_dist):
    #default uses the sources with rot=0
    source_energy_list = []
    fm = []
    fm_sigma = []

    #output_folder_finalPlot = f'{outputFolder_base}/3-GrenobleGeneralConclusions'
    output_folder_finalPlot = outputFolder_base

    for source in sources:
        output_folder = os.path.join(outputFolder_base, source)

        source_energy = get_energy_from_source_name(source)
        rot = get_rot_from_source_name(source)
        
        if rot == 0:
            max_merit, best_min_dist, best_angle_bin, sigma_max_merit = get_bestPolarimetryConditions(output_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max
            
            merit, merit_sigma = get_AbsMeritFigure(output_folder, best_min_dist, best_angle_bin, max_dist)
            
            '''
            VERY IMPORTANT
            FOR THE PAPER WE ARE GOING TO USE THE FIGURE OF MERIT IS THE SQRT OF
            THE FIGURE OF MERIT PREVIOUSELY CALCULATED, GOING TO FORCE THIS NOW 
            BUT UPDATE THE WHOLE CODE AFTER!!!
            '''

            merit = np.sqrt(merit)
            merit_sigma = 0.5 * (1/(np.sqrt(merit))) * merit_sigma

            fm.append(merit*100)
            fm_sigma.append(merit_sigma*100)


            source_energy_list.append(source_energy)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(7,6.5))

    # Plot Q values on the left y-axis
    ax1.errorbar(source_energy_list, fm, yerr=fm_sigma, fmt='o', color='k', ecolor='k', capsize=5, markersize=6, label = '$Q_{100} \sqrt{\epsilon_{cp}}$' )

    ax1.errorbar(source_energy_list, fm, linestyle='--', 
             alpha=0.3, yerr=fm_sigma, fmt='o', 
             color='k', ecolor='k', capsize=5, markersize=6, 
             markerfacecolor='k', markeredgecolor='k')
    ax1.set_xlabel(r'Energy (keV)')
    ax1.set_ylabel(r'Merit Figure, $F$ ($\times 10^{-2}$)', color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    ax1_ylim_max = np.max(fm) * 1.1
    ax1_ylim_min = np.min(fm) * 0.85
    #ax1_ylim_min = 40**-3

    ax1.set_ylim(ax1_ylim_min, ax1_ylim_max)

    ax1.minorticks_on()
    


    # Add title and grid
    #plt.title(rf'Q$_{{100}}$ and Compton Efficiency vs Radius with Binning = {angle_bin}deg')
    #plt.grid(True)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    ax1.legend(lines1, labels1, loc='upper right')

    plt.tight_layout()

    ax1.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True, pad = 11)

    ax1.grid(False)
    
    plt.savefig(f'{output_folder_finalPlot}/MeritFigurevrsEnergy.png')
    plt.close()

def plot_QAbsComptonEffvrsEnergy(outputFolder_base, sources, min_dist_list, angle_bin_list, max_dist):
    #default uses the sources with rot=0
    measured_q= []
    measured_q_uncertanty = []
    source_energy_list = []

    compton_eff = []
    compton_eff_simga = []

    fm = []
    fm_sigma = []

    #output_folder_final_plot = f'{outputFolder_base}/3-GrenobleGeneralConclusions'
    output_folder_final_plot = outputFolder_base

    for source in sources:
        output_folder = os.path.join(outputFolder_base, source)

        source_energy = get_energy_from_source_name(source)
        rot = get_rot_from_source_name(source)
        
        if rot == 0:
            max_merit, best_min_dist, best_angle_bin, sigma_max_merit = get_bestPolarimetryConditions(output_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max
            
            eff, eff_simga = get_absoluteComptonEff(output_folder, best_min_dist, best_angle_bin, max_dist)
            compton_eff.append(eff*100) #y lable on plot should have x10^-2
            compton_eff_simga.append(eff_simga*100)

            q, q_sigma = get_Q(output_folder, best_min_dist, best_angle_bin, max_dist)
            measured_q_uncertanty.append(q_sigma)
            measured_q.append(q)

            merit, merit_sigma = get_AbsMeritFigure(output_folder, best_min_dist, best_angle_bin, max_dist)
            fm.append(merit)
            fm_sigma.append(merit_sigma)


            source_energy_list.append(source_energy)

    # Create the plot
    fig, ax1 = plt.subplots(figsize=(7.5,6.5))

    # Plot Q values on the left y-axis
    ax1.errorbar(source_energy_list, compton_eff, yerr=compton_eff_simga, fmt='o', color='k', ecolor='k', capsize=5, label=r'Compton Efficiency, $\epsilon_{cp}$', markersize=9)
    ax1.errorbar(source_energy_list, compton_eff, yerr=compton_eff_simga, linestyle='--', alpha = 0.3, color='k', ecolor='k', capsize=5)
    ax1.set_xlabel(r'Energy (keV)')
    ax1.set_ylabel(r'Compton Efficiency, $\epsilon_{cp}$ ($\times 10^{-2}$) ', color='k')
    ax1.tick_params(axis='y', labelcolor='k')

    # Create a second y-axis for Compton efficiency
    ax3 = ax1.twinx()
    ax3.errorbar(source_energy_list, measured_q, yerr = measured_q_uncertanty, fmt='^', color='b', label='Modulation Factor, $Q_{100}$', markersize=9)
    ax3.errorbar(source_energy_list, measured_q, yerr = measured_q_uncertanty, linestyle = '--', alpha = 0.3,color='b')
    ax3.set_ylabel(r'Modulation Factor, $Q_{100}$', color='b')
    ax3.tick_params(axis='y', labelcolor='b')

    
    ax1_ylim_max = np.max(compton_eff) * 1.1
    ax1_ylim_min = np.min(compton_eff) * 0.7


    ax3_ylim_max = np.max(measured_q) * 1.1
    ax3_ylim_min = np.min(measured_q)* 0.9

    ax1.set_ylim(ax1_ylim_min, ax1_ylim_max)
    ax3.set_ylim(ax3_ylim_min, ax3_ylim_max)


    ax3.minorticks_on()
    ax1.minorticks_on()
    


    # Add title and grid
    #plt.title(rf'Q$_{{100}}$ and Compton Efficiency vs Radius with Binning = {angle_bin}deg')
    #plt.grid(True)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right')


    plt.tight_layout()

    ax1.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, pad = 11)
    #ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))
    ax3.tick_params(direction='in', axis='both', which='both')

    ax1.grid(False)
    ax3.grid(False)

    plt.savefig(f'{output_folder_final_plot}/QAbsComptonEffvrsEnergy.png')
    plt.close()



def plot_AbsComptonEffvrsEnergy(outputFolder_base, sources, min_dist_list, angle_bin_list, max_dist):
    measured_q= []
    measured_q_uncertanty = []
    source_energy_list = []
    
    for source in sources:
        output_folder = os.path.join(outputFolder_base, source)

        source_energy = get_energy_from_source_name(source)
        rot = get_rot_from_source_name(source)
        
        if rot == 0:
            max_merit, best_min_dist, best_angle_bin, sigma_max_merit = get_bestPolarimetryConditions(output_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max
            
            q, q_uncertanty = get_absoluteComptonEff(output_folder, best_min_dist, best_angle_bin, max_dist)
            measured_q_uncertanty.append(q_uncertanty)
            measured_q.append(q)

            source_energy_list.append(source_energy)

    plt.errorbar(source_energy_list, measured_q, yerr=measured_q_uncertanty, fmt='o', capsize=5, color='k', ecolor='k')
    plt.xlabel('Source energy (keV)')
    plt.ylabel('Absotute Compton Efficiency')
    plt.show()

def plot_QvrsEnergy(outputFolder_base, sources, min_dist_list, angle_bin_list, max_dist):
    measured_q= []
    measured_q_uncertanty = []
    source_energy_list = []
    
    for source in sources:
        output_folder = os.path.join(outputFolder_base, source)

        source_energy = get_energy_from_source_name(source)
        rot = get_rot_from_source_name(source)
        
        if rot == 0:
            max_merit, best_min_dist, best_angle_bin, sigma_max_merit = get_bestPolarimetryConditions(output_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max
            
            q, q_uncertanty = get_Q(output_folder, best_min_dist, best_angle_bin, max_dist)
            measured_q_uncertanty.append(q_uncertanty)
            measured_q.append(q)

            source_energy_list.append(source_energy)

    plt.errorbar(source_energy_list, measured_q, yerr=measured_q_uncertanty, fmt='o', capsize=5, color='k', ecolor='k')
    plt.xlabel('Source energy (keV)')
    plt.ylabel('Modulation Factor, Q_100')
    plt.show()

def plot_AbsComptonEffvrsRot_fixedEnergy(outputFolder_base, sources, min_dist_list, angle_bin_list, max_dist):
    measured_q= []
    measured_q_uncertanty = []
    source_energy_list = []
    rot_list = []

    energies_all_sources = []
    rot_all_sources = []

    dict_ = {}
    dict_sigma = {}
    
    for source in sources:
        output_folder = os.path.join(outputFolder_base, source)

        source_energy = get_energy_from_source_name(source)

        if source_energy not in energies_all_sources:
            energies_all_sources.append(source_energy)
        else:
            print('continue ENERGY check')
        
        source_rot = get_rot_from_source_name(source)
        if source_rot not in rot_all_sources:
            rot_all_sources.append(source_rot)
        else:
            print('continue ROT check')

        
        max_merit, best_min_dist, best_angle_bin, sigma_max_merit = get_bestPolarimetryConditions(output_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max
        
        q, q_uncertanty = get_absoluteComptonEff(output_folder, best_min_dist, best_angle_bin, max_dist)
        

        if (source_energy, source_rot) not in dict_:
            dict_[(source_energy, source_rot)] = []
        dict_[(source_energy, source_rot)].append(q)
        
        if (source_energy, source_rot) not in dict_sigma:
            dict_sigma[(source_energy, source_rot)] = []
        dict_sigma[(source_energy, source_rot)].append(q_uncertanty)



    for energy_plot in energies_all_sources:
        x = []
        y = [] 
        y_sigma = []
        for energy, rot in dict_.keys():
            if energy == energy_plot:
                y.append(dict_[energy, rot])
                x.append(rot)
                y_sigma.append(dict_sigma[energy, rot])

        x = np.array(x).flatten()
        y_sigma = np.array(y_sigma).flatten()
        y = np.array(y).flatten()


        plt.errorbar(x, y, yerr=y_sigma, capsize=3, ecolor='k', label=f'{energy_plot} keV')

    plt.legend()
    plt.xlabel('Theoretical Rotation Angle')
    plt.ylabel('Absolute Compton Efficiency')
    plt.show()

def plot_AbsMeritFigurevrsRot_fixedEnergy(outputFolder_base, sources, min_dist_list, angle_bin_list, max_dist):
    measured_q= []
    measured_q_uncertanty = []
    source_energy_list = []
    rot_list = []

    energies_all_sources = []
    rot_all_sources = []

    dict_ = {}
    dict_sigma = {}
    
    for source in sources:
        output_folder = os.path.join(outputFolder_base, source)

        source_energy = get_energy_from_source_name(source)

        if source_energy not in energies_all_sources:
            energies_all_sources.append(source_energy)
        else:
            print('continue ENERGY check')
        
        source_rot = get_rot_from_source_name(source)
        if source_rot not in rot_all_sources:
            rot_all_sources.append(source_rot)
        else:
            print('continue ROT check')

        
        max_merit, best_min_dist, best_angle_bin, sigma_max_merit = get_bestPolarimetryConditions(output_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max
        
        q, q_uncertanty = get_AbsMeritFigure(output_folder, best_min_dist, best_angle_bin, max_dist)
        

        if (source_energy, source_rot) not in dict_:
            dict_[(source_energy, source_rot)] = []
        dict_[(source_energy, source_rot)].append(q)
        
        if (source_energy, source_rot) not in dict_sigma:
            dict_sigma[(source_energy, source_rot)] = []
        dict_sigma[(source_energy, source_rot)].append(q_uncertanty)



    for energy_plot in energies_all_sources:
        x = []
        y = [] 
        y_sigma = []
        for energy, rot in dict_.keys():
            if energy == energy_plot:
                y.append(dict_[energy, rot])
                x.append(rot)
                y_sigma.append(dict_sigma[energy, rot])

        x = np.array(x).flatten()
        y_sigma = np.array(y_sigma).flatten()
        y = np.array(y).flatten()


        plt.errorbar(x, y, yerr=y_sigma, capsize=3, ecolor='k', label=f'{energy_plot} keV')
   
    plt.legend()
    plt.xlabel('Theoretical Rotation Angle')
    plt.ylabel('Merit Figure')
    plt.show()


def plot_QvrsRot_fixedEnergy(outputFolder_base, sources, min_dist_list, angle_bin_list, max_dist):
    measured_q= []
    measured_q_uncertanty = []
    source_energy_list = []
    rot_list = []

    energies_all_sources = []
    rot_all_sources = []

    dict_ = {}
    dict_sigma = {}
    
    for source in sources:
        output_folder = os.path.join(outputFolder_base, source)

        source_energy = get_energy_from_source_name(source)

        if source_energy not in energies_all_sources:
            energies_all_sources.append(source_energy)
        else:
            print('continue ENERGY check')
        
        source_rot = get_rot_from_source_name(source)
        if source_rot not in rot_all_sources:
            rot_all_sources.append(source_rot)
        else:
            print('continue ROT check')

        
        max_merit, best_min_dist, best_angle_bin, sigma_max_merit = get_bestPolarimetryConditions(output_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max
        
        q, q_uncertanty = get_Q(output_folder, best_min_dist, best_angle_bin, max_dist)
        


        if (source_energy, source_rot) not in dict_:
            dict_[(source_energy, source_rot)] = []
        dict_[(source_energy, source_rot)].append(q)
        
        if (source_energy, source_rot) not in dict_sigma:
            dict_sigma[(source_energy, source_rot)] = []
        dict_sigma[(source_energy, source_rot)].append(q_uncertanty)



    for energy_plot in energies_all_sources:
        x = []
        y = [] 
        y_sigma = []
        for energy, rot in dict_.keys():
            if energy == energy_plot:
                y.append(dict_[energy, rot])
                x.append(rot)
                y_sigma.append(dict_sigma[energy, rot])

        x = np.array(x).flatten()
        y_sigma = np.array(y_sigma).flatten()
        y = np.array(y).flatten()


        plt.errorbar(x, y, yerr=y_sigma, capsize=0, ecolor='k', label=f'{energy_plot} keV')


    plt.legend()
    plt.xlabel('Theoretical Rotation Angle')
    plt.ylabel('Modulation factor Q_100')
    plt.show()


def plot_rotationMeasurements(outputFolder_base, sources, min_dist_list, angle_bin_list, max_dist, energies_overlap = None):

    theoretical_rot = [0,1,2,3,5,7.5,10,20,45]

    energies_all_sources = []
    rot_all_sources = [] 

    dict_rot_source = {}
    sigma_dict_rot_source = {}
    
    output_folder_final_plot = f'{outputFolder_base}/3-GrenobleGeneralConclusions'
    print(outputFolder_base)

    for source in sources:
        output_folder = os.path.join(outputFolder_base, source)

        source_energy = get_energy_from_source_name(source)
        if source_energy not in energies_all_sources:
            energies_all_sources.append(source_energy)
        else:
            print('continue ENERGY check')

        
        source_rot = get_rot_from_source_name(source)
        if source_rot not in rot_all_sources:
            rot_all_sources.append(source_rot)
        else:
            print('continue ROT check')


        max_merit, best_min_dist, best_angle_bin, sigma_max_merit = get_bestPolarimetryConditions(output_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max
        
        pa, pa_uncertanty = get_PA(output_folder, best_min_dist, best_angle_bin, max_dist)
        
        if (source_energy, source_rot) not in dict_rot_source:
            dict_rot_source[(source_energy, source_rot)] = []
        dict_rot_source[(source_energy, source_rot)].append(pa)
        
        if (source_energy, source_rot) not in sigma_dict_rot_source:
            sigma_dict_rot_source[(source_energy, source_rot)] = []
        sigma_dict_rot_source[(source_energy, source_rot)].append(pa_uncertanty)

    
    energy_check = 0

    for energy_plot in energies_all_sources:
        if energies_overlap == None:
            plot_rot_PA_measurements(output_folder_final_plot, dict_rot_source, sigma_dict_rot_source, energy_plot, energies_overlap)

        else:
            if energy_plot in energies_overlap:
                if energy_check == 0:
                    plt.figure(figsize=(8,7))
                    plot_rot_PA_measurements(output_folder_final_plot, dict_rot_source, sigma_dict_rot_source, energy_plot, energies_overlap)
                    energy_check = 1
                else:   
                    energy_check = energy_check + 1
                    plot_rot_PA_measurements(output_folder_final_plot, dict_rot_source, sigma_dict_rot_source, energy_plot, energies_overlap)
                plt.savefig(f'{output_folder_final_plot}/measuredPA_{energies_overlap}.png')
                break
        


def plot_rot_PA_measurements(output_folder_final_plot, dict_rot_source, sigma_dict_rot_source, energy_plot, energies_overlap):

    dict_markers ={'100':'o','150': '^','200': 's','250': 'X','300':'d'}
    dict_colors = {'100':'k', '150': 'blue', '200': 'red', '250': 'orange', '300': 'green'}
    
    first_time = True
    for energy_plot in energies_overlap:

        x = []
        y_sigma = []
        y =[]
        for energy, rot in dict_rot_source.keys():
            if energy == energy_plot:
                y.append(dict_rot_source[(energy, rot)]) # measured rot = PA
                y_sigma.append(sigma_dict_rot_source[(energy, rot)])
                x.append(rot)
         
        if not x:
            continue
        #plot
        x = np.array(x).flatten()
        y_sigma = np.array(y_sigma).flatten()
        y = np.array(y).flatten()

        popt, pcov = curve_fit(
            linear_func,
            x,
            y,
            sigma=y_sigma,
            absolute_sigma=True
        )
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))

        fit_x = np.linspace(min(x), max(x), 100)
        fit_y = linear_func(fit_x, slope, intercept)

        def compute_linearRsquared(x, y, a, b):
            ss_res = 0
            ss_tot = 0
           
            y_sum = 0

            print(x,y)
            for i in range(len(y)):
                y_sum = y_sum + y[i]

            y_average = y_sum/(len(y))

            for i in range(len(x)):
                ss_res = ss_res + (y[i] - (x[i]*a + b))**2
                ss_tot = ss_tot + (y[i] - y_average)**2

            r_square = 1 - ss_res/ss_tot

            return r_square

        r_square = compute_linearRsquared(x, y, slope, intercept)

        if energies_overlap == None:
            plt.figure(figsize=(7, 7))
        else:
            a=0


        marker = dict_markers[f'{energy_plot}']
        color = dict_colors[f'{energy_plot}']


        plt.errorbar(
            x,
            y,
            yerr=y_sigma,
            fmt=marker,
            ecolor=color,
            color = color,
            capsize=5,
 #           label=f'Measured PA, \n{energy_plot}keV ',
            zorder=1
        )


        if first_time == True:
            plt.plot(fit_x, fit_y, color=color, alpha=0.3, linewidth=1)
            first_time = False
        else:
            plt.plot(fit_x, fit_y, color=color, alpha=0.3, linewidth=1)

# Plot invisible lines (alpha=1) just for the legend
        plt.plot([], [], color=color, alpha=1, linewidth=3,
                 label=f'{energy_plot} keV\n'
                       f'a: {slope:.3f} ± {slope_err:.3f}\n'
                       f'b: {intercept:.3f} ± {intercept_err:.3f}\n'
                       fr'$R^2: {r_square:.3f}$')


    plt.ylabel(r'Measured Angle ($^{\circ}$)')
    plt.xlabel(r'Rotation Angle ($^{\circ}$)')
    #plt.title(f'{energy_plot} keV')
    plt.axline((0, 0), slope=1, color='gray', linestyle='--', label='Ideal Correlation')
    plt.legend(loc='upper left', ncol = 2,  fontsize=13, frameon=False )
    #plt.grid(True, linestyle='--', alpha=0.7)
    plt.grid(False)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', top=True, bottom=True, right = True)


    if energy_plot != 100:
        scale_max = y.max()+5
        scale_min = y.min()-3
        plt.xlim(scale_min, scale_max)
        plt.ylim(scale_min, scale_max+5)

        plt.annotate(
            '',  # No text
            xy=(scale_min, scale_min),  # Point in the inset plot
            xytext=(scale_max*0.55, scale_max*0.04),  # Point in the main plot
            arrowprops=dict(
                arrowstyle='-',  # No arrowhead, just a line
                color='grey',  # Color of the connecting line
                linestyle='--',  # Line style
                linewidth=1,  # Line width
                alpha=0.7
            ),
            zorder=2  # Ensure the lines are drawn on top
        )
        plt.annotate(
            '',  # No text
            xy=(5, 5),  # Point in the inset plot
            xytext=(scale_max*0.55, scale_max*0.4),  # Point in the main plot
            arrowprops=dict(
                arrowstyle='-',  # No arrowhead, just a line
                color='grey',  # Color of the connecting line
                linestyle='--',  # Line style
                linewidth=1,  # Line width
                alpha=0.7
            ),
            zorder=2  # Ensure the lines are drawn on top
        )

        ax_inset = inset_axes(plt.gca(), width="35%", height="35%", loc='lower left',bbox_to_anchor=(0.55, 0.07, 1, 1), bbox_transform=plt.gca().transAxes)  # Adjust size and location


        for energy_plot in energies_overlap:

            x = []
            y_sigma = []
            y =[]
            for energy, rot in dict_rot_source.keys():
                if energy == energy_plot:
                    y.append(dict_rot_source[(energy, rot)]) # measured rot = PA
                    y_sigma.append(sigma_dict_rot_source[(energy, rot)])
                    x.append(rot)
            
            if not x:
                continue
        #plot

            marker = dict_markers[f'{energy_plot}']
            color = dict_colors[f'{energy_plot}']

            x = np.array(x).flatten()
            y_sigma = np.array(y_sigma).flatten()
            y = np.array(y).flatten()
            ax_inset.errorbar(
                x, y,
                yerr=y_sigma,
                fmt=marker,
                ecolor=color,
                color=color,
                capsize=5,
                label='Measured PA',
                zorder=1
            )
            ax_inset.plot(fit_x, fit_y, color=color, alpha = 0.3)

        ax_inset.axline((0, 0), slope=1, color='gray', linestyle='--')
        ax_inset.set_xlim(-1, 5)  # Zoom in on the x-axis (0 to 8 degrees)
        ax_inset.set_ylim(-1, 5)  # Zoom in on the y-axis (0 to 8 degrees)
        ax_inset.grid(False)
        ax_inset.minorticks_on()
        ax_inset.tick_params(axis='both', which='both', top=True, bottom=True, right=True)

    if energies_overlap == None:
        plt.savefig(f'{output_folder_final_plot}/measuredPA_{energy_plot}kev.png')
    else:
        return

def limit_dist_photon_travel(data, min_dist, max_dist):    
    """
    cluster data points for each event and assign globally unique cluster ids.
    
    parameters:
    - data (pd.dataframe): dataframe containing the data to be clustered.
    - global_cluster_id (int): initial global cluster id to start with.

    returns:
    - cluster_dict (dict): a dictionary mapping index to global cluster id.
    - global_cluster_id (int): updated global cluster id after clustering.
    """		
    # convert pixel coordinates to millimeters
    #data['E1_X0_mm'] = data['E1_X0'] * 0.055
    #data['E1_Y0_mm'] = data['E1_Y0'] * 0.055

    #data['E2_X0_mm'] = data['E2_X0'] * 0.055
    #data['E2_Y0_mm'] = data['E2_Y0'] * 0.055

    data['E1_X0_mm'] = data['E1_X0']
    data['E1_Y0_mm'] = data['E1_Y0']

    data['E2_X0_mm'] = data['E2_X0']
    data['E2_Y0_mm'] = data['E2_Y0']

    # determine which coordinates correspond to the photon
    data['photon_X0_mm'] = np.where(
        data['E_photon=E1'] == 'yes', data['E1_X0_mm'], data['E2_X0_mm']
    )
    data['photon_Y0_mm'] = np.where(
        data['E_photon=E1'] == 'yes', data['E1_Y0_mm'], data['E2_Y0_mm']
    )

    # determine the coordinates to the photoelectron
    data['electron_X0_mm'] = np.where(
        data['E_elect=E2'] == 'yes', data['E2_X0_mm'], data['E1_X0_mm']
    )
    data['electron_Y0_mm'] = np.where(
        data['E_elect=E2'] == 'yes', data['E2_Y0_mm'], data['E1_Y0_mm']
    )
    
    x_center_mm = 37.73
    y_center_mm = 5.06
    # calculate euclidean distance between the center pixel and the photon cluster coordinates
    #data['distance_mm'] = np.sqrt(
    #    (data['photon_X0_mm'] - x_center_mm)**2 +
    #    (data['photon_Y0_mm'] - y_center_mm)**2
    #)

    #calculate the absolute min distance between electron and photon
    data['distance_mm'] = np.sqrt(
        (data['photon_X0_mm'] - data['electron_X0_mm'])**2 +
        (data['photon_Y0_mm'] - data['electron_Y0_mm'])**2
    )
        
    mindist_filtered_df = data[data['distance_mm'] >= min_dist]

    maxDist_filtered_df = mindist_filtered_df[mindist_filtered_df['distance_mm'] <= max_dist]

    final_filtered_df = maxDist_filtered_df
    

    return final_filtered_df

def compute_polarimetry_phi(data):    
    """
    uses the barycenter of both photoelectorn and compton photon, in a way it recenters the interaction to a central pixel and computes the phi angle.
    """    
    all_phi = []

    for event, line in data.iterrows():
        phi = math.atan2(line['E_photon_Y0'] - line['E_elect_Y0'],  line['E_photon_X0'] - line['E_elect_X0']) # Angle measured at electron, x axis as reference
        phi_deg = math.degrees(phi)
        if phi_deg < 0:
            phi_deg += 360
        all_phi.append(round(phi_deg,5))

    data['Phi'] = all_phi
    return data


def polarimetry_task(args):
    folder_input_polarimetry_pol, folder_input_polarimetry_Nonpol, result_polarimetry, min_dist, angle_bin, energy, max_dist, z_cdte, z_si, cdte_detSize, si_detSize = args
    concat_df = fits.fit_radial_plot(folder_input_polarimetry_pol, folder_input_polarimetry_Nonpol, result_polarimetry, energy, angle_bin, min_dist, max_dist, z_cdte, z_si, cdte_detSize, si_detSize)
    count_finalComptons(folder_input_polarimetry_pol, folder_input_polarimetry_Nonpol, result_polarimetry, concat_df, min_dist, max_dist, angle_bin)
    #plot_compton_events_used_characteristics(folder_input_polarimetry_pol, folder_input_polarimetry_Nonpol, concat_df, energy, max_dist, min_dist, angle_bin)
    del concat_df
    gc.collect()

def polarimetry_task_residual(args):
    min_dist, angle_bin, output_folder, energy, max_dist, z_cdte, z_si= args
    concat_df = fits.fit_radial_plot(output_folder, energy, angle_bin, min_dist, max_dist, residual=True)
    count_finalComptons(output_folder, concat_df, min_dist, max_dist, angle_bin, residual = True)
    plot_compton_events_used_characteristics(output_folder, concat_df, energy, max_dist, min_dist, angle_bin)
    del concat_df
    gc.collect()


def plot_compton_events_used_characteristics(output_folder, concat_df, energy, max_dist, min_dist, angle_bin):
    
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    plot_save_folder = os.path.join(output_folder, 'photonPolarimetry', f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
    #comptons_path = os.path.join(output_folder, 'parquet', 'doubles', 'inPeak', 'comptons', 'Phi' )
    #parquet_files = [f for f in os.listdir(comptons_path) if (f.endswith('.parquet') and f.startswith('final'))]
    
    min_spectra_range = 0
    max_spectra_range = energy
    n_bins = np.arange(min_spectra_range, max_spectra_range, 1)
    cumulative_counts_electron = np.zeros(len(n_bins) - 1)
    cumulative_counts_photon = np.zeros(len(n_bins) - 1)
    bin_centers_electron = (n_bins[:-1] + n_bins[1:]) /2
    bin_centers_photon = (n_bins[:-1] + n_bins[1:]) /2
   
    n_bins_radius = np.arange(0,max_dist+min_dist,0.055)
    cumulative_counts_radius = np.zeros(len(n_bins_radius) - 1)
    bin_centers_radius = (n_bins_radius[:-1] + n_bins_radius[1:]) /2



    #for parquet_file in parquet_files:
    #    df = pd.read_parquet(f'{comptons_path}/{parquet_file}')
    #    cnts_electron, _ = np.histogram(df['E2'], bins = n_bins, range=(min_spectra_range, max_spectra_range))

    #    cnts_photon, _ = np.histogram(df['E1'], bins = n_bins, range=(min_spectra_range, max_spectra_range))
    #    cnts_radius, _ = np.histogram(df['distance_mm'], bins = n_bins_radius, range=(0, 10))
    #    cumulative_counts_electron += cnts_electron
    #    cumulative_counts_photon += cnts_photon
    #    cumulative_counts_radius += cnts_radius

    
    cumulative_counts_electron, _ = np.histogram(concat_df['E_elect'], bins = n_bins, range=(min_spectra_range, max_spectra_range))   
    cumulative_counts_photon, _ = np.histogram(concat_df['E_photon'], bins = n_bins, range=(min_spectra_range, max_spectra_range))
    cumulative_counts_radius, _ = np.histogram(concat_df['distance_mm'], bins = n_bins_radius, range=(0, max_dist +1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Plot energy spectrum (electrons and photons) on the first subplot
    ax1.plot(bin_centers_electron, cumulative_counts_electron, label='Compton Electrons', color='blue', linestyle='-', linewidth=2, drawstyle='steps-mid')
    ax1.plot(bin_centers_photon, cumulative_counts_photon, label='Photoelectrons', color='orange', linestyle='-', linewidth=2, drawstyle='steps-mid')
    ax1.set_xlim(min_spectra_range, max_spectra_range)
    ax1.set_xlabel('Energy (keV)', fontsize=14)
    ax1.set_ylabel('Counts', fontsize=14)
    ax1.set_title('Energy Spectrum', fontsize=16)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.legend(fontsize=12)

# Plot radius distribution on the second subplot
    ax2.plot(bin_centers_radius, cumulative_counts_radius, label='radius', color='green', linestyle='-', linewidth=2, drawstyle='steps-mid')
    ax2.set_xlim(0, max_dist+min_dist)
    ax2.set_xlabel('Distance (mm)', fontsize=14)
    ax2.set_ylabel('Counts', fontsize=14)
    ax2.set_title('Radius Distribution', fontsize=16)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.legend(fontsize=12)

# Adjust layout and display the plot
    plt.tight_layout()
    plt.savefig(f'{plot_save_folder}/comptonsUsed_characteristics.png')
    plt.close()
    
    energy_spectrum_data = pd.DataFrame({
        'Energy (keV)': bin_centers_electron,
        'Compton Electrons Counts': cumulative_counts_electron,
        'Photoelectrons Counts': cumulative_counts_photon
    })
    energy_spectrum_data.to_csv(f'{plot_save_folder}/comptonsUsed_characteristics_energy_spectrum_data.csv', index=False)

    radius_distribution_data = pd.DataFrame({
    'Distance (mm)': bin_centers_radius,
    'Radius Counts': cumulative_counts_radius
    })
    radius_distribution_data.to_csv(f'{plot_save_folder}/comptonsUsed_characteristics_radius_distribution_data.csv', index=False)
    #plt.show()
    plt.close()
    plt.close()
    
    del concat_df
    gc.collect()

def linear_func(x, m, c):
    return m * x + c


def plot_EventtypeEffsimulationAbsComptonvrsEnergy(outputFolder_base, sources, min_dist_list, angle_bin_list, max_dist):
    '''
    only performs analysis for souces with rot=0
    '''
    
    outputFolder = os.path.join(outputFolder_base, '3-GrenobleGeneralConclusions')
    energies_all_sources = []
    sim_eff_list = []
    
    dict_compton = {}
    dict_compton_all = {}
    dict_single = {}
    dict_double = {}
    dict_multiple = {}
    


    dict_ = {}
    dict_sigma = {}
    
    for source in sources:
        source_folder = os.path.join(outputFolder_base, source)

        source_energy = get_energy_from_source_name(source)
        sim_eff = get_sim_efficiency(source_energy)

        if source_energy not in energies_all_sources:
            energies_all_sources.append(source_energy)
        else:
            print('continue ENERGY check')
        
        source_rot = get_rot_from_source_name(source)
        
        if source_rot == 0:
            sim_eff_list.append(sim_eff)
            max_merit, best_min_dist, best_angle_bin, sigma_max_merit = get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max
            
            compton_events, compton_events_sigma = get_ComptonEvents_used(source_folder,
                                                                          best_min_dist,
                                                                          best_angle_bin,
                                                                          max_dist)

            compton_events_all, compton_events_all_sigma = get_ComptonEvents_used(source_folder,
                                                                                  0.055,
                                                                                  1,
                                                                                  max_dist)

            total_events, total_events_sigma = get_source_total_events(source_folder)

            single_events, single_events_sigma = get_source_single_events(source_folder)

            double_events, double_events_sigma = get_source_double_events(source_folder)

            multiple_events, multiple_events_sigma = get_source_multiple_events(source_folder)

            norm_single_events = single_events/total_events
            norm_single_events_sigma = error_propagation_fraction(single_events,
                                                                  single_events_sigma,
                                                                  total_events,
                                                                  total_events_sigma)
            norm_double_events = double_events/total_events
            norm_double_events_sigma = error_propagation_fraction(double_events,
                                                                  double_events_sigma,
                                                                  total_events,
                                                                  total_events_sigma)
            norm_multiple_events = multiple_events/total_events
            norm_multiple_events_sigma = error_propagation_fraction(multiple_events,
                                                                    multiple_events_sigma,
                                                                    total_events,
                                                                    total_events_sigma)
            norm_compton_events = compton_events/total_events
            norm_compton_events_sigma = error_propagation_fraction(compton_events,
                                                                    compton_events_sigma,
                                                                    total_events,
                                                                    total_events_sigma)

            norm_compton_events_all = compton_events_all/total_events
            norm_compton_events_all_sigma = error_propagation_fraction(compton_events_all,
                                                                       compton_events_all_sigma,
                                                                       total_events,
                                                                       total_events_sigma)
            

            dict_single[source_energy] = (norm_single_events, norm_single_events_sigma)
            dict_double[source_energy] = (norm_double_events, norm_double_events_sigma)
            dict_multiple[source_energy] = (norm_multiple_events, norm_multiple_events_sigma)
            dict_compton[source_energy] = (norm_compton_events, norm_compton_events_sigma)
            dict_compton_all[source_energy] = (norm_compton_events_all, norm_compton_events_all_sigma)

    x = []
    ysingle = []
    ysingle_err = []
    ydouble = []
    ydouble_err = []
    ymultiple = []
    ymultiple_err = []
    ycompton = []
    ycompton_err = []
    ycompton_all = []
    ycompton_all_err = []


    for energy in energies_all_sources:
        x.append(energy)
        
        ysingle.append(dict_single[energy][0])
        ysingle_err.append(dict_single[energy][1])
        
        ydouble.append(dict_double[energy][0])
        ydouble_err.append(dict_double[energy][1])
        
        ymultiple.append(dict_multiple[energy][0])
        ymultiple_err.append(dict_multiple[energy][1])
        
        ycompton.append(dict_compton[energy][0])
        ycompton_err.append(dict_compton[energy][1])

        ycompton_all.append(dict_compton_all[energy][0])
        ycompton_all_err.append(dict_compton_all[energy][1])

    fig, ax1 = plt.subplots(figsize=(8.5, 7.5))

    ax1.errorbar(x, ysingle, yerr=ysingle_err, fmt='o-', label='Single Events', capsize=3, color='k', markersize=9)
    ax1.errorbar(x, ydouble, yerr=ydouble_err, fmt='^-', label='Double Events', capsize=3, color='b', markersize=9)
    ax1.errorbar(x, ymultiple, yerr=ymultiple_err, fmt='s-', label='Multiple Events', capsize=3, color='r', markersize=9)
    ax1.errorbar(x, ycompton_all, yerr=ycompton_all_err, fmt='X-', label='All Compton Events', capsize = 3, color='g', markersize=11)
    ax1.errorbar(x, ycompton, yerr=ycompton_err, fmt='X-', label=r'Compton Events, best $F$', capsize=3, color='green', markersize=9, markerfacecolor = 'none')
    
    ax2 = ax1.twinx()
    sim_eff_array = np.array(sim_eff_list)
    ysingle_array = np.array(ysingle)
    ydouble_array = np.array(ydouble)
    ymultiple_array = np.array(ymultiple)
    ycompton_all_array = np.array(ycompton_all)
    ycompton_array = np.array(ycompton)

    #ax2.errorbar(x, sim_eff_array * ysingle_array, fmt='o--', label='Single Events (Sim)', color='k', markersize=10, markerfacecolor = 'none')
    #ax2.errorbar(x, sim_eff_array * ydouble_array, fmt='^--', label='Double Events (Sim)', color='b', markersize=10, markerfacecolor = 'none')
    #ax2.errorbar(x, sim_eff_array * ymultiple_array, fmt='s--', label='Multiple Events (Sim)', color='r', markersize=10, markerfacecolor = 'none')
    #ax2.errorbar(x, sim_eff_array * ycompton_all_array, fmt='x--', label='All Compton Events (Sim)', color='g', markersize=10, markerfacecolor = 'none')
    #ax2.errorbar(x, sim_eff_array * ycompton_array, fmt='X--', label=r'Compton Events, best $F$ (Sim)', color='orange', markersize=10, markerfacecolor = 'none')

    ax2.errorbar(x, sim_eff_list, fmt='d--', label='Simulated Efficiency', color='orange', markersize=9)

    ax1.set_xlabel('Energy (keV)')
    ax1.set_ylabel('Relative Efficiency')
    ax2.set_ylabel('Simulated Detection Efficiency')
    #ax2.tick_params(axis='y')

    ax1.set_ylim(0.001,1)
    #ax2.set_ylim(0.001,1)

    ax2.set_ylim(0,1)
       
    ax1.set_yscale('log')
    #ax2.set_yscale('log')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=12, bbox_to_anchor=(0.15,0), frameon=True)
    #ax1.legend(fontsize = 13, loc='lower left', bbox_to_anchor=(0.15,0))

    ax1.grid(False)
    ax2.grid(False)

    ax1.minorticks_on()
    ax2.minorticks_on()

    ax1.tick_params(axis='both', which='both', top=True, bottom=True, right = False)
    plt.tight_layout()
        
    plt.savefig(f'{outputFolder}/event_efficiencies.png')


def get_sim_efficiency(energy):
    sim_path = "/media/josesousa/joseHard/quad_characterization/2-Simulation/eff.txt"

    with open(sim_path) as f:
        a = f.readlines()
        for i in range(len(a)):
            if i == 0:
                continue
            line = a[i]
            split_lines = line.split(' ')
            if int(split_lines[0]) == energy:
                print(split_lines[0])
                return float(split_lines[-1])

def get_source_total_events(source_folder):
    event_counts_path = f'{source_folder}/AllEventsCount.txt'

    with open(event_counts_path) as f:
        line = f.readlines()
        line_totalEvents = line[0]
        line_totalEvents_split = line_totalEvents.split(' ')
        total_counts = int(line_totalEvents_split[-1])
        
        emmited_photons = total_counts
        sigma = np.sqrt(emmited_photons)
    return emmited_photons, sigma


def get_source_single_events(source_folder):
    event_counts_path = f'{source_folder}/AllEventsCount.txt'

    with open(event_counts_path) as f:
        line = f.readlines()
        line_singleEvents = line[1]
        line_singleEvents_split = line_singleEvents.split(' ')
        single_counts = int(line_singleEvents_split[-1])
        
        emmited_photons = single_counts
        sigma = np.sqrt(emmited_photons)
    return emmited_photons, sigma


def get_source_double_events(source_folder):
    event_counts_path = f'{source_folder}/AllEventsCount.txt'

    with open(event_counts_path) as f:
        line = f.readlines()
        line_doubleEvents = line[2]
        line_doubleEvents_split = line_doubleEvents.split(' ')
        double_counts = int(line_doubleEvents_split[-1])
        
        emmited_photons = double_counts
        sigma = np.sqrt(emmited_photons)
    return emmited_photons, sigma


def get_source_multiple_events(source_folder):
    event_counts_path = f'{source_folder}/AllEventsCount.txt'

    with open(event_counts_path) as f:
        line = f.readlines()
        line_multipleEvents = line[3]
        line_multipleEvents_split = line_multipleEvents.split(' ')
        multiple_counts = int(line_multipleEvents_split[-1])
        
        emmited_photons = multiple_counts
        sigma = np.sqrt(emmited_photons)
    return emmited_photons, sigma

def error_propagation_contantsSum(sigma_a, sigma_b):
    '''
    y = a + b
    '''
    sigma_y = np.sqrt(sigma_a**2 + sigma_b**2)
    return sigma_y

def error_propagation_fraction(a, sigma_a, b, sigma_b):
    '''
    y = a/b
    '''

    sigma_y = np.sqrt( (sigma_a/b)**2 + ((-a * sigma_b)/b**2)**2)

    return sigma_y






def imshow_eventType(output_folder_base, sources, min_dist = None, max_dist = 4.18, event_type = 'all', dist_cuts = False, plot_energy_source = None, plot_rot_source = None):
    """
    Generate energy deposition heatmaps for different event types (singles/doubles/comptons) 
    with optional distance cuts and save as PNG.
    At this stage the function only works for Grenoble data.

    Args:
        output_folder_base (str): Base directory to save plots and read data.
        sources (list): List of source names (e.g., ['grenoble_200kev_0deg', 'grenoble_100kev_90deg']).
        min_dist (float, optional): Minimum photon travel distance for Compton cuts. Default=None.
        max_dist (float, optional): Maximum photon travel distance for Compton cuts. Default=4.18.
        event_type (str): Event type to plot ('singles', 'doubles', 'comptons_inPeak', 'multiples', 'all').
        dist_cuts (bool): Apply distance cuts for Compton events. Default=False.
        plot_energy_source (float): Combines with rot_source, you select which source to plot....
        plot_rot_source (float): Filter sources by rotation angle (deg).
    """

    grenoble_results = f'{output_folder_base}/3-GrenobleGeneralConclusions'
    
    for source in sources:
        
        energy_source = get_energy_from_source_name(source)
        rot_source = get_rot_from_source_name(source)

        if rot_source != plot_rot_source:
            continue

        if energy_source != plot_energy_source:
            continue
        

        source_folder = f'{output_folder_base}/{source}'


        all_data_folder = f'{source_folder}/parquet'
        singles_folder = f'{all_data_folder}/singles'
        doubles_folder = f'{all_data_folder}/doubles'
        multiples_folder = f'{all_data_folder}/multiples'
        doubles_inpeak_folder = f'{doubles_folder}/inPeak'
        comptons_folder = f'{doubles_inpeak_folder}/comptons'

        list_all_data= pathlib.get_list_files(all_data_folder, endswith='.parquet')
        list_singles = pathlib.get_list_files(singles_folder, endswith='.parquet')
        list_doubles = pathlib.get_list_files(doubles_folder, endswith='.parquet')
        list_multiples = pathlib.get_list_files(multiples_folder, endswith='.parquet')
        list_inPeak = pathlib.get_list_files(doubles_inpeak_folder, endswith='.parquet')
        list_comptons = pathlib.get_list_files(comptons_folder, endswith='.parquet')


        def extract_id(filename):
            match = re.search(r'r(\d+)', filename)
            return int(match.group(1)) if match else 0

        list_all_data_sorted = sorted(list_all_data, key=extract_id)
        list_singles_sorted = sorted(list_singles, key=extract_id)
        list_doubles_sorted = sorted(list_doubles, key=extract_id)
        list_multiples_sorted = sorted(list_multiples, key=extract_id)
        list_inPeak_sorted = sorted(list_inPeak, key=extract_id)
        list_comptons_sorted = sorted(list_comptons, key=extract_id)


        if event_type == 'comptons_inPeak':
            if dist_cuts == True:
                pixel_matrix = imshow_det_pixelMatrix_typeEvent(all_data_folder, list_all_data_sorted, comptons_folder, list_comptons_sorted, min_dist=min_dist, max_dist=max_dist, cuts = True) 

            else:
                pixel_matrix = imshow_det_pixelMatrix_typeEvent(all_data_folder, list_all_data_sorted, comptons_folder, list_comptons_sorted, cuts = True)

        elif event_type == 'singles':
                pixel_matrix = imshow_det_pixelMatrix_typeEvent(all_data_folder, list_all_data_sorted, singles_folder, list_singles_sorted)

        elif event_type == 'doubles':
                pixel_matrix = imshow_det_pixelMatrix_typeEvent(all_data_folder, list_all_data_sorted, doubles_folder, list_doubles_sorted)

        elif event_type == 'multiples':
                pixel_matrix = imshow_det_pixelMatrix_typeEvent(all_data_folder, list_all_data_sorted, multiples_folder, list_multiples_sorted)

        elif event_type == 'all':
                pixel_matrix = imshow_det_pixelMatrix_typeEvent(all_data_folder, list_all_data_sorted, all_data_folder, list_all_data_sorted)



        x_energy_deposited = np.sum(pixel_matrix, axis=0)
        y_energy_deposited = np.sum(pixel_matrix, axis=1)


        x_interesting_points = detect_sudden_changes(x_energy_deposited, threshold=5.0)
        y_interesting_points = detect_sudden_changes(y_energy_deposited, threshold=5.0)

        x_max = max(x_interesting_points) + 1
        x_min = min(x_interesting_points)

        y_max = max(y_interesting_points) + 2
        y_min = min(y_interesting_points) - 1

        chip_id = 2

        x_matrix_Ledge = int(chip_id) * 256
        x_matrix_Redge = int(chip_id) * 256 + 255

        fig = plt.figure(figsize=(7, 7))

        energy_source = get_energy_from_source_name(source)

        vmin, vmax = 100, 10e4  # Your current LogNorm bounds

        #cmap = cc.m_fire # Correct way to access colocet colormaps

        ax_img = plt.gcf().add_axes((0.13, 0.1, 0.7, 0.7))
        im = ax_img.imshow(pixel_matrix, origin='lower', cmap='jet' , aspect='auto', 
                   norm=LogNorm(vmin=vmin, vmax=vmax))

        #cbar = plt.colorbar(im, ax=ax_img, orientation='vertical', pad=0.01)
        #cbar.set_label(f'{energy_source} keV')
        rect = patches.Rectangle((x_max + 47, y_max + 65), 27, 13, 
                                 linewidth=1.5, edgecolor='black', facecolor='white', alpha=1)
        ax_img.add_patch(rect)

# Annotate inside the rectangle
        ax_img.annotate(f'{energy_source} keV', (x_max + 50, y_max + 70), 
                        color='black', fontsize=14, ha='left', va='center')

        cbar_ax = fig.add_axes((0.931, 0.105, 0.03, 0.696))  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(r'Energy per Pixel (keV)')

# Choose the number of decades (log steps) you want
        num_ticks = 4
        ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=num_ticks)

# Set the ticks and format them properly
        cbar.set_ticks(ticks)
        #cbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
        cbar.set_ticklabels([f"$10^{{{int(np.log10(tick))}}}$" for tick in ticks])  # Formats as 10², 10³, etc.
        cbar.formatter = ticker.ScalarFormatter(useMathText=True)  # Enables scientific notation
        cbar.formatter.set_powerlimits((0, 0))  # Force scientific notation for all ticks
        cbar.update_ticks()  # Apply changes
# Add colorbar
        #cbar.set_label(f'{energy_source} keV')
        #ax_img.imshow(
        #pixel_matrix, 
        #origin='lower', 
        #cmap='tab20c', 
        #aspect='auto', 
        #vmin=200, 
        #vmax=30e3
      # )

        #ax_img._colorbars(label='Cumulative ToT (keV)', orientation='horizontal')
        #ax_img.set_xlim(x_matrix_Ledge, x_matrix_Redge)
        ax_img.set_xlim(x_min-80, x_max+80)
        ax_img.set_ylim(y_min-80, y_max+80)
        ax_img.set_xlabel(r'X-position [pixel \#]')
        ax_img.set_ylabel(r'Y-position [pixel \#]')
        #ax_img.set_title(f'Cumulative Pixel Energy Heatmap for Source: {source}')
        ax_img.grid(False)

# Add vertical lines as markers
        for x in [256, 512, 768]:
            plt.axvline(x=x, color='red', linestyle='--', linewidth=1, label=f'x={x}')
        
        #plt.axvline(x=x_min, color = '#C6B800', linewidth=2)
        #plt.axvline(x=x_max, color ='#F6E800' , linewidth=2)
        #plt.axhline(y=y_min, color = '#F6E800', linewidth=2)
        #plt.axhline(y=y_max, color = '#F6E800', linewidth=2)

# Add inset for x-energy histogram
        ax_x = plt.gcf().add_axes((0.13, 0.8, 0.7, 0.1))  # [left, bottom, width, height]
        #ax_x.bar(range(len(x_energy_deposited)), x_energy_deposited, color='black', alpha=0.7)
        ax_x.fill_between(range(len(x_energy_deposited)), x_energy_deposited, color='black', alpha=0.5)
        #x_heatmap = np.expand_dims(x_energy_deposited, axis=0)  # Convert to 2D for imshow
        #ax_x.imshow(x_heatmap, aspect='auto', cmap='Greys', extent=[0, len(x_energy_deposited), 0, 1], norm=LogNorm(vmin=1, vmax=np.max(x_heatmap)))
        ax_x.set_xlim(x_min-50, x_max+50)
        #ax_x.set_yticks([])
        ticks = []
        ax_x.set_xticks(ticks)
        ax_x.set_yticks(ticks)
        ax_x.set_yscale('log')
        ax_x.set_ylabel('(keV)', fontsize = 15)
        #ax_x.set_title(f'Energy Deposition Map for Source: {source}')
        ax_x.grid(False)

# Add inset for y-energy histogram
        ax_y = plt.gcf().add_axes((0.83, 0.1, 0.1, 0.7))  # [left, bottom, width, height]
        #ax_y.barh(range(len(y_energy_deposited)), y_energy_deposited, color='black', alpha=0.7)
        ax_y.fill_between(y_energy_deposited, range(len(y_energy_deposited)), color='black', alpha=0.5)
        ax_y.set_ylim(y_min-50, y_max+50)
        ax_y.set_xticks(ticks)
        ax_y.set_yticks(ticks)
        ax_y.set_xscale('log')
        ax_y.set_xlabel('(keV)', fontsize = 15)
        #ax_y.set_title('Y-Axis Energy Distribution', rotation=90)

        plt.grid(False)

        plt.subplots_adjust(left = 1.1, right=2)

        if dist_cuts:
            plt.savefig(f'{grenoble_results}/imshow_{energy_source}keV_{rot_source}deg_{event_type}_cuts.png', bbox_inches = 'tight')
        else:
            plt.savefig(f'{grenoble_results}/imshow_{energy_source}keV_{rot_source}deg_{event_type}.png', bbox_inches = 'tight')
        plt.close()


def imshow_det_pixelMatrix_typeEvent(all_data_folder, list_all_data_sorted, typeEvent_folder, list_typeEvent_sorted, min_dist = 0.055, max_dist = 4.18, cuts = False):

    pixel_matrix = np.ones((256,1024))

    for i, file in enumerate(list_all_data_sorted):
        print(file)

        df_all_data = pd.read_parquet(f'{all_data_folder}/{file}')
        df_typeEvent = pd.read_parquet(f'{typeEvent_folder}/{list_typeEvent_sorted[i]}')


        if cuts == True:
            df_compton_limit = limit_dist_photon_travel(df_typeEvent, min_dist, max_dist)
            event_ids_in_dfcompton = df_compton_limit['Event'].unique()
            filtered_dfpeak = df_all_data[df_all_data['Event'].isin(event_ids_in_dfcompton)]

        else:
            events_in_typeEvent = df_typeEvent['Event'].unique()
            filtered_dfpeak = df_all_data[df_all_data['Event'].isin(events_in_typeEvent)]

        filtered_dfpeak = filtered_dfpeak.reset_index(drop=True)
        grouped_df = filtered_dfpeak.groupby(['X', 'Y'])['ToT (keV)'].sum().reset_index()

        for _, row in grouped_df.iterrows():
            x, y, tot = int(row['X']) - 2*256, int(row['Y']), row['ToT (keV)']
            if x == 530 and y == 32:
                continue
            pixel_matrix[y, x] += tot

    return pixel_matrix


def plot_comptonEventsSpectra(output_folder_base, sources, min_dist_list = [], angle_bin_list =[], min_dist = 0.055, max_dist= 4.18, plot_energy_source = 100, plot_rot_source = 0):

    
    outputFolder = os.path.join(output_folder_base, '3-GrenobleGeneralConclusions')
    
    for source in sources:
        
        energy_source = get_energy_from_source_name(source)
        rot_source = get_rot_from_source_name(source)

        if rot_source != plot_rot_source:
            continue

        if energy_source != plot_energy_source:
            continue
        

        source_folder = f'{output_folder_base}/{source}'


        all_data_folder = f'{source_folder}/parquet'
        doubles_folder = f'{all_data_folder}/doubles'
        doubles_inpeak_folder = f'{doubles_folder}/inPeak'
        comptons_folder = f'{doubles_inpeak_folder}/comptons'

        list_all_data= pathlib.get_list_files(all_data_folder, endswith='.parquet')
        list_comptons = pathlib.get_list_files(comptons_folder, endswith='.parquet')


        def extract_id(filename):
            match = re.search(r'r(\d+)', filename)
            return int(match.group(1)) if match else 0

        list_all_data_sorted = sorted(list_all_data, key=extract_id)
        list_comptons_sorted = sorted(list_comptons, key=extract_id)
        
        _, min_dist_bestF, _ , _ = get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist)
        counts_matrix_bestF, theta_matrix_bestF, bins_ph_bestF, bins_elec_bestF  = get_ComptonEventsEnergyMatrix(all_data_folder, list_all_data_sorted, comptons_folder, list_comptons_sorted, min_dist = min_dist_bestF, max_dist=max_dist, cuts=True)
        
        max_dist_all = 10000
        counts_matrix_all, theta_matrix_all, bins_ph_all, bins_elec_all  = get_ComptonEventsEnergyMatrix(all_data_folder, list_all_data_sorted, comptons_folder, list_comptons_sorted, min_dist = min_dist, max_dist=max_dist_all, cuts=True)
        
        theta_matrix_xSize_bestF = theta_matrix_bestF.shape[0]
        theta_matrix_ySize_bestF = theta_matrix_bestF.shape[1]
        
        list_thetas_bestF = []
        list_cnts_bestF = []
        list_ph_tmp_bestF = []
        list_elec_tmp_bestF = []

        for i in range(theta_matrix_xSize_bestF):
            for j in range(theta_matrix_ySize_bestF):
                list_thetas_bestF.append(theta_matrix_bestF[i,j])
                list_cnts_bestF.append(counts_matrix_bestF[i,j])

        counts_matrix_xSize_bestF = counts_matrix_bestF.shape[0]
        counts_matrix_ySize_bestF = counts_matrix_bestF.shape[1]

        dict_ph_bestF = {i: [] for i in range(counts_matrix_xSize_bestF)}
        dict_elec_bestF = {j: [] for j in range(counts_matrix_ySize_bestF)}

        for i in range(counts_matrix_xSize_bestF):
            for j in range(counts_matrix_ySize_bestF):
                dict_ph_bestF[i].append(counts_matrix_bestF[i,j]/10000)
                dict_elec_bestF[j].append(counts_matrix_bestF[i,j]/10000)

        list_ph_cnts_bestF = [np.sum(vals) for vals in dict_ph_bestF.values()]
        list_elec_cnts_bestF = [np.sum(vals) for vals in dict_elec_bestF.values()]

# Get the energy bin indices (keys)
        photon_bins_bestF = list(dict_ph_bestF.keys())
        electron_bins_bestF = list(dict_elec_bestF.keys())


        theta_matrix_xSize_all= theta_matrix_all.shape[0]
        theta_matrix_ySize_all= theta_matrix_all.shape[1]
        
        list_thetas_all= []
        list_cnts_all= []
        list_ph_tmp_all= []
        list_elec_tmp_all= []

        for i in range(theta_matrix_xSize_all):
            for j in range(theta_matrix_ySize_all):
                list_thetas_all.append(theta_matrix_all[i,j])
                list_cnts_all.append(counts_matrix_all[i,j])

        counts_matrix_xSize_all = counts_matrix_all.shape[0]
        counts_matrix_ySize_all = counts_matrix_all.shape[1]

        dict_ph_all = {i: [] for i in range(counts_matrix_xSize_all)}
        dict_elec_all = {j: [] for j in range(counts_matrix_ySize_all)}

        for i in range(counts_matrix_xSize_all):
            for j in range(counts_matrix_ySize_all):
                dict_ph_all[i].append(counts_matrix_all[i,j]/10000)
                dict_elec_all[j].append(counts_matrix_all[i,j]/10000) #divide by 10000 for better y axis plot units


        list_ph_cnts_all = [np.sum(vals) for vals in dict_ph_all.values()]         
        list_elec_cnts_all = [np.sum(vals) for vals in dict_elec_all.values()]

# Get the energy bin indices (keys)
        photon_bins_all = list(dict_ph_all.keys())
        electron_bins_all = list(dict_elec_all.keys())
        
        fig, ax1 = plt.subplots(1, 1, figsize=(5,5))
        ax1.plot(electron_bins_bestF, list_elec_cnts_bestF, label='Compton Electrons (best F)', color='r', linewidth=2, drawstyle='steps-mid', linestyle = '--')
        ax1.plot(photon_bins_bestF, list_ph_cnts_bestF, label='Photoelectrons (best F)', color='g', linewidth=2, drawstyle='steps-mid', linestyle = '--')
        y_max = np.max(list_elec_cnts_bestF)
        ax1.set_ylim(0,y_max*1.2)

        ax1.set_xlabel('Energy (keV)')
        ax1.set_ylabel(r'Counts ($\times 10^{4}$)')
        ax1.minorticks_on()
        ax1.legend()
        ax1.grid(False)

        ax2 = ax1.twiny()

        ax2.plot(electron_bins_all, list_elec_cnts_all, label='Compton Electrons (all)', color='k', linewidth=2, drawstyle='steps-mid', linestyle = '-')
        ax2.plot(photon_bins_all, list_ph_cnts_all, label='Photoelectrons (all)', color='b', linewidth=2, drawstyle='steps-mid', linestyle = '-')
        y_max = np.max(list_elec_cnts_all)
        ax2.set_ylim(0,y_max*1.2)

        #ax2.set_xticks(False)
        ax2.set_xticklabels([])

        ax2.set_ylabel('Counts (Event type)')
        ax2.minorticks_on()
        ax2.grid(False)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize = 10)

        #plt.title(fr'{energy_source} keV, rot = {rot_source}$^{{\circ}}$, $r_{{min}}$ = {round(min_dist,3)} mm, $r_{{max}}$ = {max_dist} mm', fontsize = 16)
        ax1.grid(False)
        ax2.grid(False)
        ax2.grid(False)
        

        ax1.tick_params(direction='in', axis='both', which='both', top=True, bottom=True, right=True)
        ax2.tick_params(direction='in', axis='both', which='both', top=True, bottom=True, right=True)
        #plt.title(f'Source: {energy_source} keV')
        plt.tight_layout()
        plt.minorticks_on()
        #plt.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True)

        plt.savefig(f'{outputFolder}/Compton_energydistribution_{energy_source}keV_{rot_source}deg_bestF-All.png', dpi=600, bbox_inches='tight')


def get_compton_spectra(source_folder, angle_bin, min_dist, max_dist):
    '''
    Goes to a given source folder that has been analysied for polarimetry for a given polarimetry phi bin, min_dist and max_dist. It reads the spectrum folder and returns the energy bins, electrons cnts, photons cnts. (histogram)
    '''
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    folder = f'{source_folder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'

    file = f'{folder}/comptonsUsed_characteristics_energy_spectrum_data.csv'
    
    energy_bin = []
    electron_cnts = []
    photon_cnts = []

    with open(file) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if "Energy" in line:
                continue
            split_lines = line.split(',')
            energy_bin.append(float(split_lines[0]))
            electron_cnts.append(int(split_lines[1]))
            photon_cnts.append(int(split_lines[2]))


    return energy_bin, electron_cnts, photon_cnts

def plot_comptonEventsThetaDistribution(output_folder_base, sources, min_dist_list = [], angle_bin_list =[], min_dist = 0.055, max_dist= 4.18, plot_energy_source = 100, plot_rot_source = 0):

    
    outputFolder = os.path.join(output_folder_base, '3-GrenobleGeneralConclusions')
    
    for source in sources:
        
        energy_source = get_energy_from_source_name(source)
        rot_source = get_rot_from_source_name(source)

        if rot_source != plot_rot_source:
            continue

        if energy_source != plot_energy_source:
            continue
        

        source_folder = f'{output_folder_base}/{source}'


        all_data_folder = f'{source_folder}/parquet'
        doubles_folder = f'{all_data_folder}/doubles'
        doubles_inpeak_folder = f'{doubles_folder}/inPeak'
        comptons_folder = f'{doubles_inpeak_folder}/comptons'

        list_all_data= pathlib.get_list_files(all_data_folder, endswith='.parquet')
        list_comptons = pathlib.get_list_files(comptons_folder, endswith='.parquet')


        def extract_id(filename):
            match = re.search(r'r(\d+)', filename)
            return int(match.group(1)) if match else 0

        list_all_data_sorted = sorted(list_all_data, key=extract_id)
        list_comptons_sorted = sorted(list_comptons, key=extract_id)
        
        _, min_dist_bestF, _ , _ = get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist)
        counts_matrix_bestF, theta_matrix_bestF, bins_ph_bestF, bins_elec_bestF  = get_ComptonEventsEnergyMatrix(all_data_folder, list_all_data_sorted, comptons_folder, list_comptons_sorted, min_dist = min_dist_bestF, max_dist=max_dist, cuts=True)
        
        max_dist_all = 10000
        counts_matrix_all, theta_matrix_all, bins_ph_all, bins_elec_all  = get_ComptonEventsEnergyMatrix(all_data_folder, list_all_data_sorted, comptons_folder, list_comptons_sorted, min_dist = min_dist, max_dist=max_dist_all, cuts=True)
        
        theta_matrix_xSize_bestF = theta_matrix_bestF.shape[0]
        theta_matrix_ySize_bestF = theta_matrix_bestF.shape[1]
        
        list_thetas_bestF = []
        list_cnts_bestF = []
        list_ph_tmp_bestF = []
        list_elec_tmp_bestF = []

        for i in range(theta_matrix_xSize_bestF):
            for j in range(theta_matrix_ySize_bestF):
                list_thetas_bestF.append(theta_matrix_bestF[i,j])
                list_cnts_bestF.append(counts_matrix_bestF[i,j])

        counts_matrix_xSize_bestF = counts_matrix_bestF.shape[0]
        counts_matrix_ySize_bestF = counts_matrix_bestF.shape[1]

        dict_ph_bestF = {i: [] for i in range(counts_matrix_xSize_bestF)}
        dict_elec_bestF = {j: [] for j in range(counts_matrix_ySize_bestF)}

        for i in range(counts_matrix_xSize_bestF):
            for j in range(counts_matrix_ySize_bestF):
                dict_ph_bestF[i].append(counts_matrix_bestF[i,j])
                dict_elec_bestF[j].append(counts_matrix_bestF[i,j])

        list_ph_cnts_bestF = [np.sum(vals) for vals in dict_ph_bestF.values()]
        list_elec_cnts_bestF = [np.sum(vals) for vals in dict_elec_bestF.values()]

# Get the energy bin indices (keys)
        photon_bins_bestF = list(dict_ph_bestF.keys())
        electron_bins_bestF = list(dict_elec_bestF.keys())


        theta_matrix_xSize_all= theta_matrix_all.shape[0]
        theta_matrix_ySize_all= theta_matrix_all.shape[1]
        
        list_thetas_all= []
        list_cnts_all= []
        list_ph_tmp_all= []
        list_elec_tmp_all= []

        for i in range(theta_matrix_xSize_all):
            for j in range(theta_matrix_ySize_all):
                list_thetas_all .append(theta_matrix_all[i,j])
                list_cnts_all.append(counts_matrix_all[i,j])

        counts_matrix_xSize_all = counts_matrix_all.shape[0]
        counts_matrix_ySize_all = counts_matrix_all.shape[1]

        dict_ph_all = {i: [] for i in range(counts_matrix_xSize_all)}
        dict_elec_all = {j: [] for j in range(counts_matrix_ySize_all)}

        for i in range(counts_matrix_xSize_all):
            for j in range(counts_matrix_ySize_all):
                dict_ph_all[i].append(counts_matrix_all[i,j])
                dict_elec_all[j].append(counts_matrix_all[i,j])

        list_ph_cnts_all = [np.sum(vals) for vals in dict_ph_all.values()]
        list_elec_cnts_all = [np.sum(vals) for vals in dict_elec_all.values()]

# Get the energy bin indices (keys)
        photon_bins_all = list(dict_ph_all.keys())
        electron_bins_all = list(dict_elec_all.keys())

        thetas_bestF = np.array(list_thetas_bestF)
        counts_bestF = np.array(list_cnts_bestF)

        n_bins = 60
        bin_edges_bestF = np.linspace(0, 180, n_bins)  

        bin_indices_bestF = np.digitize(thetas_bestF, bin_edges_bestF)

        binned_counts_bestF = np.zeros(len(bin_edges_bestF)-1)
        for i in range(1, len(bin_edges_bestF)):
            mask = (bin_indices_bestF == i)
            binned_counts_bestF[i-1] = counts_bestF[mask].sum()

        bin_centers_bestF = (bin_edges_bestF[:-1] + bin_edges_bestF[1:]) / 2
        x_ticks = np.arange(0,180 + 15, 15)
    
        
        thetas_all = np.array(list_thetas_all)
        counts_all = np.array(list_cnts_all)

        n_bins = 60
        bin_edges_all = np.linspace(0, 180, n_bins)  

        bin_indices_all = np.digitize(thetas_all, bin_edges_all)

        binned_counts_all = np.zeros(len(bin_edges_all)-1)
        for i in range(1, len(bin_edges_all)):
            mask = (bin_indices_all == i)
            binned_counts_all[i-1] = counts_all[mask].sum()

        bin_centers_all = (bin_edges_all[:-1] + bin_edges_all[1:]) / 2
        x_ticks = np.arange(0,180 + 15, 15)

        max_count_all = np.max(binned_counts_all)
        upper_limit_all = 10**np.ceil(np.log10(max_count_all)+1)  # Next power of 10


        fig, ax1 = plt.subplots(1, 1, figsize=(5,5))

        max_count_bestF = np.max(binned_counts_bestF)
        upper_limit_bestF = 10**np.ceil(np.log10(max_count_bestF)+1)  # Next power of 10

        ax1.plot(bin_centers_bestF, binned_counts_bestF, label=r'Best F', drawstyle = 'steps-mid', linewidth = 2, color='b', linestyle = '--')
        ax1.set_xlabel(fr'Polar scatter angle, $\theta$ ($^{{\circ}}$)', labelpad = 8)
        ax1.set_ylabel(r'Counts')
        ax1.tick_params(axis='y')
        ax1.set_yscale('log')

        #ax1.set_ylim(bottom=0.9 if max_count_bestF > 0 else 1,  # Avoid log(0) issues
        #     top=upper_limit_bestF)
        lines1, labels1 = ax1.get_legend_handles_labels()

        ax2 = ax1.twiny()

        ax2.plot(bin_centers_all, binned_counts_all, label=r'All', drawstyle = 'steps-mid', linewidth = 2, color='k', linestyle = '-')
        ax2.tick_params(axis='y')
        ax2.set_yscale('log')
        #ax1.set_ylim(bottom=0.9 if max_count_all > 0 else 1,  # Avoid log(0) issues
        #     top=upper_limit_all)
        lines2, labels2 = ax2.get_legend_handles_labels()
        
        #plt.title(fr'{energy_source} keV, rot = {rot_source}$^{{\circ}}$, $r_{{min}}$ = {round(min_dist,3)} mm, $r_{{max}}$ = {max_dist} mm', fontsize = 16)
        ax1.grid(False)
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize = 10)
        ax2.grid(False)
        
        list_xticks = np.arange(0,210,30)
        ax1.set_xticks(list_xticks)
        ax2.set_xticks(list_xticks)
        ax2.set_xticklabels([])
        ax1.tick_params(axis='y', which='both', left=True, right=True, labelleft=True, bottom = False)
        #ax2.tick_params(axis='y', which='both', right=True, labelright=True)

        ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=5))
        ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=5))
        ax1.minorticks_on()
        ax2.minorticks_on()

        #plt.title(f'Source: {energy_source} keV')
        plt.tight_layout()
        plt.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True, left=True)

        plt.xticks(x_ticks)
        
        plt.savefig(f'{outputFolder}/Compton_Thetadistribution_{energy_source}keV_{rot_source}deg_bestF-all.png', dpi=600, bbox_inches='tight')

def plot_comptonEventsSpectraThetadistribution(output_folder_base, sources, min_dist_list = [], angle_bin_list =[], min_dist = 0.055, max_dist= 100000, plot_energy_source = 100, plot_rot_source = 0, bestF = False):
    '''
    if argument 'bestF = True' user DOES NOT NEED to give 'min_dist' and 'max_dist', user NEEDS to give 'min_dist_list' and 'angle_bin_list'
    if argument 'bestF = False' iser DOES NOT NEED to give 'min_dist_list' and 'angle_bin_list', user NEEDS to give 'min_dist' and 'max_dist'
    
    'plot_energy_source' - write the energy of the source. Only works if 'sources' list has the energy on its name.
    'plot_rot_source' - writhe the rotation of the detector. Only works if 'sources' list has the rotation on its name.
    '''
    grenoble_results = f'{output_folder_base}/3-GrenobleGeneralConclusions'
    
    for source in sources:
        
        energy_source = get_energy_from_source_name(source)
        rot_source = get_rot_from_source_name(source)

        if rot_source != plot_rot_source:
            continue

        if energy_source != plot_energy_source:
            continue
        

        source_folder = f'{output_folder_base}/{source}'


        all_data_folder = f'{source_folder}/parquet'
        doubles_folder = f'{all_data_folder}/doubles'
        doubles_inpeak_folder = f'{doubles_folder}/inPeak'
        comptons_folder = f'{doubles_inpeak_folder}/comptons'

        list_all_data= pathlib.get_list_files(all_data_folder, endswith='.parquet')
        list_comptons = pathlib.get_list_files(comptons_folder, endswith='.parquet')


        def extract_id(filename):
            match = re.search(r'r(\d+)', filename)
            return int(match.group(1)) if match else 0

        list_all_data_sorted = sorted(list_all_data, key=extract_id)
        list_comptons_sorted = sorted(list_comptons, key=extract_id)
        
        if bestF == True:
            max_dist = 4.18
            _, min_dist, _ , _ = get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist)
            counts_matrix, theta_matrix, bins_ph, bins_elec  = get_ComptonEventsEnergyMatrix(all_data_folder, list_all_data_sorted, comptons_folder, list_comptons_sorted, min_dist = min_dist, max_dist=max_dist, cuts=True)
        else:
            counts_matrix, theta_matrix, bins_ph, bins_elec  = get_ComptonEventsEnergyMatrix(all_data_folder, list_all_data_sorted, comptons_folder, list_comptons_sorted, min_dist = min_dist, max_dist=max_dist, cuts=True)
        
        theta_matrix_xSize = theta_matrix.shape[0]
        theta_matrix_ySize = theta_matrix.shape[1]
        
        list_thetas = []
        list_cnts = []
        list_ph_tmp = []
        list_elec_tmp = []

        for i in range(theta_matrix_xSize):
            for j in range(theta_matrix_ySize):
                list_thetas.append(theta_matrix[i,j])
                list_cnts.append(counts_matrix[i,j])

        counts_matrix_xSize = counts_matrix.shape[0]
        counts_matrix_ySize = counts_matrix.shape[1]

        dict_ph = {i: [] for i in range(counts_matrix_xSize)}
        dict_elec = {j: [] for j in range(counts_matrix_ySize)}

        for i in range(counts_matrix_xSize):
            for j in range(counts_matrix_ySize):
                dict_ph[i].append(counts_matrix[i,j])
                dict_elec[j].append(counts_matrix[i,j])

        list_ph_cnts = [np.sum(vals) for vals in dict_ph.values()]
        list_elec_cnts = [np.sum(vals) for vals in dict_elec.values()]

# Get the energy bin indices (keys)
        photon_bins = list(dict_ph.keys())
        electron_bins = list(dict_elec.keys())
        
        fig, ax1 = plt.subplots(1, 1, figsize=(8,7))
        ax1.plot(electron_bins, list_elec_cnts, label='Compton Electrons', color='k', linewidth=2, drawstyle='steps-mid')
        ax1.plot(photon_bins, list_ph_cnts, label='Photoelectrons', color='b', linewidth=2, drawstyle='steps-mid')
        y_max = np.max(list_elec_cnts)
        ax1.set_ylim(0,y_max*1.2)

        ax1.set_xlabel('Energy (keV)')
        ax1.set_ylabel('Counts (Event type)')
        ax1.minorticks_on()
        ax1.legend()
        ax1.grid(False)

        thetas = np.array(list_thetas)
        counts = np.array(list_cnts)

        n_bins = 60
        bin_edges = np.linspace(0, 180, n_bins)  

        bin_indices = np.digitize(thetas, bin_edges)

        binned_counts = np.zeros(len(bin_edges)-1)
        for i in range(1, len(bin_edges)):
            mask = (bin_indices == i)
            binned_counts[i-1] = counts[mask].sum()

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        x_ticks = np.arange(0,180 + 15, 15)
    
        
        ax2 = ax1.twiny()
        ax3 = ax2.twinx()

        max_count = np.max(binned_counts)
        upper_limit = 10**np.ceil(np.log10(max_count)+1)  # Next power of 10

        ax3.plot(bin_centers, binned_counts, label=r'Photon Scatter Angle, $\theta$', drawstyle = 'steps-mid', linewidth = 2, color='r', linestyle = '--')
        ax2.set_xlabel(fr'$\theta$ ({180/n_bins}$^{{\circ}}$ bin)', labelpad = 8)
        ax3.set_ylabel(r'Counts ($\theta$)')
        ax3.tick_params(axis='y')
        ax3.set_yscale('log')
        ax3.set_ylim(bottom=0.9 if max_count > 0 else 1,  # Avoid log(0) issues
             top=upper_limit)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize = 14, ncol=2)

        #plt.title(fr'{energy_source} keV, rot = {rot_source}$^{{\circ}}$, $r_{{min}}$ = {round(min_dist,3)} mm, $r_{{max}}$ = {max_dist} mm', fontsize = 16)
        ax1.grid(False)
        ax2.grid(False)
        ax3.grid(False)

        plt.tight_layout()
        plt.minorticks_on()
        plt.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True)
        plt.xticks(x_ticks)

        if bestF == True:
            plt.savefig(f'{grenoble_results}/photonTheta_distribution_{energy_source}keV_{rot_source}deg_bestF.png')
        else:
            plt.savefig(f'{grenoble_results}/photonTheta_distribution_{energy_source}keV_{rot_source}deg_md{min_dist}_maxd{max_dist}.png')
        plt.close()


def plot_comptonEventsEnergyMatrixdistribution(output_folder_base, sources, min_dist_list = [], angle_bin_list =[], min_dist = 0.055, max_dist= 100000, plot_energy_source = 100, plot_rot_source = 0, bestF = False):
    '''
    if argument 'bestF = True' user DOES NOT NEED to give 'min_dist' and 'max_dist', user NEEDS to give 'min_dist_list' and 'angle_bin_list'
    if argument 'bestF = False' iser DOES NOT NEED to give 'min_dist_list' and 'angle_bin_list', user NEEDS to give 'min_dist' and 'max_dist'
    
    'plot_energy_source' - write the energy of the source. Only works if 'sources' list has the energy on its name.
    'plot_rot_source' - writhe the rotation of the detector. Only works if 'sources' list has the rotation on its name.
    '''
    grenoble_results = f'{output_folder_base}/3-GrenobleGeneralConclusions'
    
    for source in sources:
        
        energy_source = get_energy_from_source_name(source)
        rot_source = get_rot_from_source_name(source)

        if rot_source != plot_rot_source:
            continue

        if energy_source != plot_energy_source:
            continue
        

        source_folder = f'{output_folder_base}/{source}'


        all_data_folder = f'{source_folder}/parquet'
        doubles_folder = f'{all_data_folder}/doubles'
        doubles_inpeak_folder = f'{doubles_folder}/inPeak'
        comptons_folder = f'{doubles_inpeak_folder}/comptons'

        list_all_data= pathlib.get_list_files(all_data_folder, endswith='.parquet')
        list_comptons = pathlib.get_list_files(comptons_folder, endswith='.parquet')


        def extract_id(filename):
            match = re.search(r'r(\d+)', filename)
            return int(match.group(1)) if match else 0

        list_all_data_sorted = sorted(list_all_data, key=extract_id)
        list_comptons_sorted = sorted(list_comptons, key=extract_id)
        
        if bestF == True:
            max_dist = 4.18
            _, min_dist, _ , _ = get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist)
            counts_matrix, theta_matrix, bins_ph, bins_elec  = get_ComptonEventsEnergyMatrix(all_data_folder, list_all_data_sorted, comptons_folder, list_comptons_sorted, min_dist = min_dist, max_dist=max_dist, cuts=True)
        else:
            counts_matrix, theta_matrix, bins_ph, bins_elec  = get_ComptonEventsEnergyMatrix(all_data_folder, list_all_data_sorted, comptons_folder, list_comptons_sorted, min_dist = min_dist, max_dist=max_dist, cuts=True)
        
        plt.figure(figsize=(8, 7))

        cmap = plt.get_cmap('gist_stern').copy()
        cmap.set_bad('white',1)
        im = plt.imshow(
            counts_matrix.T,
            origin='lower',
            extent=[bins_ph[0], bins_ph[-1], bins_elec[0], bins_elec[-1]],
            cmap= cmap,  
            aspect='auto',
            alpha=0.9  
        )

        xmin = compton_photon(int(energy_source), 180)
        xmax =  compton_photon(int(energy_source), 0)
        ymin = energy_electron(int(energy_source), 0)
        ymax = energy_electron(int(energy_source),180)
        ymax = counts_matrix.shape[1]
        plt.xlim(xmin - xmin * 0.1,xmax + xmin * 0.3)
        plt.ylim(ymin,ymax)
        plt.xlabel("Photon Energy (keV)")
        plt.ylabel("Electron Energy (keV)")
        cbar = plt.colorbar(im, label="Compton Scattering Angle  (deg)")
        cbar.set_label("Counts")

        plt.grid(False)  # Disable grid to avoid clutter
        plt.tight_layout()

        if bestF == True:
            plt.savefig(f'{grenoble_results}/comptons_spectra_matrix_{energy_source}keV_{rot_source}deg_bestF.png')
        else:
            plt.savefig(f'{grenoble_results}/comptons_spectra_matrix_{energy_source}keV_{rot_source}deg_md{min_dist}_maxd{max_dist}.png')


def get_ComptonEventsEnergyMatrix(all_data_folder, list_all_data_sorted, typeEvent_folder, list_typeEvent_sorted, min_dist = 0, max_dist = 10000, cuts=True):
    pixel_matrix = np.zeros((256,1024))


    e_ph_list = []
    e_elec_list = []
    theta_list = []

    for i, file in enumerate(list_all_data_sorted):
        print(file)

        df_all_data = pd.read_parquet(f'{all_data_folder}/{file}')
        df_typeEvent = pd.read_parquet(f'{typeEvent_folder}/{list_typeEvent_sorted[i]}')


        if cuts == True:
            df_compton_limit = limit_dist_photon_travel(df_typeEvent, min_dist, max_dist)
            for event, line in df_compton_limit.iterrows():
                if line['E_photon=E1'] == 'yes' and line['E_elect=E2'] == 'yes':
                    E_in = float(line['E_in'])
                    E_ph = float(line['E1'])
                    E_elec = float(line['E2'])

                    try:
                        theta = get_compton_theta_angle(E_in, E_ph)
                        e_ph_list.append(E_ph)
                        e_elec_list.append(E_elec)
                        theta_list.append(theta)
                    except Exception:
                        continue
                else:
                    E_in = float(line['E_in'])
                    E_ph = float(line['E2'])
                    E_elec = float(line['E1'])
                    theta = get_compton_theta_angle(E_in, E_ph)

                    e_ph_list.append(E_ph)
                    e_elec_list.append(E_elec)
                    theta_list.append(theta)

        else:
            for event, line in df_typeEvent.iterrows():
                if line['E_photon=E1'] == 'yes' and line['E_elect=E2'] == 'yes':
                    E_in = float(line['E_in'])
                    E_ph = float(line['E1'])
                    E_elec = float(line['E2'])
                    theta = get_compton_theta_angle(E_in, E_ph)

                    e_ph_list.append(E_ph)
                    e_elec_list.append(E_elec)
                    theta_list.append(theta)
                else:
                    E_in = float(line['E_in'])
                    E_ph = float(line['E2'])
                    E_elec = float(line['E1'])
                    theta = get_compton_theta_angle(E_in, E_ph)

                    e_ph_list.append(E_ph)
                    e_elec_list.append(E_elec)
                    theta_list.append(theta)

    e_ph = np.array(e_ph_list)
    e_elec = np.array(e_elec_list)
    theta = np.array(theta_list)

    # Define 1 keV bins
    max_ph = np.ceil(np.max(e_ph))
    max_elec = np.ceil(np.max(e_elec))
    bins_ph = np.arange(0, max_ph + 1, 1)
    bins_elec = np.arange(0, max_elec + 1, 1)

    # Create counts matrix
    counts_matrix, _, _ = np.histogram2d(e_ph, e_elec, bins=[bins_ph, bins_elec])

    # Create theta matrix (average theta per bin)
    theta_matrix, xedges, yedges = np.histogram2d(
        e_ph, e_elec, 
        bins=[bins_ph, bins_elec],
        weights=theta
    )
    theta_matrix = np.divide(
        theta_matrix,
        counts_matrix,
        out=np.zeros_like(theta_matrix),
        where=(counts_matrix != 0)  # Avoid division by zero
    )

    return counts_matrix, theta_matrix, bins_ph, bins_elec   



def get_compton_theta_angle(E0, E1):
    '''
    E0 = Incoming photon energy
    E1 = Outgoing photon energy
    return, theta outgoing photon angle
    '''
    me = 511 # electron mass in keV/c²
    c = 299792458 # speed of light in m/s
    cos_theta = 1 - (me)*(1/E1 - 1/E0)
    theta = np.arccos(cos_theta)
    degrees = np.degrees(theta)
    return degrees


def determine_slope_comptonPlot(theta, sourceEnergy):
    E_1 = random.randint(50, sourceEnergy)
    E_ph_1 = compton_photon(E_1, theta)
    E_elec_1 = energy_electron(E_1, theta)

    E_2 = random.randint(50, sourceEnergy)
    E_ph_2 = compton_photon(E_2, theta)
    E_elec_2 = energy_electron(E_2, theta)

    slope = (E_elec_2 - E_elec_1)/(E_ph_2 - E_ph_1)

    b = E_elec_1 - slope*E_ph_1

    return slope, b



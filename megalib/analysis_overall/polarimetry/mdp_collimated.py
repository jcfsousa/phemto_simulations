from math import sqrt
from pathlib import Path
import re
import argparse
from numpy.linalg.linalg import det
from scipy import integrate
from scipy.interpolate import interp1d
from unicodedata import normalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import ticker
import subprocess
import sys
from matplotlib.ticker import FuncFormatter
import  manalysis.comptons as compton

print(sys.path)
#import manalysis.specLib as specLib
#import manalysis.comptons as compton
#import polarimetry as polarimetry
#import manalysis.polarizationfits as fits

from datetime import datetime
#from calibration.calibration import Calibration

SECONDS_INDAY = 60 * 60 * 24

def plot_MDP_Energy(mdp_crab_dict, energy_bin_edges, time_lst, mcrab_lst=[1], source = 'Crab'):
    linestyles = ['-', '-', '-']
    colors = ['#0072B2', '#D55E00', '#009E73']
    markers = ['o', 's', '^']

    for delta_t in time_lst:
        if delta_t != 1e5:  # Plot only for t = 1e5 seconds
            continue
        
        for mcrab_idx, (mcrab, linestyle, color, marker) in enumerate(
                zip(mcrab_lst, linestyles, colors, markers)):
            
            x, y = [], []
            xerr_left, xerr_right = [], []
            yerr_lower, yerr_upper = [], []
            
            for idx, E in enumerate(Energy_lst):
                x.append(E)
                mdp_crab_value = mdp_crab_dict.get((E, delta_t, mcrab), float('inf'))
                y.append(mdp_crab_value * 100)  # Convert to percentage
                E_min, E_max = energy_bin_edges[idx]
                xerr_left.append(E - E_min)
                xerr_right.append(E_max - E)
                
            
            xerr = np.array([xerr_left, xerr_right])
            print(xerr)
            
            if source == 'Crab':
                plt.errorbar(x, y, xerr=xerr, capsize=6, 
                        fmt=marker, color=color, markersize=8,
                        label=f'{format_flux_label(mcrab)}', linewidth=2)
            else:
                plt.errorbar(x, y, xerr=xerr, capsize=6, 
                        fmt=marker, color=color, markersize=8,
                        linewidth=2)

            plt.plot(x, y, linewidth=2, linestyle=linestyle, alpha=0.5, color=color)
        
        plt.xlabel('Energy (keV)', fontsize=14)
        exp = int(np.log10(delta_t))
        plt.ylabel(f'MDP (%) (t = 10$^{{{exp}}}$ s)', fontsize=14)
        plt.yscale('log')
        plt.axhline(5, color='darkred', linestyle='--', linewidth=2, label='5% MDP threshold')
        plt.title(f'{source} Minimum Detectable Polarization vs Energy (t = {delta_t/SECONDS_INDAY:.1f} days)')
        plt.legend(fontsize=12, frameon=False, loc='best')

        plt.minorticks_on()
        plt.grid(which='major', linestyle='-', linewidth=0.8, alpha=0.7, color='gray')
        plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.7, color='lightgray')
        plt.tick_params(which='major', length=8, width=1.5, direction='in', top=True, right=True)
        plt.tick_params(which='minor', length=4, width=1, direction='in', top=True, right=True)
        plt.xlim(0, max(Energy_lst) + 25)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(plt.MultipleLocator(20))  # Minor ticks every 25 keV
        ax.xaxis.set_major_locator(plt.MultipleLocator(100))  # Major ticks every 100 keV
        ax.yaxis.set_minor_formatter(plt.NullFormatter())  # No labels on minor ticks
        plt.tight_layout()

def plot_MDP_Time(mdp_crab_dict, energy_bin_edges, time_lst, mcrab_lst=[1], source='Crab'):

    cmap = plt.cm.jet
    nE = len(energy_bin_edges)

    colors = cmap(np.linspace(0, 1, nE))

    for idx,E in enumerate(Energy_lst):
        color = colors[idx]
        for mcrab in mcrab_lst:
            x, y = [], []
            for delta_t in time_lst:
                x.append(delta_t)
                mdp_value = mdp_crab_dict.get((E, delta_t, mcrab), float('inf'))
                y.append(mdp_value * 100)

            E_min, E_max = energy_bin_edges[idx]
            legend_e = f'{E_min}-{E_max} keV'

            plt.plot(x, y, color=color,label=legend_e)

    plt.title(source)
    plt.xlabel('Time [s]')
    plt.ylabel('MDP [%]')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1,1e7)
    ax = plt.gca()

    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth=0.7)
    ax.grid(which='minor', linestyle=':', linewidth=0.5)

    plt.legend(title="Energy bin", ncol = 3)                                                   
    plt.tight_layout()

def integrate_flux(flux_dict, E_min, E_max, solid_angle):
    
    filtered_dict_energies = {k: v for k, v in flux_dict.items() if E_min <= float(k) <= E_max}
    
    #if len(filtered_dict_energies) < 2:
    #    raise ValueError("Not enough data points in the specified energy range.")
    
    energies = np.array(sorted(filtered_dict_energies.keys()))
    fluxes = np.array([filtered_dict_energies[e] for e in energies])

    integrated_flux = np.trapezoid(fluxes, energies) 

    return integrated_flux * solid_angle # cnts cm-2 s-1


def compute_MDP(n_source, q_100, n_background, delta_t):
   
    MDP = (4.29/(q_100*n_source)) * np.sqrt((n_source + n_background)/delta_t)

    if MDP > 1:
        MDP = 1

    return MDP


def compton_angle(E0, E1):
    me = 511 # electron mass in keV/c²
    c = 299792458 # speed of light in m/s
    cos_theta = 1 - (me)*(1/E1 - 1/E0)
    theta = np.arccos(cos_theta)
    degrees = np.degrees(theta)
    print("----------------------")
    print("E0: ", E0)
    print("E1: ", E1)
    print("degrees: ", degrees)
    print("----------------------")
    print("\n")
    return degrees
    
def N_for_given_MDP(Q, MDP):
    # No background
    # https://arxiv.org/pdf/1006.3711
    return (4.29/(Q*MDP))**2

def compton_photon(E0, theta):
    me = 511 # electron mass in keV/c²
    c = 299792458 # speed of light in m/s
    theta = np.radians(theta)
    cos_theta = np.cos(theta)
    E1 = E0/(1 + (E0/me)*(1 - cos_theta))
    return E1

def angle_electron(E0,theta):
    me = 511 # electron mass in keV/c²
    c = 299792458 # speed of light in m/s
    theta = np.radians(theta)
    tan_alpha = (1/(1+(E0/me))*(1/np.tan(theta/2)))
    alpha = np.arctan(tan_alpha)
    degrees = np.degrees(alpha)
    return degrees

def get_energy_electron(E0, theta):
    me = 511 # electron mass in keV/c²
    c = 299792458 # speed of light in m/s
    theta = np.radians(theta)
    elec_energy = E0*(((E0/me)*(1-np.cos(theta))))/(1+(E0/me)*(1-np.cos(theta)))
    return elec_energy

def compute_E_electron(E_incoming_photon, theta):
    E_electron = round(get_energy_electron(E_incoming_photon, theta),1)
    return E_electron

def compute_E_photon(E_incoming_photon,theta):
    E_photon = round(compton_photon(E_incoming_photon, theta),1)
    return E_photon

def formula_Q_theta(epsilon, theta):
    theta = np.radians(theta)
    return (np.sin(theta)**2)/((1/epsilon) + epsilon - np.sin(theta)**2)

def theoretical_QvrsEnergy(output_folder, list_energies):
    thetas = np.arange(0, 180, 1)

    dict_colors = {'100':'k', '150': 'blue', '200': 'red', '250': 'orange', '300': 'green', '500':'purple', '1000': 'brown'}
    linestyles = {'100': '-','150': '--','200': '-.','250': ':','300': (0, (3, 1, 1, 1)),'500': (0, (5, 1)),'1000': (0, (1, 1))}


    plt.figure(figsize = (8,7))
    for energy in list_energies:
        Q = []
        for theta in thetas:
            electron = compute_E_electron(energy, theta)
            photon = compute_E_photon(energy, theta)

            epsilon = photon / energy

            q_value = formula_Q_theta(epsilon, theta)

            Q.append(q_value)
        
        color = dict_colors[f'{energy}']
        linestyle = linestyles[f'{energy}']

        print(thetas[np.argmax(Q)])
        x_loc_max = thetas[np.argmax(Q)]
        plt.plot(thetas, Q, color = color, label = f'{energy} keV', linestyle=linestyle, linewidth = 2)
        plt.vlines(x_loc_max, -1, np.max(Q), color = 'k', linestyle = linestyle, alpha = 0.3)
   
    plt.ylabel(r'Modulation Factor, $Q_{{100}}$')
    plt.xlabel(r'Compton Scattering Polar Angle, $\theta$ ($^{{\circ}}$)')
    ticks = np.arange(0, 210, 30)
    plt.xlim(0,180)
    plt.ylim(0,1)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', top=True, right=True)
    plt.xticks(ticks)
    plt.legend(ncol = 1, fontsize=16)
    plt.grid(False)
    #plt.savefig(f'{output_folder}/theoretical_QvsThetaCompton.png')



def integrate_power_law(K, alpha, x_min, x_max):
    return (K / (1 - alpha)) * (x_max**(1 - alpha) - x_min**(1 - alpha))

def integrate_powerlaw_with_area(K, alpha, area_dict, E_min, E_max, solid_angle=1):
    energies = np.array(sorted(area_dict.keys()))
    mask = (energies >= E_min) & (energies <= E_max)
    energies = energies[mask]
    areas = np.array([area_dict[e] for e in energies])
    
    fluxes = K * energies**(-alpha)
    fluxes = np.array(fluxes) 
    dE = np.diff(energies, prepend=energies[0])
    dE[0] = 25


    counts_per_second_perkev = fluxes * areas



    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=100)

    ax1.plot(energies, counts_per_second_perkev,
                color='crimson', linewidth=1.5, 
                label=r'Count rate (${\rm cnts\,s^{-1}}\,keV^{-1}$)')

    ax1.set_xlabel('Energy (keV)', fontsize=20, fontweight='bold')
    ax1.set_ylabel('Count Rate (cnts s$^{-1}$ keV$^{-1}$)', fontsize=20, 
                   color='crimson', fontweight='bold')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='crimson', labelsize=17)
    ax1.grid(True, alpha=0.3, linewidth=0.8)

    ax2 = ax1.twinx()
    ax2.semilogy(energies, fluxes, '--', color='darkblue', linewidth=2, 
                 alpha=0.8, label=r'Flux (${\rm ph\,cm^{-2}\,s^{-1}\,keV^{-1}}$)')

    ax2.set_ylabel('Flux (ph cm$^{-2}$ s$^{-1}$ keV$^{-1}$)', 
                   fontsize=20, color='darkblue', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='darkblue', labelsize=17)

    ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax2.yaxis.set_major_formatter(ticker.ScalarFormatter())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
               fontsize=13, framealpha=0.95)

    plt.title('Phemto: Crab Flux (6.5 keV bins)', 
              fontsize=21, fontweight='bold', pad=20)

    plt.tight_layout()
    total_crab_rate = np.trapezoid(counts_per_second_perkev, energies)
    return total_crab_rate  # cnts/s/kev

def integrate_area_scipy(area_dict, E_min, E_max):
    """
    Integrate area using scipy's integration with interpolation.
    """
    # Extract and sort data
    energies = np.array(list(area_dict.keys()))
    areas = np.array(list(area_dict.values()))
    sort_idx = np.argsort(energies)
    energies = energies[sort_idx]
    areas = areas[sort_idx]
    
    # Create interpolation function
    interp_func = interp1d(energies, areas, kind='linear', bounds_error=False, fill_value=0)
    
    # Integrate using scipy's quad function
    total_area, error = integrate.quad(interp_func, E_min, E_max)
    
    return total_area, error


def integrate_area_from_dict(area_dict, E_min, E_max):
    """
    Integrate area values from area_dict between E_min and E_max.
    
    Parameters:
    -----------
    area_dict : dict
        Dictionary with energy (float) as keys and area values as values
    E_min : float
        Minimum energy for integration
    E_max : float
        Maximum energy for integration
        
    Returns:
    --------
    total_area : float
        Integrated area between E_min and E_max
    """
    
    # Extract energies and areas from the dictionary
    energies = np.array(list(area_dict.keys()))
    areas = np.array(list(area_dict.values()))
    
    sort_idx = np.argsort(energies)
    energies = energies[sort_idx]
    areas = areas[sort_idx]
    
    # Check if E_min and E_max are within the data range
    if E_min < energies[0]:
        print(f"Warning: E_min ({E_min}) is below minimum energy in data ({energies[0]})")
        E_min = energies[0]
    if E_max > energies[-1]:
        print(f"Warning: E_max ({E_max}) is above maximum energy in data ({energies[-1]})")
        E_max = energies[-1]
    
    # Select data within the integration range
    mask = (energies >= E_min) & (energies <= E_max)
    energies_subset = energies[mask]
    areas_subset = areas[mask]
    
    if E_min not in energies_subset or E_max not in energies_subset:
        # Create interpolation function
        interp_func = interp1d(energies, areas, kind='linear', bounds_error=False, fill_value=0)
        
        # Create finer grid for integration
        n_points = max(100, len(energies_subset) * 10)
        fine_energies = np.linspace(E_min, E_max, n_points)
        fine_areas = interp_func(fine_energies)
        
        # Integrate using trapezoidal rule
        total_area = np.trapezoid(fine_areas, fine_energies)
    else:
        # If exact points exist, use them directly
        total_area = np.trapezoid(areas_subset, energies_subset)
    
    return total_area

def sum_areas_in_range(area_dict, E_min, E_max):
    """
    Sum all area values for energies between E_min and E_max.
    
    Parameters:
    -----------
    area_dict : dict
        Dictionary with energy as keys and area values
    E_min, E_max : float
        Energy range (inclusive)
        
    Returns:
    --------
    total_area : float
        Sum of all area values in the energy range
    """
    total = 0.0
    count = 0
    
    for energy, area in area_dict.items():
        if E_min <= energy <= E_max:
            total += area
            count += 1
    
    return total

def format_flux_label(mcrab):
    if np.isclose(mcrab, 1.0):
        return '1 Crab'
    elif mcrab >= 0.1:
        return f'{int(mcrab * 1000)} mCrab'
    else:
        return f'{int(mcrab * 1000)} mCrab'

def crab_flux_low_energy(E, mcrab=1):
    """Crab spectrum for E < 50 keV: dN/dE = 11.6e4 * E^-2.18 ph/s/m2/keV"""
    K = 11.6  # ph/cm2/s/keV 
    alpha = 2.18
    return mcrab * K * E**(-alpha)

def crab_flux_high_energy(E, mcrab=1):
    """Crab spectrum for E >= 50 keV: F(E) = k (E/1keV)^(-alpha) ph/cm2/s/keV"""
    K = 14.44  # ph/cm2/s/keV
    alpha = 2.169
    return mcrab * K * (E/1.0)**(-alpha)

def integrated_crab_flux(E_min, E_max, mcrab=1, n_points=500):
    """Integrate crab flux over energy range using appropriate spectrum"""
    energies = np.linspace(E_min, E_max, n_points)
    fluxes = []
    
    for E in energies:
        if E < 50:
            fluxes.append(crab_flux_low_energy(E, mcrab))
        else:
            fluxes.append(crab_flux_high_energy(E, mcrab))
    
    integral = np.trapezoid(fluxes, energies)
    return integral  # ph/cm2/s

def read_xcm_model(xcm_path):
    xcm_path = Path(xcm_path)
    lines = xcm_path.read_text().splitlines()

    # Find the 'model' line
    model_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("model"):
            model_line = line.strip()
            model_index = i
            break
    if model_line is None:
        raise ValueError("No 'model' line found in xcm file")

    # Example: "model  TBabs*cutoffpl"
    model_expr = model_line.split(None, 1)[1]  # ignore 'model'
    model_expr = model_expr.strip()

    # For TBabs*cutoffpl in your file, there are 4 parameter lines after 'model'
    # (this is specific to that model; for general use you need a mapping)
    param_lines = lines[model_index+1 : model_index+5]
    params = []
    for pl in param_lines:
        if pl.strip():
            # first token is the value
            val = float(pl.split()[0])
            params.append(val)

    return model_expr, params

def cutoffpl(E, Gamma, Ecut, K):
    # E in keV, Gamma is photon index, Ecut in keV, K normalization
    return K * (E**(-Gamma)) * np.exp(-E / Ecut)

def tbabs_approx(E, NH):
    # Extremely crude approximation: sigma(E) ∝ E^{-3}
    # NH in 10^22 cm^-2 (as in TBabs), E in keV
    sigma0 = 3e-22  # adjust as needed
    sigma = sigma0 * (E**-3)
    return np.exp(-NH * 1e22 * sigma)

def xspec_TBabs_cutoffpl(E, NH, Gamma, Ecut, K):
    return tbabs_approx(E, NH) * cutoffpl(E, Gamma, Ecut, K)

def integrated_model_flux(E_min, E_max, model_func, params, n_points=200):

    energies = np.linspace(E_min, E_max, n_points)

    flux = model_func(energies, *params)

    integral = np.trapezoid(flux, energies)

    return integral

def compute_background(cosmic_background_file, angle_min_deg, angle_max_deg, E_min, E_max):
    '''
    Returns ph/cm2/s
    '''

    background_dict = {}
    with open(cosmic_background_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or '#' in stripped or 'IP' in stripped or 'EN' in stripped:
                continue

            parts = stripped.split()
            if len(parts) >= 3 and parts[0] == 'DP':
                energy = float(parts[1])  
                flux = float(parts[2])
                background_dict[energy] = flux


    # Solid angle in deg (apperture of cone) to integrate background
    angle_min_deg = 0 
    angle_max_deg = 90
    def deg_to_rad(deg):
        return deg * (np.pi/180)
    angle_max_rad = deg_to_rad(angle_max_deg) # in rad
    angle_min_rad = deg_to_rad(angle_min_deg)
    angle_rad = abs(angle_min_rad - angle_max_rad)
    solid_angle = 2*np.pi * (1 - np.cos(angle_rad))

    #print(f'Solid Angle: {solid_angle} rad')
    total_flux_cosmic_background = integrate_flux(background_dict, E_min, E_max, solid_angle) #ph/cm2/s
    return total_flux_cosmic_background

if __name__ == '__main__':

    xcm_path = "./mdp/cosmic_sources/1A0535.xcm"
    #xcm_path = "./sources/mo_base.xcm"
    model_expr, params = read_xcm_model(xcm_path)
    NH, Gamma, Ecut, K = params

    E = np.linspace(1, 100, 1000)  # keV
    flux_1A0535 = xspec_TBabs_cutoffpl(E, NH, Gamma, Ecut, K)

    plt.figure()
    plt.title("Flux 1A0535 ")
    plt.plot(E, flux_1A0535, c='k')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel("Energy [keV]")
    plt.ylabel(r"Flux [ph cm$^2$ s$^{-1}$ kev$^{-1}$]")


    #Energy_lst = [100, 200, 300, 400]

    Energy_lst = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100, 125, 150]
    #Energy_lst = [15, 30, 40, 50, 60, 80,100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    #Energy_lst = [20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    #Energy_lst = [15, 35, 55, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    #Energy_lst = [15, 100, 300, 600]

    #All energy
    Energy_lst = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    Energy_lst = [20, 60, 150]
    # SIMULATION constants
    source_type = "CollimatedBeam"
    spectra_binnig = 'all'
    n_event_sourceCollimated = 1e6
    distance_dets = 1
    HED_config = 3
    min_dist = 0.075
    angle_bin = 15
    max_dist = 100000
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    base_polarimetry_folder = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry_results'
    
    lst_Q = []
    lst_abs_eff = []
    lst_Q_uncert = []
    detector = 'Both'
    
    # Binning the spectra properly
    energy_bin_edges = []
    energy_bin_widths = []
    lst_energy_centers = []  # Store energy bin centers for plotting
    lst_energy_widths = []   # Store energy bin widths
    for i, E in enumerate(Energy_lst):
        if i == 0:
            # First bin: from E - half distance to next point
            half_dist = (Energy_lst[i+1] - E) / 2
            low_edge = E
            high_edge = E + half_dist
        elif i == len(Energy_lst) - 1:
            # Last bin: from previous point to E + half distance
            half_dist = (E - Energy_lst[i-1]) / 2
            low_edge = E - half_dist
            high_edge = E
        else:
            # Middle bins: edges halfway between energy points
            low_edge = E - (E - Energy_lst[i-1]) / 2
            high_edge = E + (Energy_lst[i+1] - E) / 2
        
        energy_bin_edges.append((low_edge, high_edge))
        energy_bin_widths.append(high_edge - low_edge)
        lst_energy_centers.append(E)
        lst_energy_widths.append(high_edge - low_edge)


    plt.figure(figsize=(10, 6))
    
    # Plot both spectra and the combined one, Crab
    E_debug = np.linspace(10, 600, 1000)
    flux_low = [crab_flux_low_energy(E, 1) for E in E_debug]
    flux_high = [crab_flux_high_energy(E, 1) for E in E_debug]
    flux_combined = []
    for E in E_debug:
        if E < 50:
            flux_combined.append(crab_flux_low_energy(E, 1))
        else:
            flux_combined.append(crab_flux_high_energy(E, 1))
    
    #plt.plot(E_debug, flux_low, 'b--', label='Low Energy (E<50 keV)', alpha=0.7)
    #plt.plot(E_debug, flux_high, 'r--', label='High Energy (E>=50 keV)', alpha=0.7)
    plt.plot(E_debug, flux_combined, 'k-', label='Combined Spectrum', linewidth=2)
    plt.axvline(x=50, color='gray', linestyle=':', label='Transition at 50 keV')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('Energy (keV)')
    plt.ylabel('Flux (ph/cm²/s/keV)')
    plt.title('Crab Spectrum - Combined Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.savefig('crab_spectrum_debug.png', dpi=150)
    plt.show()
    


    # Get the Q100, Compt_eff for collimated source in Energy_lst
    for source_energy in Energy_lst:
        result_polarimetry= f"{base_polarimetry_folder}/{source_type}{source_energy}keV_config{HED_config}x{HED_config}_{distance_dets}cm" # When using collimated beam i am using ALL double comptons, no spectral bin
        result_polarimetry = f'{result_polarimetry}/all'
        folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
        Q, Q_uncer = compton.get_Q(folder_result_polarimetry, detector) # For all sources gonna reject first pixel order compton events and use 36deg bin
        n_comptons = compton.get_n_comptons(folder_result_polarimetry, detector)
        abs_eff = n_comptons/n_event_sourceCollimated
        lst_abs_eff.append(abs_eff)
        lst_Q.append(Q)
        lst_Q_uncert.append(Q_uncer)

    
    plt.figure()
    plt.plot(Energy_lst, lst_abs_eff)
    plt.scatter(Energy_lst, lst_abs_eff)
    plt.xlabel('Energy')
    plt.ylabel('Abs Compton Eff')


    plt.figure()
    plt.errorbar(Energy_lst, lst_Q, yerr=lst_Q_uncert)
    plt.xlabel('Energy')
    plt.ylabel('Q [0-1]')



    #cdte_res = []
    #with open('cdte_res.txt', 'r') as f:
    #    lines = f.readlines()
    #    
    #    for line in lines:
    #        if 'energy' in line:
    #            continue
    #        parse = line.split(',')
    #        cdte_res.append(float(parse[-1]))
    
    
    # Eff Area lenses + mirror
    file_area = "/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/sensitivity/area_mirror_lenses.csv"
    
    area_dict = {}
    with open(file_area, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Energy' in line:
                continue
            parts = line.split(',')
            energy = float(parts[0])  
            area = float(parts[1])

            area_dict[energy] = area
    
        
    energies = np.array(sorted(area_dict.keys()))
    areas = np.array([area_dict[e] for e in energies])

    plt.figure()
    plt.plot(energies, areas, label='Optics Eff Area (cm2)')
    plt.scatter(energies, areas)
    plt.ylabel('Area cm2')
    plt.xlabel('Energy')

    

    mdp_crab_lst = []
    #mcrab_lst = [1,0.1, 0.01, 0.001]
    mcrab_lst = [1, 0.1, 0.01]

    mdp_crab_dict = {} #key:day value:mdp
    n_days = 3
    time_lst = np.arange(1000, 1e6+1000,1000)
    seconds_inDay = 60 * 60 * 24
    

    mdp_crab_time_dict = {}
    mdp_crab_dict = {}
    cnts_crab_dict = {}
    cnts_background_dict = {}

    # to write megalib sources
    with open("crab_optics_spectrum.dat", "w") as f:

        f.write("IP LINLIN\n")
        for i,E in enumerate(Energy_lst):
            E_min, E_max = energy_bin_edges[i]
            total_area = integrate_area_from_dict(area_dict, E_min, E_max)/(E_max-E_min)
            integrated_flux_crab = integrated_crab_flux(E_min, E_max)
            photons_per_second = integrated_flux_crab * total_area
            dNdE = photons_per_second/(E_max-E_min)
            f.write(f"DP {E} {dNdE}\n")
        f.write("EN")
    
    # to write total flux for megalib
    total_flux = 0
    for i,E in enumerate(Energy_lst):
        E_min,E_max = energy_bin_edges[i]
        total_area = integrate_area_from_dict(area_dict,E_min,E_max)/(E_max-E_min)
        integrated_flux_crab = integrated_crab_flux(E_min,E_max)
        photons_per_second = integrated_flux_crab * total_area
        total_flux += photons_per_second
    print(f"Total flux for MEGAlib: {total_flux}")


    V0332spec_file = "./mdp/cosmic_sources/V0332_flux.dat"

    V0332spec_bins = []
    V0332spec_flux = []

    with open(V0332spec_file, 'r') as f:
        for line in f:
            if 'E_min' in line:
                continue
            parts = line.split()
            E_min = float(parts[0])
            E_max = float(parts[1])
            flux = float(parts[2])  # already integrated over the bin

            V0332spec_bins.append((E_min, E_max))
            V0332spec_flux.append(flux)
    
    print(f"V0332spec_bins {V0332spec_bins}")
    # Crab MDP Analysis
    for idx, mcrab in enumerate(mcrab_lst):
        for delta_t in time_lst:  # Skip n=0 directly in range
            total_crab_cnts = 0
            total_background_cnts = 0
            mdp_crab_lst = []
            lst_total_area = []

            MEGAlib_crabSource_E = []
            MEGAlib_crabSource_flux = []
            MEGALIB_background_flux = []
            for i,E in enumerate(Energy_lst):
                E_min, E_max = energy_bin_edges[i]
                # Get effective area for this energy bin
                total_area = integrate_area_from_dict(area_dict, E_min, E_max) / (E_max - E_min)
                
                # Integrated crab flux over the energy bin
                #integrated_flux_crab = integrated_crab_flux(E_min, E_max)
                #integrated_flux_1A0535= integrated_model_flux(E_min, E_max, xspec_TBabs_cutoffpl, params)
                integrated_flux_crab = 0.0
                for (Emin_spec, Emax_spec), flux in zip(V0332spec_bins, V0332spec_flux):
                    # Compute overlap between your bin and the spectrum bin
                    overlap_min = max(E_min, Emin_spec)
                    overlap_max = min(E_max, Emax_spec)
                    print(f"E_min {E_min}")
                    print(f"E_max {E_max}")
                    print(f"Emin_spec {Emin_spec}")
                    print(f"Emax_spec {Emax_spec}")
                    print(f"overlap_min {overlap_min}")
                    print(f"overlap_max {overlap_max}")

                    if overlap_max > overlap_min:
                        spec_width = Emax_spec - Emin_spec
                        
                        # Fraction of the spectrum bin contributing
                        weight = (overlap_max - overlap_min) / spec_width
                        
                        integrated_flux_crab += flux * weight
                # Counts from source
                total_crab_cnts_s = integrated_flux_crab * total_area  # ph/s
                #crab_cnts = crab_cnts_s * delta_t
                print(f'Total CRAB integrated flux from {E_min} keV to {E_max} keV: {integrated_flux_crab:.4e} ph/cm2/s')
                print(f'Total CRAB integrated flux from {E_min} keV to {E_max} keV: {total_crab_cnts_s:.4e} ph/s')
                
                print(f'Total Sensitive Area: {total_area} cm^2')

                dNdE = total_crab_cnts_s/(E_max-E_min) #ph/s/kev for Megalib

                MEGAlib_crabSource_E.append(E)
                MEGAlib_crabSource_flux.append(dNdE)

                #total_crab_cnts += crab_cnts * lst_abs_eff[i]  # Apply Compton efficiency
                # total_background_cnts += background_cnts


                #A_eff = np.interp(E, energies, areas)
                # Power law parameters
                #K = 14.44  # ph/cm²/s/keV
                #alpha = 2.169
                #
                #half_bin = 50 # keV
                #E_min = E - half_bin
                #if E == 400:
                #    E_max = E + half_bin
                #else:
                #    E_max = E + half_bin

                ## Integrate power law flux
                ## arXiv:astro-ph/0406058v1 2 Jun 2004
                ## F(E) = k (E/1kev)^(-alpha) ph/cm2/s/keV
                ## We will Integrate flux over 50 kev Bin
                #integrated_flux_crab = integrate_power_law(K, alpha, E_min, E_max) #ph/cm2/s

                # Integrate the area then divide by bin size
                #total_area = integrate_area_from_dict(area_dict, E_min, E_max) / (half_bin*2)
                #total_area = sum_areas_in_range(area_dict, E_min, E_max)
                
                #total_crab_cnts_s = integrated_flux_crab * total_area #ph/s


                ## for continuos source need to compute the flux and eff ares per bins of energy 
                ## (to take into consideration the eff area with energy) then in the end sum ALL ->cnts/s 
                ##then multiply by compton eff of that energy band and use Q of that energy band


                
                ## BACKGROUND 
                #background_file = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry/mdp/CosmicPhotonSpectrum.dat'
                cosmic_background_file = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/background/Data/CosmicPhotons_Spec_600.0km_5.0deg_1100.0solarmod.dat'
                total_flux_cosmic_background = compute_background(cosmic_background_file, 0, 90, E_min, E_max)
                area_background = np.pi * 0.075 **2 # Only take into account the background on the laue lenses focus, 1.5mm diameter
                total_background_cnts_s = total_flux_cosmic_background * area_background
                print(f'Total BACKGROUND integrated flux from {E_min} keV to {E_max} keV: {total_background_cnts_s:.20e} ph/s')
                MEGALIB_background_flux.append(total_background_cnts_s) 
                ## APPLY COMPTON EFFICIENCY
                compton_background_cnts_s = total_background_cnts_s * lst_abs_eff[i] # comptons cnts/s
                compton_crab_cnts_s = mcrab*total_crab_cnts_s * lst_abs_eff[i] # comptons cnts/s
                
                print(f'Total CRAB COMPTONS integrated flux from {E_min} keV to {E_max} keV: {compton_crab_cnts_s:.20e} ph/s')
                print(f'Total BACKGROUND COMPTONS integrated flux from {E_min} keV to {E_max} keV: {compton_background_cnts_s:.20e} ph/s')
                print('##################################################')



                mdp_crab = compute_MDP(compton_crab_cnts_s, lst_Q[i], compton_background_cnts_s, delta_t)

                mdp_crab_dict[E, delta_t, mcrab] = mdp_crab
                cnts_crab_dict[E, delta_t, mcrab] = compton_crab_cnts_s * delta_t
                cnts_background_dict[E, delta_t, mcrab] = compton_background_cnts_s * delta_t
                lst_total_area.append(total_area)
                #mdp_lst.append(compute_MDP(compton_crab_cnts_s, lst_Q[i], compton_background_cnts_s, delta_t))
            if mcrab == 1 and delta_t == 1000:
                plt.plot(MEGAlib_crabSource_E, MEGAlib_crabSource_flux, label='Crab flux (ph/s)')
                plt.scatter(MEGAlib_crabSource_E, MEGAlib_crabSource_flux)
                plt.plot(MEGAlib_crabSource_E, MEGALIB_background_flux, label='Background flux (ph/s)')
                plt.yscale('log')
                plt.xlabel("Energy [keV]")
                plt.ylabel("Crab Flux at instrument [ph/s]")
                
                print(MEGAlib_crabSource_E)
                print(MEGAlib_crabSource_flux)
                plt.legend()
                plt.show()
   
    mdp_1A0535_lst = []
    mdp_1A0535_dict = {} #key:day value:mdp
    cnts_1A0535_dict = {}
    cnts_background_1A0535_dict = {}
    
    # Crab MDP Analysis
    for delta_t in time_lst:  # Skip n=0 directly in range
        total_1A0535_cnts = 0
        total_background_cnts = 0
        mdp_1A0535_lst = []
        lst_total_area = []
        for i,E in enumerate(Energy_lst):
            E_min, E_max = energy_bin_edges[i]
            # Get effective area for this energy bin
            total_area = integrate_area_from_dict(area_dict, E_min, E_max) / (E_max - E_min)
            
            # Integrated 1A0535 flux over the energy bin
            integrated_flux_1A0535= integrated_model_flux(E_min, E_max, xspec_TBabs_cutoffpl, params)

            # Counts from source
            total_1A0535_cnts_s = integrated_flux_1A0535 * total_area  # ph/s
            #crab_cnts = crab_cnts_s * delta_t
            print(f'Total 1A0535 integrated flux from {E_min} keV to {E_max} keV: {integrated_flux_1A0535:.4e} ph/cm2/s')
            print(f'Total 1A0535 integrated flux from {E_min} keV to {E_max} keV: {total_1A0535_cnts_s:.4e} ph/s')
            
            print(f'Total Sensitive Area: {total_area} cm^2')
            #total_crab_cnts += crab_cnts * lst_abs_eff[i]  # Apply Compton efficiency
            # total_background_cnts += background_cnts


            #A_eff = np.interp(E, energies, areas)
            # Power law parameters
            #K = 14.44  # ph/cm²/s/keV
            #alpha = 2.169
            #
            #half_bin = 50 # keV
            #E_min = E - half_bin
            #if E == 400:
            #    E_max = E + half_bin
            #else:
            #    E_max = E + half_bin

            ## Integrate power law flux
            ## arXiv:astro-ph/0406058v1 2 Jun 2004
            ## F(E) = k (E/1kev)^(-alpha) ph/cm2/s/keV
            ## We will Integrate flux over 50 kev Bin
            #integrated_flux_crab = integrate_power_law(K, alpha, E_min, E_max) #ph/cm2/s

            # Integrate the area then divide by bin size
            #total_area = integrate_area_from_dict(area_dict, E_min, E_max) / (half_bin*2)
            #total_area = sum_areas_in_range(area_dict, E_min, E_max)
            
            #total_crab_cnts_s = integrated_flux_crab * total_area #ph/s


            ## for continuos source need to compute the flux and eff ares per bins of energy 
            ## (to take into consideration the eff area with energy) then in the end sum ALL ->cnts/s 
            ##then multiply by compton eff of that energy band and use Q of that energy band


            
            ## BACKGROUND 
            #background_file = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry/mdp/CosmicPhotonSpectrum.dat'
            cosmic_background_file = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/background/Data/CosmicPhotons_Spec_600.0km_5.0deg_1100.0solarmod.dat'
            total_flux_cosmic_background = compute_background(cosmic_background_file, 0, 90, E_min, E_max)
            area_background = np.pi * 0.075 **2 # Only take into account the background on the laue lenses focus, 1.5mm diameter
            total_background_cnts_s = total_flux_cosmic_background * area_background
            print(f'Total BACKGROUND integrated flux from {E_min} keV to {E_max} keV: {total_background_cnts_s:.20e} ph/s')
            
            ## APPLY COMPTON EFFICIENCY
            compton_background_cnts_s = total_background_cnts_s * lst_abs_eff[i] # comptons cnts/s
            compton_1A0535_cnts_s = total_1A0535_cnts_s * lst_abs_eff[i] # comptons cnts/s
            
            print(f'Total 1A0535 COMPTONS integrated flux from {E_min} keV to {E_max} keV: {compton_1A0535_cnts_s:.20e} ph/s')
            print(f'Total BACKGROUND COMPTONS integrated flux from {E_min} keV to {E_max} keV: {compton_background_cnts_s:.20e} ph/s')
            print('##################################################')



            mdp_1A0535 = compute_MDP(compton_1A0535_cnts_s, lst_Q[i], compton_background_cnts_s, delta_t)

            mdp_1A0535_dict[E, delta_t, 1] = mdp_1A0535
            cnts_1A0535_dict[E, delta_t, 1] = compton_1A0535_cnts_s * delta_t
            cnts_background_dict[E, delta_t, 1] = compton_background_cnts_s * delta_t
            lst_total_area.append(total_area)
            #mdp_lst.append(compute_MDP(compton_crab_cnts_s, lst_Q[i], compton_background_cnts_s, delta_t))
    # Improve Commented plots:
    #plt.figure()
    #plt.scatter(Energy_lst, lst_total_area)
    #plt.plot(Energy_lst, lst_total_area)
    ##plt.ylabel(f'Total Area, {half_bin*2}keV bin')
    #plt.xlabel('Energy')

    ## Plot 1: Counts per energy bin for fixed time
    #plt.figure(figsize=(12, 8))
    #for delta_t in time_lst:
    #    if delta_t != 1e5:  # Plot only for t = 1e5 seconds
    #        continue
    #    
    #    for mcrab in mcrab_lst:
    #        x, y, y2 = [], [], []
    #        xerr_left, xerr_right = [], []
    #        
    #        for idx, E in enumerate(Energy_lst):
    #            x.append(E)
    #            y.append(cnts_crab_dict[(E, delta_t, mcrab)])
    #            y2.append(cnts_background_dict[(E, delta_t, mcrab)])
    #            
    #            # Calculate asymmetric errors for energy bins
    #            #E_min, E_max = energy_bin_edges[idx]
    #            E_min, E_max = energy_bin_edges[idx]

    #            xerr_left.append(E - E_min)
    #            xerr_right.append(E_max - E)
    #        
    #        xerr = np.array([xerr_left, xerr_right])
    #        
    #        plt.errorbar(x, y, xerr=xerr, yerr=None, capsize=3, 
    #                    label=f'Source: {format_flux_label(mcrab)}', 
    #                    marker='o', linestyle='-', linewidth=2)
    #        plt.errorbar(x, y2, xerr=xerr, yerr=None, capsize=3, 
    #                    label=f'Background: {format_flux_label(mcrab)}', 
    #                    marker='s', linestyle='--', linewidth=1.5, alpha=0.7)
    #    
    #    # Plot required counts for MDP = 0.05
    #    np_lst_Q = np.array(lst_Q)
    #    required_counts = [N_for_given_MDP(np_lst_Q[i], 0.05) for i in range(len(Energy_lst))]
    #    plt.plot(Energy_lst, required_counts, 'k-', linewidth=2, 
    #             label='Counts Required for MDP=0.05')
    #    
    #    plt.yscale('log')
    #    exp = int(np.log10(delta_t))
    #    plt.ylabel(f'Compton Counts (#) (t = 10$^{{{exp}}}$ s)', fontsize=13)
    #    plt.xlabel('Energy (keV)', fontsize=13)
    #    plt.title(f'Counts per Energy Bin (t = {delta_t/seconds_inDay:.1f} days)')
    #    plt.legend(loc='best', fontsize=10)
    #    plt.grid(True, alpha=0.3, which='both')
    #    plt.tight_layout()

# Pl#ot 2: Count rate per energy bin
    #plt.figure(figsize=(12, 8))
    #for delta_t in time_lst:
    #    if delta_t != 1e5:  # Plot only for t = 1e5 seconds
    #        continue
    #    
    #    for mcrab in mcrab_lst:
    #        x, y, y2 = [], [], []
    #        xerr_left, xerr_right = [], []
    #        
    #        for idx, E in enumerate(Energy_lst):
    #            x.append(E)
    #            y.append(cnts_crab_dict[(E, delta_t, mcrab)] / delta_t)
    #            y2.append(cnts_background_dict[(E, delta_t, mcrab)] / delta_t)
    #            E_min, E_max = energy_bin_edges[idx]
    #            #E_min, E_max = energy_bin_edges[idx]
    #            xerr_left.append(E - E_min)
    #            xerr_right.append(E_max - E)
    #        
    #        xerr = np.array([xerr_left, xerr_right])
    #        
    #        plt.errorbar(x, y, xerr=xerr, yerr=None, capsize=3, 
    #                    label=f'Source: {format_flux_label(mcrab)}', 
    #                    marker='o', linestyle='-', linewidth=2)
    #        plt.errorbar(x, y2, xerr=xerr, yerr=None, capsize=3, 
    #                    label=f'Background: {format_flux_label(mcrab)}', 
    #                    marker='s', linestyle='--', linewidth=1.5, alpha=0.7)
    #    
    #    # Plot required count rate for MDP = 0.05
    #    np_lst_Q = np.array(lst_Q)
    #    required_rate = [N_for_given_MDP(np_lst_Q[i], 0.05) / delta_t for i in range(len(Energy_lst))]
    #    plt.plot(Energy_lst, required_rate, 'k-', linewidth=2, 
    #             label='Count Rate Required for MDP=0.05')
    #    
    #    plt.yscale('log')
    #    exp = int(np.log10(delta_t))
    #    plt.ylabel(f'Compton Count Rate (counts/s) (t = 10$^{{{exp}}}$ s)', fontsize=13)
    #    plt.xlabel('Energy (keV)', fontsize=13)
    #    plt.xlim(0, 600)
    #    plt.title(f'Count Rate per Energy Bin (t = {delta_t/seconds_inDay:.1f} days)')
    #    plt.legend(loc='best', fontsize=10)
    #    plt.grid(True, alpha=0.3, which='both')
    #    plt.tight_layout()
    


# Plot 3: MDP vs Energy
    plt.figure(figsize=(12, 8))
    
    # Crab
    #plot_MDP_Energy(mdp_crab_dict, energy_bin_edges, time_lst, mcrab_lst) 
    #plt.figure(figsize=(12, 8))
    #plot_MDP_Time(mdp_crab_dict, energy_bin_edges, time_lst, mcrab_lst, source='Crab')

    # V0332
    plot_MDP_Energy(mdp_crab_dict, energy_bin_edges, time_lst, [1], source='V0332') 
    plt.figure(figsize=(12, 8))
    plot_MDP_Time(mdp_crab_dict, energy_bin_edges, time_lst, [1], source='V0332')
    

    plt.figure(figsize=(12, 8))
    # 1A0535
    plot_MDP_Energy(mdp_1A0535_dict, energy_bin_edges, time_lst, [1], source='1A0535') 
    plt.figure(figsize=(12, 8))
    plot_MDP_Time(mdp_1A0535_dict, energy_bin_edges, time_lst, [1], source='1A0535')
    plt.show()

        #plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.4)
        #plt.xlim(0, 620)
        #plt.tight_layout()
        
        # Save the figure
        #plt.savefig("../../../results/megalib_v2/MDP_CrabFlux_NewBinning.png", dpi=600)

## Additional plot: Energy bin widths visualization
#    plt.figure(figsize=(12, 6))
#    bin_centers = Energy_lst
#    bin_widths = energy_bin_widths
#    bin_left = energy_bin_edges[:-1]
#    bin_right = energy_bin_edges[1:]
#
#    plt.bar(bin_centers, bin_widths, width=bin_widths, align='center', 
#            alpha=0.7, color='skyblue', edgecolor='navy', linewidth=1)
#    plt.xlabel('Energy (keV)', fontsize=12)
#    plt.ylabel('Bin Width (keV)', fontsize=12)
#    plt.title('Energy Bin Widths Used in Analysis')
#    plt.grid(True, alpha=0.3, axis='y')
#
## Add bin edge labels for some points
#    for i, (left, right, center) in enumerate(zip(bin_left, bin_right, bin_centers)):
#        if i % 3 == 0:  # Label every 3rd bin to avoid crowding
#            plt.text(center, bin_widths[i] + 2, f'{left:.0f}-{right:.0f}', 
#                    ha='center', va='bottom', fontsize=8, rotation=45)
#
#    plt.tight_layout()
#    plt.savefig("../../../results/megalib_v2/Energy_Bin_Widths.png", dpi=600)
#    plt.show()

# Print summary statistics
    print("\n" + "="*60)
    print("ENERGY BINNING SUMMARY CRAB")
    print("="*60)
    for idx, E in enumerate(Energy_lst):
        E_min, E_max = energy_bin_edges[idx]
        bin_width = energy_bin_widths[idx]
        print(f"E = {E:3.0f} keV: Bin [{E_min:5.1f}-{E_max:5.1f}] keV, "
              f"Width = {bin_width:5.1f} keV, "
              f"Q = {lst_Q[idx]:.3f} ± {lst_Q_uncert[idx]:.3f}, "
              f"Eff = {lst_abs_eff[idx]:.4f}")
    print("="*60)
    plt.show()
    sys.exit(1)
    breakpoint()
    plt.figure()
    for delta_t in time_lst:
        if delta_t != 1e5:
            continue
        for mcrab in mcrab_lst:
            x, y, y2 = [], [], []
            for i, E in enumerate(Energy_lst):
                x.append(E)
                y.append(cnts_crab_dict[E, delta_t, mcrab] )
                y2.append(cnts_background_dict[E, delta_t, mcrab])
            plt.errorbar(x, y, xerr=half_bin, capsize = 3, label=f'Source: {format_flux_label(mcrab)}')
            plt.errorbar(x, y2, xerr=half_bin, capsize = 3, label=f'Background')
            plt.plot(x,y, linewidth = 2, alpha=0.3)

        np_lst_Q = np.array(lst_Q)
        plt.plot(Energy_lst, N_for_given_MDP(np_lst_Q, 0.05), label = 'Cnts Required for 0.05 MDP', c='k')
        plt.yscale('log')
        exp = int(np.log10(delta_t))
        coeff = delta_t / (10**exp)
        plt.ylabel(f'Compton Cnts (#) (t = 10$^{{{exp}}}$ s)', fontsize=13)
        plt.xlabel('Energy [keV]', fontsize=13)
        plt.legend()

    plt.figure()
    for delta_t in time_lst:
        if delta_t != 1e5:
            continue
        for mcrab in mcrab_lst:
            x, y, y2 = [], [], []
            xerr_left, xerr_right = [], []
            for i, E in enumerate(Energy_lst):
                x.append(E)
                y.append(cnts_crab_dict[E, delta_t, mcrab]/delta_t )
                y2.append(cnts_background_dict[E, delta_t, mcrab]/delta_t)
                if E == 400:
                    xerr_left.append(half_bin)   # Left error
                    xerr_right.append(0)         # No right error
                else:
                    xerr_left.append(half_bin)
                    xerr_right.append(half_bin)
            xerr = np.array([xerr_left, xerr_right])
            plt.errorbar(x, y, xerr=xerr, capsize = 3, label=f'Source: {format_flux_label(mcrab)}')
            plt.errorbar(x, y2, xerr=half_bin, capsize = 3, label=f'Background')
            plt.plot(x,y, linewidth = 2, alpha=0.3)

        np_lst_Q = np.array(lst_Q)
        plt.plot(Energy_lst, N_for_given_MDP(np_lst_Q, 0.05)/delta_t, label = 'Cnts Required for 0.05 MDP', c='k')
        plt.yscale('log')
        exp = int(np.log10(delta_t))
        coeff = delta_t / (10**exp)
        plt.ylabel(f'Compton Cnts/s (#/s) (t = 10$^{{{exp}}}$ s)', fontsize=13)
        plt.xlabel('Energy [keV]', fontsize=13)
        plt.xlim(0, 600)
        plt.legend()

    x = []
    y = []
    plt.figure(figsize=(8, 6))

    linestyles = ['-', '-', '-']
    colors = ['#0072B2', '#D55E00', '#009E73']
    markers = ['o', 's', '^']  # circle, square, triangle
    for delta_t in time_lst:
        if delta_t != 1e5:
            continue

        for mcrab, linestyle, color, marker in zip(mcrab_lst, linestyles, colors, markers):
            x, y = [], []
            for i, E in enumerate(Energy_lst):
                x.append(E)
                y.append(mdp_crab_dict[E, delta_t, mcrab] * 100)  # %
            plt.errorbar(x, y, xerr=xerr, capsize = 6, fmt=marker, color=color, label=format_flux_label(mcrab))
            plt.plot(x,y, linewidth = 2, linestyle=linestyle, alpha=0.3, c=color)

        plt.xlabel('Energy [keV]', fontsize=13)
        exp = int(np.log10(delta_t))
        coeff = delta_t / (10**exp)
        plt.ylabel(f'MDP [%] (t = 10$^{{{exp}}}$ s)', fontsize=13)
        plt.yscale('log')
        plt.axhline(5, color='darkred', linestyle='--', linewidth=2)
#plt.title('Minimum Detectable Polarization', fontsize=14)
        plt.legend(fontsize=11 ,frameon=False, loc='best')

        plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.4)

        plt.tight_layout()
        #plt.savefig("../../../results/megalib_v2/MDP_CrabFlux_100keVbin.png", dpi=600)
        plt.show()

    sys.exit()    

    plt.figure(figsize = (8,7))
    for delta_t in time_lst:  # Skip n=0 directly in range
        if delta_t != 1e5:
            continue
        for idx, mcrab in enumerate(mcrab_lst):
            mdp_lst = []
            for i,E in enumerate(Energy_lst):
                x.append(E)
                y.append(mdp_dict[E,delta_t,mcrab]*100)
            plt.plot(x,y, label=f'flux ={mcrab}Crab')
            x = []
            y=[]
        plt.title(f'Time {delta_t}') 
        plt.xlabel('Energy')
        plt.ylabel(r'MDP (\%), 100 ks')
        plt.yscale('log')
        plt.legend()
        plt.show()

    plt.figure(figsize = (8,7))
    for idx, mcrab in enumerate(mcrab_lst):
        for i,E in enumerate(Energy_lst):
            for delta_t in time_lst:  # Skip n=0 directly in range
                x.append(delta_t)
                y.append(mdp_dict[E,delta_t,mcrab])
            plt.plot(x,y, label=f'energy ={E}Crab')
            x = []
            y=[]
        plt.title(f'Flux {mcrab}') 
        plt.legend()
        plt.show()


            #plt.plot(Energy_lst, mdp, c='k', alpha = 0.3)
            #plt.errorbar(Energy_lst, mdp, xerr = 25, capsize=5, fmt='o', markersize=5, label = f'{mcrab}Crab, t={delta_t}')
            #plt.ylabel('MDP, 1e6 s')
            #plt.xlabel('Energy')
            #plt.yscale('log')
            #plt.legend()
            #mdp = []

            # Plot with consistent styling
            #color = colors[idx]
            #marker = markers[idx % len(markers)]
            #
            ## Plot line with transparency
            #ax1.plot(Energy_lst, mdp_lst, color=color, alpha=0.3, linewidth=1.5)
            #
            ## Error bars with markers
            #ax1.errorbar(Energy_lst, mdp_lst, xerr=25, capsize=4, fmt=marker, 
            #            markersize=7, color=color, alpha=0.8, 
            #            label=f'{mcrab}Crab, t={n}d', linewidth=1)
    # Set primary axis properties
    #ax1.set_ylabel('Minimum Detectable Polarization (MDP)', fontsize=12, fontweight='bold')
    #ax1.set_xlabel('Energy (keV)', fontsize=12, fontweight='bold')
    #ax1.set_yscale('log')
    #ax1.grid(True, which='both', linestyle='--', alpha=0.3)

# Cr#eate secondary x-axis for observation time
    #def energy_to_time(energy):
    #    # This is a placeholder - you'll need to map energy to time
    #    # Based on your loop structure, each energy point corresponds to delta_t
    #    # You might need to adjust this based on your actual data structure
    #    return delta_t / 3600  # Convert to hours for example

# If# you want time on top axis representing observation duration
    #ax2 = ax1.twiny()
    #ax2.set_xlabel('Observation Time (hours)', fontsize=12, fontweight='bold')
# Se#t appropriate ticks based on your delta_t values
    #time_ticks = np.array([24, 48, 72])  # Example: 1, 2, 3 days in hours
    #time_labels = [f'{t/24:.0f}d' for t in time_ticks]
    #ax2.set_xticks(time_ticks)
    #ax2.set_xticklabels(time_labels)
    #ax2.set_xlim(ax1.get_xlim())  # Match limits with primary axis

# Or# if you want to show integration time per energy bin
# Th#is would require storing time values for each point

# Ad#d legend with better placement
    #legend = ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    #legend.get_frame().set_edgecolor('black')

# Ad#d title
    #plt.title('MDP vs Energy for Different Crab Flux Levels', fontsize=14, fontweight='bold', pad=20)

# Ad#just layout and add annotations
    #plt.tight_layout()

# Ad#d grid for secondary axis
    #ax2.grid(False)  # Usually secondary axis doesn't need grid

# Cu#stomize tick parameters
    #ax1.tick_params(axis='both', which='major', labelsize=10)
    #ax2.tick_params(axis='x', which='major', labelsize=10)

# Ad#d minor grid
    #ax1.minorticks_on()
    #ax1.grid(which='minor', linestyle=':', alpha=0.2)

    #plt.show()

    
    #for energy, time, mcrab in mdp_energy_dict.keys():


    KEV_TO_ERG = 1.602176634e-9
        


    def integrate_power_law_energy(K, alpha, E_min, E_max):
        """Energy flux: erg/cm²/s"""
        if alpha == 2:
            flux_keV = K * np.log(E_max / E_min)
        else:
            flux_keV = K / (2 - alpha) * (E_max**(2 - alpha) - E_min**(2 - alpha))
        return flux_keV * KEV_TO_ERG

    flux_levels = [1e-11, 1e-10, 1e-9] 
    K = 14.44        # ph/cm²/s/keV
    alpha = 2.169
    

    mdp_dict = {}

    for target_flux in flux_levels:
        for delta_t in time_lst:
            for i, E in enumerate(Energy_lst):

                E_min = max(E - 25, 1e-3)
                E_max = E + 25

                A_eff = np.interp(E, energies, areas)

                # --- Crab photon & energy flux ---
                crab_ph_flux = integrate_power_law(K, alpha, E_min, E_max)
                crab_en_flux = integrate_power_law_energy(K, alpha, E_min, E_max)

                crab_cnts_s = crab_ph_flux * A_eff

                # --- Convert target energy flux → effective Crab fraction ---
                mcrab_eff = target_flux / crab_en_flux

                source_cnts_s = mcrab_eff * crab_cnts_s * lst_abs_eff[i]

                # --- Background ---
                cosmic_background_file = (
                    '/local/home/jf285468/documents/phd/phemto/'
                    'phemto_simulations/megalib/sources/background/'
                    'Data/CosmicPhotons_Spec_600.0km_5.0deg_1100.0solarmod.dat'
                )

                background_dict = {}
                with open(cosmic_background_file) as f:
                    for line in f:
                        if not line.strip() or line.startswith('#'):
                            continue
                        parts = line.split()
                        if parts[0] == 'DP':
                            background_dict[float(parts[1])] = float(parts[2])

                angle_rad = np.deg2rad(90)
                solid_angle = 2 * np.pi * (1 - np.cos(angle_rad))

                bg_flux = integrate_flux(background_dict, E_min, E_max, solid_angle)
                area_bg = np.pi * 0.075**2

                bg_cnts_s = bg_flux * area_bg * lst_abs_eff[i]

                # --- MDP ---
                mdp = compute_MDP(source_cnts_s, lst_Q[i], bg_cnts_s, delta_t)

                mdp_dict[E, delta_t, target_flux] = mdp


    plt.figure(figsize=(8,7))

    for target_flux in flux_levels:
        x, y = [], []
        for E in Energy_lst:
            x.append(E)
            y.append(mdp_dict[E, 1e6, target_flux])

        plt.plot(
            x, y,
            label=rf'$F={target_flux:.0e}$ erg cm$^{{-2}}$ s$^{{-1}}$'
        )

    plt.xlabel('Energy [keV]')
    plt.ylabel(r'MDP (10$^6$ s)')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    #plt.show()
    breakpoint()
        #print(f'Total COSMIC BACKGROUND flux from {E_min} keV to {E_max} keV: {total_flux_cosmic_background:.4e} ph/cm²/s')

        #background_file = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry/mdp/AlbedoPhotonSpectrum.dat'
        #albedo_background_file = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/background/Data/AlbedoPhotons_Spec_550.0km_0.0deg_650.0solarmod.dat'

        #background_dict = {}
        #with open(albedo_background_file, 'r') as f:
        #    lines = f.readlines()

        #    for line in lines:
        #        if '#' in line:
        #            continue
        #        if 'IP' in line:
        #            continue
        #        if 'EN' in line:
        #            continue
        #        energy = float(line.split(' ')[1])
        #        flux = float(line.split(' ')[-1])

        #        background_dict[energy] = flux

        #solid_angle_albedo = 4 * np.pi
        #solid_angle_albedo = 3.917
        #total_flux_albedo_background = integrate_flux(background_dict, E_min, E_max, solid_angle_albedo)
        #print(f'Total ALBEDO BACKGROUND flux from {E_min} keV to {E_max} keV: {total_flux_albedo_background:.4e} ph/cm²/s')



# Usage


    relative_eff = compton.get_relativeComptonEff(source_folder, best_min_dist, best_angle_bin, max_dist)
    q_100, q_uncert = compton.get_Q(source_folder, best_min_dist, best_angle_bin, max_dist)

    config_file = "/home/josesousa/Documents/thor/detector/detSoftware/detanalysis/polarimetry/config_prettyPlots.json"
    

    specLib.global_config = specLib.Config(config_file)
    
    sources = specLib.global_config.sources
    sources_peaks = specLib.global_config.sources_peaks
    abct_folder = specLib.global_config.abct_folder
    output_folder_base = specLib.global_config.output_folder
    input_folder = specLib.global_config.input_dir

    chip = 'K10-W0060'
    chip_id = specLib.get_chip_id(chip)


    calib = Calibration('', '')


   
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


    
    simulation_folder = f'{output_folder_base}/2-Simulation'

    background_file = f'{simulation_folder}/thor_background.txt'

    crab_file = f'{simulation_folder}/thor_crabsource.txt'



    result_folder = f'{output_folder_base}/3-GrenobleGeneralConclusions'
    list_energies = [100, 150, 200, 250, 300, 500, 1000]
    theoretical_QvrsEnergy(result_folder, list_energies)
    #breakpoint()

    
    background_dict = {}
    with open(background_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if '#' in line:
                continue
            energy = float(line.split(' ')[0])
            flux = float(line.split(' ')[-1])

            background_dict[energy] = flux


    x_min = 100
    x_max = 100.5
    total_flux = integrate_flux(background_dict, x_min, x_max)
    print(f'Total flux from {x_min} keV to {x_max} keV: {total_flux:.4e}')

    

    crab_energy_list = []
    crab_flux_list = []
    crab_dict = {}

    with open(crab_file, 'r') as f:
        lines = f.readlines()

        for line in lines:
            energy = float(line.split(' ')[0])
            flux = float(line.split(' ')[-1])

            crab_dict[energy] = flux




    experimental_dict = {}

    energy_source_list = []

    for source in sources:

        energy_source = compton.get_energy_from_source_name(source)
        rot_source = compton.get_rot_from_source_name(source)

        if rot_source != 0:
            continue

        source_folder = os.path.join(output_folder_base, source)
        result_folder = f'{output_folder_base}/3-GrenobleGeneralConclusions'

        energy_source_list.append(energy_source)

        max_dist = 4.18


        min_dist_start = 0.055  # mm
        min_dist_end = 18*0.055 - 3*0.055       # mm
        min_dist_step = 0.055

        min_dist_list = list(np.arange(min_dist_start, min_dist_end + min_dist_step , min_dist_step))
        angle_bin_list = [x for x in range(1, 2) if 360 % x == 0]

        merit, best_min_dist, best_angle_bin, sigma_merit = compton.get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist, abs=True)

        test_distance = 0.11 
        relative_eff = compton.get_relativeComptonEff(source_folder, best_min_dist, best_angle_bin, max_dist)
        q_100, q_uncert = compton.get_Q(source_folder, best_min_dist, best_angle_bin, max_dist)

        experimental_dict[energy_source] = (relative_eff,q_100)


    print(experimental_dict)

    print(calib.resolution(100))

    seconds_inDay = 60 * 60 * 24

    mdp_dict = {} #key:day value:mdp

    n_days = 60
    print(energy_source_list)

    integrated_background_flux = []
    
    energy_preivouse = 0

    for n in range(n_days+1):
        if n == 0:
            continue
        for energy in energy_source_list:
            delta_t = 60 * 60 * 24 * n
            crab_flux = float(crab_dict[energy])
            total_crab_events = crab_flux * delta_t
            print(f' Energy: {energy} keV')
            print(f'crab flux = {crab_flux} cnts/s')
            print(f'Total Crab events = {total_crab_events}')
            background_min_energy_integrate = energy - energy*calib.resolution(energy)
            background_max_energy_integrate = energy + energy*calib.resolution(energy)
            print(f'Energy_min_integrate = {background_min_energy_integrate} keV')
            print(f'Energy_max_integrate = {background_max_energy_integrate} keV')
            background_flux = float(integrate_flux(background_dict, background_min_energy_integrate, background_max_energy_integrate))
            print(f'Background flux = {background_flux} cnts/s')
            total_background_events = background_flux * delta_t
            print(f'Total background events: {total_background_events}')
            compton_eff = float(experimental_dict[energy][0])
            q_100 = float(experimental_dict[energy][-1])
            print(f'Relative Compton eff = {compton_eff}')
            print(f'Q_100 = {q_100}')
            total_crab_comptons = compton_eff * total_crab_events
            print(f'Total Crab Compton events = {total_crab_comptons}')
            total_background_compton_events = total_background_events * compton_eff
            print(f'Total Background Compton events = {total_background_compton_events}')
            
            if energy != energy_preivouse and n==1:
                print(n)
                integrated_background_flux.append(background_flux)

            energy_preivouse = energy

            mdp = compute_MDP(crab_flux, compton_eff, q_100, background_flux, delta_t)
           
            mdp_dict[n, energy] = mdp * 100

            print(f'MDP = {mdp}')
            
            print('')

    crab_energy_list = list(crab_dict.keys())
    crab_flux_list = list(crab_dict.values())

# Create the plot
    plt.figure(figsize=(8, 7))

# Plot background flux
    plt.plot(energy_source_list, integrated_background_flux, label=r'Background counts, $B$', color='blue', marker='o', markersize=9)

# Plot crab source flux
    plt.plot(crab_energy_list, crab_flux_list, label=r'Crab Nebula counts, $N_{Crab}$', color='red', marker='s', markersize=9)

# Labeling and formatting
    plt.xlabel('Energy (keV)')
    plt.ylabel(r'Event Rate (cnts/s)')
    #plt.title('Background and Crab Source Flux')
    plt.legend()
    plt.yscale('log')
    
    plt.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True)
    plt.minorticks_on()
    plt.grid(False)
    
    plt.savefig(f'{result_folder}/crab-background_flux_THOR.png')
# Display the plot

    dict_markers ={'100':'o','150': '^','200': 's','250': 'X','300':'d'}
    dict_colors = {'100':'k', '150': 'blue', '200': 'red', '250': 'orange', '300': 'green'}

    energy_mdp = {}
    for (n, energy), mdp in mdp_dict.items():
        if energy not in energy_mdp:
            energy_mdp[energy] = []
        energy_mdp[energy].append((n, mdp))
    plt.figure(figsize=(8, 7))

    for energy, values in energy_mdp.items():
        marker = dict_markers[f'{energy}']
        color = dict_colors[f'{energy}']

        days, mdp_values = zip(*values)  # Unzip into two lists
        plt.plot(days, mdp_values, label=f'{energy} keV', marker=marker, color = color, markersize = 8)

    plt.xlabel(r'Observation Time, $T$ (Days)')
    plt.ylabel(r'Minimum Detectable Polarization, MDP ($\%$)')

    secax = plt.gca().secondary_xaxis('top', functions=(lambda days: days * 0.0864, 
                                 lambda ms: ms / 0.0864))
    secax.set_xlabel(r'Observation Time, $T$ (Ms)', labelpad = 10)

    #plt.title('MDP Evolution over Time for Different Energies')
    #plt.vlines(5, 0, 100, colors='k', linestyle='--')
    plt.legend()
    plt.ylim(0,100)
    plt.grid(False)
    plt.minorticks_on()
    plt.tick_params(axis='both', which='both', right = True)
    plt.savefig(f'{result_folder}/MDP_THOR_best_MeritFigure.png')


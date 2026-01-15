from math import sqrt
from unicodedata import normalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import subprocess
import sys
print(sys.path)
import manalysis.specLib as specLib
import manalysis.comptons as compton
#import polarimetry as polarimetry
import manalysis.polarizationfits as fits

from datetime import datetime
from calibration.calibration import Calibration

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

def integrate_flux(flux_dict, x_min, x_max):
    
    filtered_dict = {k: v for k, v in flux_dict.items() if x_min <= k <= x_max}
    
    if len(filtered_dict) < 2:
        raise ValueError("Not enough data points in the specified energy range.")
    
    # Sort by energy (keys)
    energies = np.array(sorted(filtered_dict.keys()))
    fluxes = np.array([filtered_dict[e] for e in energies])

    integrated_flux = np.trapezoid(fluxes, energies)
    
    return integrated_flux


def compute_MDP(source_flux_times_effective_area, compton_relative_eff, q_100, background, delta_t):
   
    MDP = (4.29/(q_100*compton_relative_eff*source_flux_times_effective_area)) * np.sqrt((compton_relative_eff * source_flux_times_effective_area + background*compton_eff)/delta_t)

    #if MDP > 1:
    #    MDP = 1

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
    plt.savefig(f'{output_folder}/theoretical_QvsThetaCompton.png')





if __name__ == '__main__':
    print('jellow world')

# Power law parameters
    K = 14.44  # ph/cm²/s/keV
    alpha = 2.169

    def integrate_power_law(K, alpha, x_min, x_max):
        if alpha == 1:
            # Special case where alpha = 1 (avoids division by zero)
            return K * np.log(x_max / x_min)
        else:
            return K / (1 - alpha) * (x_max**(1 - alpha) - x_min**(1 - alpha))

# Energy range in keV
    x_min = 277.3614 
    x_max = 322.6386

# Integrate power law flux
    integrated_flux = integrate_power_law(K, alpha, x_min, x_max)
    print(f'Total integrated flux from {x_min} keV to {x_max} keV: {integrated_flux:.4e} ph/cm²/s')
    

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


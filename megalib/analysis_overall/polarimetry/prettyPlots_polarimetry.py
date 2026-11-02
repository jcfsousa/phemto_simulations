from math import sqrt
from unicodedata import normalize
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import LogLocator, NullFormatter
import manalysis.pathlib as pathlib
import subprocess
import sys
print(sys.path)
import manalysis.specLib as specLib
import manalysis.comptons as compton
import polarimetry as polarimetry
import manalysis.polarizationfits as fits
import manalysis.configlib as configlib
import matplotlib.ticker as ticker
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

def format_log_axes(ax):

    ax.minorticks_on()

    ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.8)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.6)

    ax.xaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto'))

    ax.yaxis.set_major_locator(ticker.LogLocator(base=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs='auto'))

def get_source_emmited_photons(outputFolder, energy):

    abs_eff = compton.get_sim_efficiency(energy)

    emmited_photons = compute_source_emmited_photons(outputFolder, abs_eff)

    return emmited_photons, np.sqrt(emmited_photons)


def compute_source_emmited_photons(outputFolder, abs_eff):
    event_counts_path = f'{outputFolder}/AllEventsCount.txt'

    with open(event_counts_path) as f:
        line = f.readlines()
        line_totalEvents = line[0]
        line_totalEvents_split = line_totalEvents.split(' ')
        total_counts = int(line_totalEvents_split[-1])
        
        emmited_photons = total_counts / abs_eff

        return emmited_photons


def determine_compton_abs_eff(result_polarimetry, n_source_photons, angle_bin_list, min_dist_list, max_dist, detector):
    
    result_polarimetry = f'{result_polarimetry}/all'
    for i, min_dist in enumerate(min_dist_list):
        for j, angle_bin in enumerate(angle_bin_list):
            angle_bin_str = str(angle_bin).replace('.','-')
            min_dist_str = str(min_dist).replace('.','-')
            max_dist_str = str(max_dist).replace('.','-')

            folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
            pathlib.creat_dir(folder_result_polarimetry)
            file = f'{folder_result_polarimetry}/ComptonsEventsCount_DETECTOR{detector}.txt'
            
            print(f"FILE {file}")

            with open(file, 'r') as f:
                lines = f.readlines()

            new_lines = []
            compton_events_used = None
            compton_abs_eff_written = False
            for line in lines:
                if '# Compton Events Used' in line:
                    compton_events_used = int(line.split(" ")[-1])  # Extract the number of Compton events
                if '# Compton Abs eff' in line:
                    continue
                if '# Sigma Compton Abs eff:' in line:
                    continue
                if '# Merit Figure Abs' in line:
                    continue
                if '# Sigma Merit Figure Abs' in line:
                    continue
                new_lines.append(line)  # Keep all other lines

            # Calculate and write the new Compton Abs eff and Sigma Compton Abs eff
            if compton_events_used is not None:
                compton_abs_eff = compton_events_used / n_source_photons
                new_lines.append(f"# Compton Abs eff: {compton_abs_eff}\n")
                
                sigma_compton_abs_eff = compute_abseff_sigma(compton_events_used, n_source_photons)
                new_lines.append(f"# Sigma Compton Abs eff: {sigma_compton_abs_eff}\n")

            # Write all lines back to the file
            with open(file, 'w') as f:
                f.writelines(new_lines)



def compute_abseff_sigma(compton_events_used, source_photons):
    sigma_eff = np.sqrt((np.sqrt(compton_events_used)/source_photons)**2 + (-(compton_events_used/(source_photons**2))*np.sqrt(source_photons))**2)
    return sigma_eff

def determine_meritFigure_abs(result_polarimetry, angle_bin_list, min_dist_list, max_dist, detector):
    result_polarimetry = f'{result_polarimetry}/all'
    for i, min_dist in enumerate(min_dist_list):
        for j, angle_bin in enumerate(angle_bin_list):
            angle_bin_str = str(angle_bin).replace('.','-')
            min_dist_str = str(min_dist).replace('.','-')
            max_dist_str = str(max_dist).replace('.','-')

            folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
            pathlib.creat_dir(folder_result_polarimetry)
            file_eff = f'{folder_result_polarimetry}/ComptonsEventsCount_DETECTOR{detector}.txt'
            file_q = f'{folder_result_polarimetry}/Fit_Values_DETECTOR{detector}.txt'

            Q, q_uncertanty = compton.get_Q(folder_result_polarimetry, detector)

            compton_abs_eff, sigma_compton_abs_eff = compton.get_absoluteComptonEff(folder_result_polarimetry, detector)

            merit_figure = Q**2 * compton_abs_eff
            sigma_merit_figure = compute_meritfigure_sigma(Q, float(q_uncertanty), compton_abs_eff, sigma_compton_abs_eff)

            with open(file_eff, 'r') as f:
                lines = f.readlines()
            
            merit_abs_exists = any('# Merit Figure Abs:' in line for line in lines)
            sigma_merit_exists = any('# Sigma Merit Figure Abs:' in line for line in lines)


            with open(file_eff, 'w') as f:
                for line in lines:
                    if "# Merit Figure Abs:" in line:
                        f.write(f"\n# Merit Figure Abs: {merit_figure}\n")
                    if "# Sigma Merit Figure Abs:" in line:
                        f.write(f"\n# Sigma Merit Figure Abs: {sigma_merit_figure}\n")
                    else:
                        f.write(line)

                if not merit_abs_exists:
                    merit_figure = (Q**2) * compton_abs_eff
                    f.write(f"\n# Merit Figure Abs: {merit_figure}\n")

                if not sigma_merit_exists:
                    f.write(f"\n# Sigma Merit Figure Abs: {sigma_merit_figure}\n")

def compute_meritfigure_sigma(Q, sigma_Q, eff, sigma_eff):
    sigma = np.sqrt((2*Q*eff*sigma_Q)**2 + ((Q**2) * sigma_eff)**2) # for  F = Q**2 * eff
    #sigma = np.sqrt((np.sqrt(eff) * sigma_Q)**2 + ( 0.5 * (1/(np.sqrt(eff))) * sigma_eff)**2) # for F = Q*\sqrt{eff}
    return sigma


def plot_figureMeritvrsEnergy(output_folder_base, merit_dict):
    energies = list(merit_dict.keys())  # X-axis: energies
    max_merits = [value[0] for value in merit_dict.values()]  # Y-axis: max_merit
    sigma_max_merits = [value[1] for value in merit_dict.values()]  # Error bars: sigma_max_merit

# Create the plot
    plt.figure(figsize=(10, 6))

# Plot with error bars
    plt.errorbar(energies, max_merits, yerr=sigma_max_merits, fmt='o', color='b', ecolor='r', capsize=5, label='Max Merit')

# Customize the plot
    plt.xlabel('Energy (keV)')
    plt.ylabel('Max Merit')
    plt.title('Max Merit vs Energy')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(f'{output_folder_base}/meritFigure_energydependance.png')
    plt.close()


if __name__ == '__main__':
    
    print('THIS SCRIPT IS OUTDATED, NEED TO ADJUST FOR SPECTRAL BINNING - WORKS WITH MID JANUARY PROCESSED POLARIMETRY!!!!!!!!!!!! PUT ON CONFIG FILE THE PATH FOR THIS DATA!!!')
    
    start_time = datetime.now()

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
    input_folder = specLib.global_config.input_dir

    #chip = 'cdte'
    #chip_id = specLib.get_chip_id(chip)



    # Polarimetry constants
    min_dist_start = 0.025  # min dist between compton events, cm
    min_dist_end = 0.05     # cm
    min_dist_step = 0.025
    min_dist_list = list(np.arange(min_dist_start, min_dist_end + min_dist_step , min_dist_step))
    min_dist_list = [0.025]

    #angle_bin_list = [x for x in range(1, 36) if 360 % x == 0]
    angle_bin_list = [15]  #bin size for polarimetry

    max_dist_list = [100000]  # max dist between compton events, cm
    max_dist_list = np.round(max_dist_list, 3)
    max_dist_on_list = max_dist_list[-1]
    max_dist = max_dist_on_list

    merit_dict = {}

    detectors = [0,1] # 0 Si, 1 CdTe
    detector = 'Both'
    for source in sources:

        print(source)
        
        energy = float(compton.get_energy_from_source_name(source))

        pol_type = compton.get_pol_type_from_source_name(source)
        if pol_type == 'NonPol':
            continue
        source_pol = source
        source_Nonpol = source.replace('Pol', 'NonPol')


        folder_input_polarimetry_pol = os.path.join(output_folder_base, source_pol)
        folder_input_polarimetry_Nonpol = os.path.join(output_folder_base, source_Nonpol)

        result_polarimetry_base = output_folder_base
        pathlib.creat_dir(result_polarimetry_base)
        
        source_analysis = source.replace('Pol', '')

        result_polarimetry = os.path.join(result_polarimetry_base, source_analysis)
        pathlib.creat_dir(result_polarimetry)



        # Detectors geometry constants depending on HED geometry
        # CdTe
        z_cdte = -float((source.split("_")[-1]).split('c')[0]) #distance form source name, negative value
        cdte_matrix = int((source.split("_")[1]).split('x')[-1])   # cdte matrix from source name, ex:GaussBeamPol50keV_config4x4_0.5cm
        cdte_single_det_size = 2.45 # 1.6x1.6 cm2
        cdte_detSize = cdte_single_det_size * cdte_matrix # cm
        cdte_pixSize = 0.025 # cm
        # Si
        z_si = 0 # position of Si detector
        si_detSize = 6.656 # cm
        si_pixSize = 0.025 # cm
        #############################

        # Polarimetry constants
        min_dist_list = [0.025, 0.05, 0.075]
        angle_bin_list = [5, 10, 15, 20]  #bin size for polarimetry
        max_dist_list = [100000]  # max dist between compton events, cm
        max_dist_list = np.round(max_dist_list, 3)
        max_dist= max_dist_list[-1]
        #############################


        #calib = Calibration(output_folder_base, None) # useless remove this feature, past artfact...

        #n_emmited_photons, sigma_n_emmited_photons = get_source_emmited_photons(output_folder, source_energy)
        n_emmited_photons = 1e6 # MEGAlib Nevents simulated

        print(f'RESULT_POLARIMETRY: {result_polarimetry}')

        determine_compton_abs_eff(result_polarimetry, n_emmited_photons, angle_bin_list, min_dist_list, max_dist, detector)
        determine_meritFigure_abs(result_polarimetry, angle_bin_list, min_dist_list, max_dist, detector)
        #compton.plot_figureMeritMap(output_folder, min_dist_list, angle_bin_list, max_dist_list, abs = True)
        #max_merit, best_min_dist, best_angle_bin, sigma_max_merit = compton.get_bestPolarimetryConditions(output_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max

        #compton.plot_QvrsBin(output_folder, [best_min_dist], angle_bin_list, max_dist) # with fixed angle bin!!
        #merit_dict[energy] = (max_merit, sigma_max_merit)
    

    #compton.imshow_eventType(output_folder_base, sources, min_dist=0.55, max_dist=4.18, dist_cuts = True, event_type = 'comptons_inPeak', plot_energy_source = 200, plot_rot_source = 0)
    
    source_type = "CollimatedBeam"

    #lst_source_energy = [15,20, 25, 30, 35, 40, 45, 50, 55, 60, 80,100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600]
    lst_source_energy = [10, 15, 35, 45, 75, 100]
    #lst_distance_dets = [1]
    lst_distance_dets = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    lst_HED_config = [3]

    min_dist = 0.025
    angle_bin = 15
    max_dist = 100000
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    def plot_Q_dist_fixedE(base_source_folder, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist):
         
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')
        for HED_config in lst_HED_config:
            for source_energy in lst_source_energy:
                lst_Q = []
                lst_Q_uncert = []
                lst_distance = []
                for distance_dets in lst_distance_dets:
                    result_polarimetry = f"{base_source_folder}/{source_type}{source_energy}keV_config{HED_config}x{HED_config}_{distance_dets}cm"
                    result_polarimetry = f'{result_polarimetry}/all'
                    folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
                    Q, Q_uncer = compton.get_Q(folder_result_polarimetry, detector) # For all sources gonna reject first pixel order compton events and use 36deg bin
                    lst_Q.append(Q)
                    lst_Q_uncert.append(Q_uncer)
                    lst_distance.append(distance_dets)
                lst_Q = np.array(lst_Q)
                plt.errorbar(lst_distance, lst_Q/lst_Q.max(), yerr=lst_Q_uncert/lst_Q.max(),  marker='o', capsize=5,label = f"{source_energy} keV")
            #plt.title(f"HED detector config {HED_config}x{HED_config}")
            plt.xlabel("Distance between LED and HED (cm)")
            plt.ylabel(r"Normalized Modulation Factor, $Q_{100}$")
            plt.ylim(0,1)
            plt.legend()
            #plt.show()
    
    plt.figure(figsize=(12,6))
    plt.subplot(1,3,1)
    plot_Q_dist_fixedE(output_folder_base, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist)
    
    def plot_comptEff_dist_fixedE(base_source_folder, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist):
         
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')
        for HED_config in lst_HED_config:
            for source_energy in lst_source_energy:
                lst_comptEff = []
                lst_comptEff_uncert = []
                lst_distance = []
                for distance_dets in lst_distance_dets:
                    result_polarimetry = f"{base_source_folder}/{source_type}{source_energy}keV_config{HED_config}x{HED_config}_{distance_dets}cm"
                    result_polarimetry = f'{result_polarimetry}/all'
                    folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
                    compton_eff, sigma_compton_eff = compton.get_absoluteComptonEff(folder_result_polarimetry, detector)
                    lst_comptEff.append(compton_eff)
                    lst_comptEff_uncert.append(sigma_compton_eff)
                    lst_distance.append(distance_dets)
                lst_comptEff = np.array(lst_comptEff, dtype=float)
                plt.errorbar(lst_distance, lst_comptEff/lst_comptEff.max(), yerr=lst_comptEff_uncert/lst_comptEff.max(),  marker='o', capsize=5,label = f"{source_energy} keV")
            #plt.title(f"HED detector config {HED_config}x{HED_config}")
            plt.xlabel("Distance between LED and HED (cm)")
            plt.ylabel("Normalized Compton Eff")
            plt.legend()
            #plt.show()

    plt.subplot(1,3,2)
    plot_comptEff_dist_fixedE(output_folder_base, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist)


    def plot_MeritFigure_dist_fixedE(base_source_folder, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist):
        colors = [
            "#4c72b0",
            "#dd8452",
            "#55a868",
            "#c44e52",
            "#8172b3",
            "#937860",
            "#da8bc3",
        ]
        markers = ['o', 's', '^', 'D', 'v', '*', 'p']
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')
        for HED_config in lst_HED_config:
            for i, source_energy in enumerate(lst_source_energy):
                lst_merit = []
                lst_merit_uncert = []
                lst_distance = []
                for distance_dets in lst_distance_dets:
                    result_polarimetry = f"{base_source_folder}/{source_type}{source_energy}keV_config{HED_config}x{HED_config}_{distance_dets}cm"
                    result_polarimetry = f'{result_polarimetry}/all'
                    folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
                    merit, merit_uncert = compton.get_AbsMeritFigure(folder_result_polarimetry, detector)
                    lst_merit.append(merit)
                    lst_merit_uncert.append(merit_uncert)
                    lst_distance.append(distance_dets)
                lst_merit = np.array(lst_merit, dtype=float)
                plt.errorbar(lst_distance, lst_merit/np.max(lst_merit), yerr=lst_merit_uncert/np.max(lst_merit),  marker=markers[i], color=colors[i],capsize=8,label = f"{source_energy} keV")
            #plt.title(f"HED detector config {HED_config}x{HED_config}")
            plt.xlabel("Distance between LED and HED (cm)")
            plt.ylabel(r"Normalized Merit Figure, $Q^2 \times \epsilon_{compton}$")
            plt.xscale('log')
            plt.yscale('log')

            # grid with minor lines
            plt.grid(which='major', linestyle='-', alpha=0.7)
            plt.grid(which='minor', linestyle='--', alpha=0.4)
            plt.legend()
            plt.tight_layout()
            #plt.show()
    
    plt.subplot(1,3,3)
    plot_MeritFigure_dist_fixedE(output_folder_base, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist)
    plt.show()
    
    def plot_Q_dist_fixedConfig(base_source_folder, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist):
         
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')
        for source_energy in lst_source_energy:
            for HED_config in lst_HED_config:
                lst_y = []
                lst_y_uncert = []
                lst_x = []
                for distance_dets in lst_distance_dets:
                    result_polarimetry = f"{base_source_folder}/{source_type}{source_energy}keV_config{HED_config}x{HED_config}_{distance_dets}cm"
                    result_polarimetry = f'{result_polarimetry}/all'
                    folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
                    Y, Y_uncert = compton.get_Q(folder_result_polarimetry, detector)
                    lst_y.append(Y)
                    lst_y_uncert.append(Y_uncert)
                    lst_x.append(distance_dets)
                
                plt.errorbar(lst_x, lst_y, yerr=lst_y_uncert,  marker='o', capsize=5,label = f"{HED_config}x{HED_config}")
            plt.title(f"Source Energy {source_energy} keV")
            plt.xlabel("Distance between LED and HED (cm)")
            plt.ylabel("Modulation Factor, Q100")
            plt.ylim(0,1)
            plt.legend()
            plt.show()

    plot_Q_dist_fixedConfig(output_folder_base, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist)

    def plot_MeritFigure_dist_fixedConfig(base_source_folder, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist):
         
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')
        for source_energy in lst_source_energy:
            for HED_config in lst_HED_config:
                lst_y = []
                lst_y_uncert = []
                lst_x = []
                for distance_dets in lst_distance_dets:
                    result_polarimetry = f"{base_source_folder}/{source_type}{source_energy}keV_config{HED_config}x{HED_config}_{distance_dets}cm"
                    result_polarimetry = f'{result_polarimetry}/all'
                    folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
                    Y, Y_uncert = compton.get_AbsMeritFigure(folder_result_polarimetry, detector)
                    lst_y.append(Y)
                    lst_y_uncert.append(Y_uncert)
                    lst_x.append(distance_dets)
                
                plt.errorbar(lst_x, lst_y, yerr=lst_y_uncert,  marker='o', capsize=5,label = f"{HED_config}x{HED_config}")
            plt.title(f"Source Energy {source_energy} keV")
            plt.xlabel("Distance between LED and HED (cm)")
            plt.ylabel(r"Absolute Merit Figure, $Q^2 \times \epsilon_{compton}$")
            plt.legend()
            plt.show()


    plot_MeritFigure_dist_fixedConfig(output_folder_base, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist)


    def plot_MeritFigure_Energy_fixedConfigfixedDist(ax,base_source_folder, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist):
         
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')
        for HED_config in lst_HED_config:
            for distance_dets in lst_distance_dets:
                lst_y = []
                lst_y_uncert = []
                lst_x = []
                for source_energy in lst_source_energy:
                    result_polarimetry = f"{base_source_folder}/{source_type}{source_energy}keV_config{HED_config}x{HED_config}_{distance_dets}cm"
                    result_polarimetry = f'{result_polarimetry}/all'
                    folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
                    Y, Y_uncert = compton.get_AbsMeritFigure(folder_result_polarimetry, detector)
                    lst_y.append(Y)
                    lst_y_uncert.append(Y_uncert)
                    lst_x.append(source_energy)
                
                ax.errorbar(lst_x, lst_y, yerr=lst_y_uncert, c='k', marker='o', capsize=5,label = f"{HED_config}x{HED_config}")
                ax.set_xlabel("Energy (keV)")
                ax.set_ylabel(r"Merit Figure, $Q^2 \times \epsilon_{compton}$")
                ax.set_yscale('log')
                ax.set_xscale('log')
                format_log_axes(ax)


    fig, axs = plt.subplots(1, 3, figsize=(15,5), sharex=True)

    plot_MeritFigure_Energy_fixedConfigfixedDist(axs[0],output_folder_base, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist)

    def plot_Q_Energy_fixedConfigfixedDist(ax, base_source_folder, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist):
         
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')
        for HED_config in lst_HED_config:
            for distance_dets in lst_distance_dets:
                lst_y = []
                lst_y_uncert = []
                lst_x = []
                for source_energy in lst_source_energy:
                    result_polarimetry = f"{base_source_folder}/{source_type}{source_energy}keV_config{HED_config}x{HED_config}_{distance_dets}cm"
                    result_polarimetry = f'{result_polarimetry}/all'
                    folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
                    Y, Y_uncert = compton.get_Q(folder_result_polarimetry, detector)
                    lst_y.append(Y)
                    lst_y_uncert.append(Y_uncert)
                    lst_x.append(source_energy)
                
                #ax.set_title(f"HED config: {HED_config}x{HED_config} - Distance dets: {distance_dets} cm")
                ax.errorbar(lst_x, lst_y, yerr=lst_y_uncert, c='k', marker='o', capsize=5,label = f"{HED_config}x{HED_config}")
                ax.set_xlabel("Energy (keV)")
                ax.set_ylabel(r"Modulation Factor, Q100")
                ax.set_xscale('log')
                ax.set_xlim(10,1000)
                ax.minorticks_on()
                ax.grid(True, which='major', linestyle='-', linewidth=0.7, alpha=0.8)
                ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.6)

    plot_Q_Energy_fixedConfigfixedDist(axs[1], output_folder_base, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist)


    def plot_ComptEff_Energy_fixedConfigfixedDist(ax, base_source_folder, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist):
         
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')
        for HED_config in lst_HED_config:
            for distance_dets in lst_distance_dets:
                lst_y = []
                lst_y_uncert = []
                lst_x = []
                for source_energy in lst_source_energy:
                    result_polarimetry = f"{base_source_folder}/{source_type}{source_energy}keV_config{HED_config}x{HED_config}_{distance_dets}cm"
                    result_polarimetry = f'{result_polarimetry}/all'
                    folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
                    Y, Y_uncert = compton.get_absoluteComptonEff(folder_result_polarimetry, detector)
                    lst_y.append(Y)
                    lst_y_uncert.append(Y_uncert)
                    lst_x.append(source_energy)
                
                ax.errorbar(lst_x, lst_y, yerr=lst_y_uncert, c='k', marker='o', capsize=5,label = f"{HED_config}x{HED_config}")
                ax.set_xlabel("Energy (keV)")
                ax.set_ylabel(r"Absolute Compton Efficiency")
                ax.set_yscale('log')
                ax.set_xscale('log')
                format_log_axes(ax)

    plot_ComptEff_Energy_fixedConfigfixedDist(axs[2], output_folder_base, lst_source_energy, lst_distance_dets, lst_HED_config, source_type, min_dist, angle_bin, max_dist)
    
    plt.tight_layout()
    plt.savefig(f'../../../results/megalib_v3/detector_polarimetry_performance.png', dpi=600)
    plt.show()
    breakpoint()



































    compton.plot_QvrsBin(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist, min_dist_definition=0.025, energies_overlap=(100, 200, 300, 400, 500)) # with fixed angle bin!! #show

    compton.plot_QvrsRadius_combined_absEffvrsRadius(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist, energies_overlap=(100, 200, 300, 400, 500))
    
    plot_figureMeritvrsEnergy(output_folder_base, merit_dict)

    compton.plot_QvrsEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist) 

    compton.plot_AbsComptonEffvrsEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist) #show

    compton.plot_MeritFigurevrsEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist)

    compton.plot_QAbsComptonEffvrsEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist)

    compton.plot_AbsComptonEffvrsRot_fixedEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist) # show

    compton.plot_QvrsRot_fixedEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist)  # show

    compton.plot_AbsMeritFigurevrsRot_fixedEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist)  # show


    #compton.plot_rotationMeasurements(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist, energies_overlap = (100, 200, 300))
   
    energy_list = (50, 100, 200, 300, 400, 500, 600, 700)
    #for energy in energy_list:
    #    compton.plot_comptonEventsSpectra(output_folder_base, sources, min_dist_list=min_dist_list, angle_bin_list=angle_bin_list, min_dist = 0.025, max_dist=max_dist , plot_energy_source = energy, plot_rot_source = 0)
   
    #energy_list = (200, 250)
    #for energy in energy_list:
    #    compton.plot_comptonEventsThetaDistribution(output_folder_base, sources, min_dist_list=min_dist_list, angle_bin_list=angle_bin_list, min_dist = 0.025, max_dist=max_dist , plot_energy_source = energy, plot_rot_source = 0)
    #compton.plot_comptonEventsEnergyMatrixdistribution(output_folder_base, sources, min_dist_list=min_dist_list, angle_bin_list=angle_bin_list, min_dist = 0.025, max_dist=max_dist , bestF = False, plot_energy_source = 100, plot_rot_source = 0)
    
    #energy_list = ( 300, )
    #for energy in energy_list:
    #    compton.plot_comptonEventsSpectraThetadistribution(output_folder_base, sources, min_dist_list=min_dist_list, angle_bin_list=angle_bin_list, min_dist = 0.055, max_dist=100000 , bestF = False, plot_energy_source = energy, plot_rot_source = 0)
    

    #perform_rmaxStudy = input('Perform R_max study, y/n')
    #rmax_study............

    #compton.plot_EventtypeEffsimulationAbsComptonvrsEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist)
    
    '''
    To perform Abseff study with the rmin and bin fixed
    '''
    perform_rmaxStudy = 'n'

    if perform_rmaxStudy == 'y':
        
        figure_created = False 
        for source in sources:
            
            rot = compton.get_rot_from_source_name(source)

            if rot == 0:

                source_folder = os.path.join(output_folder_base, source)
                result_folder = f'{output_folder_base}/3-GrenobleGeneralConclusions'

                max_dist_computed = 4.18


                min_dist_start = 0.055  # mm
                min_dist_end = 18*0.055 - 3*0.055       # mm
                min_dist_step = 0.055

                min_dist_list = list(np.arange(min_dist_start, min_dist_end + min_dist_step , min_dist_step))
                angle_bin_list = [x for x in range(1, 2) if 360 % x == 0]
                max_dist_list = list(np.arange(min_dist_end+3*0.055 , max_dist_computed + min_dist_step, min_dist_step))
                max_dist_list = np.round(max_dist_list, 3)

                for max_dist in max_dist_list:
                    
                    source_energy = compton.get_energy_from_source_name(source)

                    #already ran 1 time
                    #n_emmited_photons, sigma_n_emmited_photons = get_source_emmited_photons(source_folder, source_energy)
                    #determine_compton_abs_eff(source_folder, n_emmited_photons, angle_bin_list, min_dist_list, max_dist)
                    #determine_meritFigure_abs(source_folder, angle_bin_list, min_dist_list, max_dist)


                    merit, best_min_dist, best_angle_bin, sigma_merit = compton.get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist, abs=True)
                    ##########################
                    best_min_dist = 0.55
                    
                if not figure_created:
                    fig, ax1 = plt.subplots(figsize=(8,7))
                    figure_created = True               
                    ax1, ax2 = compton.plot_AbsEffvrsMaxdist(source_folder, best_min_dist, best_angle_bin, max_dist_list, multiple_energies=True, normalized_q=True, ax1=ax1)
                else:
                    compton.plot_AbsEffvrsMaxdist(source_folder, best_min_dist, best_angle_bin, max_dist_list, multiple_energies=True, normalized_q=True, ax1=ax1, ax2=ax2)


        text = fr'$\Phi$ binning: {best_angle_bin}$^{{\circ}}$\par' + r'\vspace{0.15cm} $r_{min}$' + fr': ${best_min_dist} mm$ '

        plt.annotate(
            text=text,
            xy=(0, 0),  # Bottom left in axes coordinates
            xycoords='axes fraction',  
            xytext=(0.7, 0.1),  # Offset in axes coordinates (fraction of plot size)
            textcoords='axes fraction',
            fontsize=16,
            color='k')

        plt.savefig(f'{result_folder}/AbsEffvrsRmax.png')
    else:
        print('Not plotting the AbsEff vrs Rmax...')




    '''
    To perform Q Rmax study with the rmin and bin fixed
    '''
    perform_rmaxStudy = 'n'

    if perform_rmaxStudy == 'y':
        
        figure_created = False 
        for source in sources:
            
            rot = compton.get_rot_from_source_name(source)

            if rot == 0:

                source_folder = os.path.join(output_folder_base, source)
                result_folder = f'{output_folder_base}/3-GrenobleGeneralConclusions'

                max_dist_computed = 4.18


                min_dist_start = 0.055  # mm
                min_dist_end = 18*0.055 - 3*0.055       # mm
                min_dist_step = 0.055

                min_dist_list = list(np.arange(min_dist_start, min_dist_end + min_dist_step , min_dist_step))
                angle_bin_list = [x for x in range(1, 2) if 360 % x == 0]
                max_dist_list = list(np.arange(min_dist_end+3*0.055 , max_dist_computed + min_dist_step, min_dist_step))
                max_dist_list = np.round(max_dist_list, 3)

                for max_dist in max_dist_list:
                    
                    source_energy = compton.get_energy_from_source_name(source)

                    #already ran 1 time
                    #n_emmited_photons, sigma_n_emmited_photons = get_source_emmited_photons(source_folder, source_energy)
                    #determine_compton_abs_eff(source_folder, n_emmited_photons, angle_bin_list, min_dist_list, max_dist)
                    #determine_meritFigure_abs(source_folder, angle_bin_list, min_dist_list, max_dist)


                    merit, best_min_dist, best_angle_bin, sigma_merit = compton.get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist, abs=True)
                    ##########################


                    best_min_dist = 0.55 ##################### IMPORTANT, FIXED R_MIN TO 0.55MM
                    
                if not figure_created:
                    fig, ax1 = plt.subplots(figsize=(8,7))
                    figure_created = True               
                    ax1, ax2 = compton.plot_QvrsMaxdist(source_folder, best_min_dist, best_angle_bin, max_dist_list, multiple_energies=True, normalized_q=False, ax1=ax1)
                else:
                    compton.plot_QvrsMaxdist(source_folder, best_min_dist, best_angle_bin, max_dist_list, multiple_energies=True, normalized_q=False, ax1=ax1, ax2=ax2)


        text = fr'$\Phi$ binning: {best_angle_bin}$^{{\circ}}$\par' + r'\vspace{0.15cm} $r_{min}$' + fr': ${best_min_dist} mm$ '

        plt.annotate(
            text=text,
            xy=(0, 0),  # Bottom left in axes coordinates
            xycoords='axes fraction',  
            xytext=(0.7, 0.1),  # Offset in axes coordinates (fraction of plot size)
            textcoords='axes fraction',
            fontsize=16,
            color='k')

        plt.savefig(f'{result_folder}/QvrsRmax.png')
    else:
        print('Not plotting the Q vrs Rmax...')

   


    '''
    To plot on the source directory the best polarization plot with rmin rmax and bin varibles that maximize the merit figure
    This function implies the data on the conf.json has the sources sorted by energy and by rot.
    '''

    perform_polarization_plots = 'n'

    if perform_polarization_plots == 'y':


        phi_angle_bin = 1
        rot_list = [100]
        
        check_rot_3plot = 0

        for source in sources:
            source_folder = f'{output_folder_base}/{source}'
            print(source)
        
            
            merit, best_min_dist, best_angle_bin, sigma_merit = compton.get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist, abs=True)

            angle_bin_str = str(best_angle_bin).replace('.','-')
            min_dist_str = str(best_min_dist).replace('.','-')
            max_dist_str = str(max_dist).replace('.','-')

            source_energy = compton.get_energy_from_source_name(source)
            rot = compton.get_rot_from_source_name(source)

            polar_data_folder = f'{source_folder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
            polar_data_file = f'{polar_data_folder}/{source_energy}keV_Comptons_Bin_Count_{angle_bin_str}bin_md{min_dist_str}_plot_data.csv'
            
            data = pd.DataFrame()
            
            data = pd.read_csv(polar_data_file)

            phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error = fits.fit_binned_counts(data, phi_angle_bin)
            
            
            fits.plot_fit_binned_counts(
            source_folder, data, phi_fit, counts_fit, popt, perr, counts_norm, phi,
            source_energy, phi_angle_bin, best_min_dist, chi2, max_dist=max_dist, multiple_plots=False)


            if rot == 0 and source_energy != 100:
                print('rot0')
                check_rot_3plot +=1
                data0 = data 
                phi_fit0, counts_fit0, popt0, perr0, counts_norm0, phi0, chi20, error0 = phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error
                rot0 = rot

            if rot == 20 and source_energy != 100:
                print('rot20')
                check_rot_3plot +=1
                data20 = data
                phi_fit20, counts_fit20, popt20, perr20, counts_norm20, phi20, chi220, error20 = phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error
                rot20 = rot

            print(check_rot_3plot)
            if rot == 45 and source_energy != 100:
                print('rot45')
                check_rot_3plot +=1
                data45 = data 
                phi_fit45, counts_fit45, popt45, perr45, counts_norm45, phi45, chi245, error45 = phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error
                rot45 = rot

            if check_rot_3plot == 3:
                check_rot_3plot = 0
                data_list = [data20, data45]
                phi_fit_list = [phi_fit20, phi_fit45]
                counts_fit_list = [counts_fit20, counts_fit45]
                counts_norm_list = [counts_norm20, counts_norm45]
                phi_list = [phi20, phi45]
                popt_list = [popt20, popt45]
                perr_list = [perr20, perr45]
                chi2_list = [chi220, chi245]
                rot_list = [rot20, rot45]


                fits.plot_fit_binned_counts(
                    source_folder, data_list, phi_fit_list, counts_fit_list, popt_list, perr_list, counts_norm_list, phi_list,
                    source_energy, phi_angle_bin, best_min_dist, chi2_list, max_dist=max_dist, multiple_plots=True, rot_list = rot_list
                )



        check_rot_3plot = 0
        for source in sources:
            source_folder = f'{output_folder_base}/{source}'
        
            
            merit, best_min_dist, best_angle_bin, sigma_merit = compton.get_bestPolarimetryConditions(source_folder, min_dist_list, angle_bin_list, max_dist, abs=True)

            angle_bin_str = str(best_angle_bin).replace('.','-')
            min_dist_str = str(best_min_dist).replace('.','-')
            max_dist_str = str(max_dist).replace('.','-')

            source_energy = compton.get_energy_from_source_name(source)
            rot = compton.get_rot_from_source_name(source)

            polar_data_folder = f'{source_folder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
            polar_data_file = f'{polar_data_folder}/{source_energy}keV_Comptons_Bin_Count_{angle_bin_str}bin_md{min_dist_str}_plot_data.csv'
            
            data = pd.DataFrame()
            
            data = pd.read_csv(polar_data_file)

            phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error = fits.fit_binned_counts(data, phi_angle_bin)

            fits.plot_fit_binned_counts_polar(source_folder, data, phi_fit, counts_fit, popt, perr, counts_norm, phi, source_energy, phi_angle_bin, best_min_dist, chi2, max_dist)

            if rot == 0 and source_energy != 100:
                check_rot_3plot +=1
                data0 = data 
                phi_fit0, counts_fit0, popt0, perr0, counts_norm0, phi0, chi20, error0 = phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error
                rot0 = rot
            
            if rot == 20 and source_energy != 100:
                check_rot_3plot +=1
                data20 = data
                phi_fit20, counts_fit20, popt20, perr20, counts_norm20, phi20, chi220, error20 = phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error
                rot20 = rot

            if rot == 45 and source_energy != 100:
                check_rot_3plot +=1
                data45 = data 
                phi_fit45, counts_fit45, popt45, perr45, counts_norm45, phi45, chi245, error45 = phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error
                rot45 = rot

            if check_rot_3plot == 3:
                check_rot_3plot = 0
                data_list = [data20, data45]
                phi_fit_list = [phi_fit20, phi_fit45]
                counts_fit_list = [counts_fit20, counts_fit45]
                counts_norm_list = [counts_norm20, counts_norm45]
                phi_list = [phi20, phi45]
                popt_list = [popt20, popt45]
                perr_list = [perr20, perr45]
                chi2_list = [chi220, chi245]
                rot_list = [rot20, rot45]

                fits.plot_fit_binned_counts_polar(
                    source_folder, data_list, phi_fit_list, counts_fit_list, popt_list, perr_list, counts_norm_list, phi_list,
                    source_energy, phi_angle_bin, best_min_dist, chi2_list, max_dist=max_dist, multiple_plots=True, rot_list = rot_list)

    else:
        print("Not performing Polarization plots....")





    '''
    To perform the residual polarization analysis.
    '''
    perform_polarimetry_residual = 'n'

    if perform_polarimetry_residual == 'y':

        phi_angle_bin = 1
        #this is the source to use to perform residual polarization 
        source = 'Ba133-col-21Fev2025-closer_356kev'
        source_folder = f'{output_folder_base}/{source}'

        c_list = []
        c_err_list = []
    
        min_dist_to_test = np.arange(0.055, 0.77, 0.055)
        for best_min_dist in min_dist_to_test:
            best_angle_bin = 1

            angle_bin_str = str(best_angle_bin).replace('.','-')
            min_dist_str = str(best_min_dist).replace('.','-')
            max_dist_str = str(max_dist).replace('.','-')

            source_energy = compton.get_energy_from_source_name(source)

            polar_data_folder = f'{source_folder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
            polar_data_file = f'{polar_data_folder}/{source_energy}keV_Comptons_Bin_Count_{angle_bin_str}bin_md{min_dist_str}_plot_data-residual.csv'
            
            data = pd.DataFrame()
            
            data = pd.read_csv(polar_data_file)

            phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error = fits.fit_binned_counts_residual(data, phi_angle_bin)

            fits.plot_fit_binned_counts_residual_polar(source_folder, data, phi_fit, counts_fit, popt, perr, counts_norm, phi, source_energy, phi_angle_bin, best_min_dist, chi2)

            fits.plot_fit_binned_counts_residual(
                source_folder, data, phi_fit, counts_fit, popt, perr, counts_norm, phi,
                source_energy, phi_angle_bin, best_min_dist, chi2)

            c = popt[2]
            c_err = perr[2]

            c_list.append(c)
            c_err_list.append(c_err)

        plt.figure(figsize=(7.5,7))
        plt.errorbar(min_dist_to_test, c_list, yerr=c_err_list, color='k', label=r'Data', marker='o', markersize = 7, capsize=3)
        plt.plot(min_dist_to_test, c_list, linestyle='--', alpha=0.3, color = 'k')
        plt.xlabel(r'$r_{min}$ (mm)')
        plt.ylabel(r'Residual Modulation Amplitude')
        plt.legend(loc='upper right')

        plt.minorticks_on()
        plt.tick_params(axis='both', which='both', top=True, bottom=True, right = True)
        plt.grid(False)
        plt.tight_layout()
        
        print(output_folder_base)
        plt.savefig(f'{output_folder_base}/3-GrenobleGeneralConclusions/residualmodulation_Cparameter_vs_Rmin.png')
    else:
        print('Didnt perform the Residual Polarimetry....')



'''
    for source in sources:
        polarimetry.perform_beam_img(output_folder_base, source, 2)

        breakpoint()

'''

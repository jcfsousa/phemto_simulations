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
import polarimetry as polarimetry
import manalysis.polarizationfits as fits

from datetime import datetime
from calibration.calibration import Calibration


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


def determine_compton_abs_eff(outputFolder, source_photons, angle_bin_list, min_dist_list, max_dist):
    
    for i, min_dist in enumerate(min_dist_list):
        for j, angle_bin in enumerate(angle_bin_list):
            angle_bin_str = str(angle_bin).replace('.','-')
            min_dist_str = str(min_dist).replace('.','-')
            max_dist_str = str(max_dist).replace('.','-')

            already_compton_writen = False
            folder = f'{outputFolder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
            print(folder)
            file = f'{folder}/ComptonsEventsCount.txt'

            #with open(file, 'r+') as f:
            #    for line in f:
            #        print(line)
            #        if "# Compton Events Used" in line:
            #            for line_compton in f:
            #                print(line_compton)
            #                if "# Compton Abs eff" in line_compton:
            #                    already_compton_writen = True
            #            if already_compton_writen:
            #                continue
            #            else:
            #                compton_line = line.split(" ")
            #                compton_events_used = int(compton_line[-1])
            #                compton_abs_eff = compton_events_used/source_photons
            #                f.write(f"\n# Compton Abs eff: {compton_abs_eff}")
            #                continue

            with open(file, 'r') as f:
                lines = f.readlines()

            #compton_abs_exists = any('# Compton Abs eff:' in line for line in lines)
            #sigma_compton_exists = any('# Sigma Compton Abs eff:' in line for line in lines)

            #with open(file, 'w') as f:
            #    for line in lines:
            #        print(line)
            #        if '# Compton Events Used' in line:
            #            compton_events_used = int(line.split(" ")[-1])
            #        if '# Compton Abs eff' in line:
            #            compton_abs_eff = compton_events_used/source_photons
            #            print('yooooo')
            #            f.write(f"# Compton Abs eff: {compton_abs_eff}")
            #        if '# Sigma Compton Abs eff:' in line:
            #            sigma_compton_abs_eff = compute_abseff_sigma(compton_events_used, source_photons)
            #            f.write(f"\n# Sigma Compton Abs eff: {sigma_compton_abs_eff}")
            #            continue
            #        if '# Merit Figure Abs' in line:
            #            continue
            #        else:
            #            if line.strip() == "":
            #                continue
            #            f.write(line)


            #    if not compton_abs_exists:
            #        compton_abs_eff = compton_events_used/source_photons
            #        f.write(f"# Compton Abs eff: {compton_abs_eff}")
            #    if not sigma_compton_exists:
            #        sigma_compton_abs_eff = compute_abseff_sigma(compton_events_used, source_photons)
            #        f.write(f"# Sigma Compton Abs eff: {sigma_compton_abs_eff}")

            new_lines = []
            compton_events_used = None
            compton_abs_eff_written = False
            for line in lines:
                if '# Compton Events Used' in line:
                    compton_events_used = int(line.split(" ")[-1])  # Extract the number of Compton events
                if '# Compton Abs eff' in line:
                    # Skip the old line (we'll write a new one later)
                    continue
                if '# Sigma Compton Abs eff:' in line:
                    # Skip the old line (we'll write a new one later)
                    continue
                if '# Merit Figure Abs' in line:
                    # Skip the old line (we'll write a new one later)
                    continue
                if '# Sigma Merit Figure Abs' in line:
                    continue
                new_lines.append(line)  # Keep all other lines

# Calculate and write the new Compton Abs eff and Sigma Compton Abs eff
            if compton_events_used is not None:
                compton_abs_eff = compton_events_used / source_photons
                new_lines.append(f"# Compton Abs eff: {compton_abs_eff}\n")
                
                sigma_compton_abs_eff = compute_abseff_sigma(compton_events_used, source_photons)
                new_lines.append(f"# Sigma Compton Abs eff: {sigma_compton_abs_eff}\n")

# Write all lines back to the file
            with open(file, 'w') as f:
                f.writelines(new_lines)



def compute_abseff_sigma(compton_events_used, source_photons):
    sigma_eff = np.sqrt((np.sqrt(compton_events_used)/source_photons)**2 + (-(compton_events_used/(source_photons**2))*np.sqrt(source_photons))**2)
    return sigma_eff

def determine_meritFigure_abs(outputFolder, angle_bin_list, min_dist_list, max_dist):
    for i, min_dist in enumerate(min_dist_list):
        for j, angle_bin in enumerate(angle_bin_list):
            angle_bin_str = str(angle_bin).replace('.','-')
            min_dist_str = str(min_dist).replace('.','-')
            max_dist_str = str(max_dist).replace('.','-')

            folder = f'{outputFolder}/photonPolarimetry/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}'
            print(folder)
            file_eff = f'{folder}/ComptonsEventsCount.txt'
            file_q = f'{folder}/Fit_Values.txt'


            Q, q_uncertanty = compton.get_Q(outputFolder, min_dist, angle_bin, max_dist)

            compton_abs_eff, sigma_compton_abs_eff = compton.get_absoluteComptonEff(outputFolder, min_dist, angle_bin, max_dist)

            merit_figure = (Q**2) * compton_abs_eff
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
    sigma = np.sqrt((2*Q*eff*sigma_Q)**2 + ((Q**2) * sigma_eff)**2)

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

    
    start_time = datetime.now()

    config_file = "/home/josesousa/Documents/thor/detector/detSoftware/detanalysis/polarimetry/config_prettyPlots.json"
    

    specLib.global_config = specLib.Config(config_file)
    
    #automatic update the chip config on the calibration.py script (its hard coded idk why)
    result = subprocess.run(['./update_chip_config.sh'],text=True, input = specLib.global_config.config_chips)
    


    sources = specLib.global_config.sources
    sources_peaks = specLib.global_config.sources_peaks
    abct_folder = specLib.global_config.abct_folder
    output_folder_base = specLib.global_config.output_folder
    input_folder = specLib.global_config.input_dir

    chip = 'K10-W0060'
    chip_id = specLib.get_chip_id(chip)


   
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




    merit_dict = {}
    max_dist = 4.18 #maxdist fixed to every grenoble source...

    min_dist_start = 0.055  # mm
    min_dist_end = 2#used for the merit figure
    #min_dist_end = max_dist - 4 * 0.055        # mm
    min_dist_step = 0.055

    min_dist_list = list(np.arange(min_dist_start, min_dist_end + min_dist_step , min_dist_step))
    angle_bin_list = [x for x in range(1, 37) if 360 % x == 0]


    max_dist_list = [max_dist]


    #for source in sources:
        #print(source)

    #    
    #    calib = Calibration(output_folder_base, abct_folder)
        #output_folder = os.path.join(output_folder_base, source)

    #    source_energy = compton.get_energy_from_source_name(source)

    #    n_emmited_photons, sigma_n_emmited_photons = get_source_emmited_photons(output_folder, source_energy)

    #    determine_compton_abs_eff(output_folder, n_emmited_photons, angle_bin_list, min_dist_list, max_dist)

    #    determine_meritFigure_abs(output_folder, angle_bin_list, min_dist_list, max_dist)
    #    
        #compton.plot_figureMeritMap(output_folder, min_dist_list, angle_bin_list, max_dist_list, abs = True)
    #    
    #    max_merit, best_min_dist, best_angle_bin, sigma_max_merit = compton.get_bestPolarimetryConditions(output_folder, min_dist_list, angle_bin_list, max_dist, abs = True) ## gets the best r_min and angle bin for a fixed r_max

    
    #    compton.plot_QvrsBin(output_folder, [best_min_dist], angle_bin_list, max_dist) # with fixed angle bin!!
    #   
    #    rot = compton.get_rot_from_source_name(source)
    #    if rot == 0:
    #        merit_dict[source_energy] = (max_merit, sigma_max_merit)
    

    #compton.imshow_eventType(output_folder_base, sources, min_dist=0.55, max_dist=4.18, dist_cuts = True, event_type = 'comptons_inPeak', plot_energy_source = 200, plot_rot_source = 0)

    #compton.plot_QvrsBin(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist, min_dist_definition=0.55, energies_overlap=(100,150, 200, 250, 300)) # with fixed angle bin!!

    #compton.plot_QvrsRadius_combined_absEffvrsRadius(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist, energies_overlap=(100, 150,200, 250, 300))
    
    #plot_figureMeritvrsEnergy(output_folder_base, merit_dict)

    #compton.plot_QvrsEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist) 

    #compton.plot_AbsComptonEffvrsEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist)

    #compton.plot_MeritFigurevrsEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist)

    #compton.plot_QAbsComptonEffvrsEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist)

    #compton.plot_AbsComptonEffvrsRot_fixedEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist)

    #compton.plot_QvrsRot_fixedEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist) 

    #compton.plot_AbsMeritFigurevrsRot_fixedEnergy(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist) 


    #compton.plot_rotationMeasurements(output_folder_base, sources, min_dist_list, angle_bin_list, max_dist, energies_overlap = (100, 200, 300))
   
    #energy_list = (200, 250)
    #for energy in energy_list:
        #compton.plot_comptonEventsSpectra(output_folder_base, sources, min_dist_list=min_dist_list, angle_bin_list=angle_bin_list, min_dist = 0.055, max_dist=4.18 , plot_energy_source = energy, plot_rot_source = 0)
   
    energy_list = (200, 250)
    for energy in energy_list:
        compton.plot_comptonEventsThetaDistribution(output_folder_base, sources, min_dist_list=min_dist_list, angle_bin_list=angle_bin_list, min_dist = 0.055, max_dist=4.18 , plot_energy_source = energy, plot_rot_source = 0)
    #compton.plot_comptonEventsEnergyMatrixdistribution(output_folder_base, sources, min_dist_list=min_dist_list, angle_bin_list=angle_bin_list, min_dist = 0.055, max_dist=100000 , bestF = False, plot_energy_source = 100, plot_rot_source = 0)
    
    #energy_list = ( 300, )
    #for energy in energy_list:
        #compton.plot_comptonEventsSpectraThetadistribution(output_folder_base, sources, min_dist_list=min_dist_list, angle_bin_list=angle_bin_list, min_dist = 0.055, max_dist=100000 , bestF = False, plot_energy_source = energy, plot_rot_source = 0)
    

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

import mdp as mdp
import matplotlib.pyplot as plt
import os
import manalysis.comptons as compton
import numpy as np

if __name__ == '__main__':

    Emin = 50
    Emax = 400
    integration_step = 5
    Energy_lst = np.arange(50, 400+integration_step, integration_step)
    

    source_type = "Crab1Mevents"
    n_event_sourceCollimated = 1e6 # 1Mevents
    distance_dets = 1.5
    HED_config = 4

    # Polarimetry Constants #
    min_dist = 0.025
    angle_bin = 15
    max_dist = 100000
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')
    #                        #
    
    # Getting Q, abs compton eff from polarimetry simulations #
    base_polarimetry_folder = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry_results'
    result_polarimetry = f"{base_polarimetry_folder}/{source_type}_{Emin}-{Emax}keV_config{HED_config}x{HED_config}_{distance_dets}cm"
    folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
    Q, Q_uncer = compton.get_Q(folder_result_polarimetry) 
    n_comptons = compton.get_n_comptons(folder_result_polarimetry)
    abs_eff = n_comptons/n_event_sourceCollimated
    
    # Getting the eff area os lenses
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


    mdp_dict = {}
    mcrab_lst = [1, 0.1, 0.01, 0.001]
    time_lst = np.arange(1000, 1e5+1000,1000)
    for idx, mcrab in enumerate(mcrab_lst):
        for delta_t in time_lst:  # Skip n=0 directly in range
            mdp_lst = []
            lst_total_area = []
            bin_crab_cnts = 0

            ################
            ##    Crab    ##
            ################
            # Integrating Crab Flux and eff_Area lenses
            for i,E in enumerate(Energy_lst):
                if i == 0:
                    continue
                
                bin_min = Energy_lst[i-1]
                bin_max = Energy_lst[i]
                #A_eff = np.interp(E, energies, areas)
                # Power law parameters
                K = 14.44  # ph/cm²/s/keV
                alpha = 2.169
                
                # Integrate power law flux
                # arXiv:astro-ph/0406058v1 2 Jun 2004
                # F(E) = k (E/1kev)^(-alpha) ph/cm2/s/keV
                # We will Integrate flux over 50 kev Bin
                integrated_flux_crab = mcrab * mdp.integrate_power_law(K, alpha, bin_min, bin_max) #ph/cm2/s

                # Integrate the area then divide by bin size
                total_area = mdp.integrate_area_from_dict(area_dict, bin_min, bin_max) / (bin_max-bin_min)
                #total_area = sum_areas_in_range(area_dict, E_min, E_max)
                #print(f'Total Sensitive Area: {total_area} cm^2')
                
                total_crab_cnts_s = integrated_flux_crab * total_area #ph/s

                bin_crab_cnts = bin_crab_cnts + total_crab_cnts_s



                ## for continuos source need to compute the flux and eff ares per bins of energy 
                ## (to take into consideration the eff area with energy) then in the end sum ALL ->cnts/s 
                ##then multiply by compton eff of that energy band and use Q of that energy band
            

                #print(f'Total CRAB integrated flux from {bin_min} keV to {bin_max} keV: {integrated_flux_crab:.4e} ph/cm2/s')
                #print(f'Total CRAB integrated flux from {bin_min} keV to {bin_max} keV: {total_crab_cnts_s:.4e} ph/s')
            
            

            ################
            ## Background ##
            ################
            cosmic_background_file = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/background/Data/CosmicPhotons_Spec_600.0km_5.0deg_1100.0solarmod.dat'
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
            total_flux_cosmic_background = mdp.integrate_flux(background_dict, Emin, Emax, solid_angle) #ph/cm2/s
            area_background = np.pi * 0.075 **2 # Only take into account the background on the laue lenses focus, 1.5mm diameter
            total_background_cnts_s = total_flux_cosmic_background * area_background



            ################
            ##     MDP    ##
            ################
            compton_background_cnts_s = total_background_cnts_s * abs_eff # comptons cnts/s
            compton_crab_cnts_s = bin_crab_cnts * abs_eff # comptons cnts/s
            

            mdp_computed = mdp.compute_MDP(compton_crab_cnts_s, Q, compton_background_cnts_s, delta_t)
            mdp_dict[mcrab, delta_t] = mdp_computed

        print('')
        print(f'Source Flux: {mcrab} Crab')
        print(f'Q: {Q}')
        print(f'Abs Compton eff: {abs_eff}')
        print(f'Delta_t: {delta_t}')
        print(f'Total CRAB integrated flux from {Emin} keV to {Emax} keV: {bin_crab_cnts:.4e} ph/s')
        print(f'Total BACKGROUND integrated flux from {Emin} keV to {Emax} keV: {total_background_cnts_s:.20e} ph/s')
        print(f'Total CRAB COMPTONS integrated flux from {Emin} keV to {Emax} keV: {compton_crab_cnts_s:.20e} ph/s')
        print(f'Total BACKGROUND COMPTONS integrated flux from {Emin} keV to {Emax} keV: {compton_background_cnts_s:.20e} ph/s')
        cnts_to_mdp = mdp.N_for_given_MDP(Q, 0.01)
        cnts_s_to_mdp = cnts_to_mdp/delta_t
        print(f'Required Counts/s for 1% MDP for Q={Q}: {cnts_s_to_mdp}')
        print('##################################################')

    x = []
    y = []
    plt.figure(figsize=(8, 6))
    plt.title(f'Q={Q}, compt_eff={abs_eff}')
    linestyles = ['-', '-', '-', '-']
    colors = ['#0072B2', '#D55E00', '#009E73', 'red']
    markers = ['o', 's', '^', 'o']  # circle, square, triangle
    for delta_t in time_lst:
        if delta_t != 1e5:
            continue

        for mcrab, linestyle, color, marker in zip(mcrab_lst, linestyles, colors, markers):
            plt.scatter(mcrab, mdp_dict[mcrab,delta_t] * 100, color=color, label=mdp.format_flux_label(mcrab)) # MDP %
            plt.plot(x,y, linewidth = 2, linestyle=linestyle, alpha=0.3, c=color)

        plt.xlabel('Flux [mCrab]', fontsize=13)
        exp = int(np.log10(delta_t))
        coeff = delta_t / (10**exp)
        plt.ylabel(f'MDP [%] (t = 10$^{{{exp}}}$ s)', fontsize=13)
        plt.yscale('log')
        plt.xscale('log')
        plt.axhline(1, color='darkred', linestyle='--', linewidth=2)
        #plt.title('Minimum Detectable Polarization', fontsize=14)
        plt.legend(fontsize=11 ,frameon=False, loc='best')

        plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.4)

        plt.tight_layout()
    plt.show()


    


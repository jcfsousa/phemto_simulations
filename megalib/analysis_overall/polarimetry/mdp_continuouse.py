import mdp_collimated as mdp
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import manalysis.comptons as compton
import numpy as np

def integrate_product(area_dict, flux_func, E_min, E_max, n_points=1000):
    """
    Integrate the product of effective area and flux over energy range.
    
    Parameters:
    -----------
    area_dict : dict
        Dictionary with energy keys and effective area values
    flux_func : callable
        Function that returns flux at given energy (ph/cm²/s/keV)
    E_min, E_max : float
        Energy range for integration
    n_points : int
        Number of points for integration grid
    
    Returns:
    --------
    total_rate : float
        Integrated count rate (ph/s)
    """
    # Extract energies and areas from dictionary
    energies = np.array(list(area_dict.keys()))
    areas = np.array(list(area_dict.values()))
    
    # Sort by energy
    sort_idx = np.argsort(energies)
    energies = energies[sort_idx]
    areas = areas[sort_idx]
    
    # Check range
    if E_min < energies[0] or E_max > energies[-1]:
        print(f"Warning: Integration range [{E_min}, {E_max}] extends beyond data range [{energies[0]}, {energies[-1]}]")
        E_min = max(E_min, energies[0])
        E_max = min(E_max, energies[-1])
    
    if E_min >= E_max:
        return 0.0
    
    # Create interpolation function for area
    from scipy.interpolate import interp1d
    area_interp = interp1d(energies, areas, kind='linear', 
                           bounds_error=False, fill_value=0)
    
    # Create fine grid for integration
    fine_energies = np.linspace(E_min, E_max, n_points)
    
    # Evaluate area and flux on fine grid
    fine_areas = area_interp(fine_energies)
    fine_flux = flux_func(fine_energies)
    
    # Integrate the product
    product = fine_areas * fine_flux
    total_rate = np.trapezoid(product, fine_energies)
    
    return total_rate

# --- Crab spectrum ---
# dN/dE = 11.6e4 * E^-2.18 ph/s/m2/keV
# convert to cm2 → divide by 1e4
def crab_flux_low_energy(E, mcrab=1):
    K = 11.6        # ph/cm2/s/keV
    alpha = 2.18
    return mcrab * K * E**(-alpha)


# --- Sco X-1 spectrum ---
# cutoffpl + powerlaw

def scox1_flux(E):

    # cutoffpl
    phoindex_cut = 0.975255
    highEcut = 4.27166
    norm_cut = 66.7298

    cutoff = norm_cut * E**(-phoindex_cut) * np.exp(-E/highEcut)

    # powerlaw
    phoindex_pl = 1.53128
    norm_pl = 9.84638e-03

    power = norm_pl * E**(-phoindex_pl)

    return cutoff + power

def frac_photons_Crab_bin(Emin, Emax, alpha, File_Emax, File_Emin):
    p = 1.0 - alpha
    return (Emax**p - Emin**p) / (File_Emax**p - File_Emin**p)

if __name__ == '__main__':

    integration_step = 0.1

    #Energy_lst =  np.logspace(1.2, 2, 2000)  #X1
    File_Emin = 15
    File_Emax = 600
    EminEmax_list = [(15,600), (15,50),(50,150),(150,250), (250,350), (350,450), (450,600)]
    global_mdp_dict = {}
    for E_edjes in EminEmax_list:
        Emin, Emax = E_edjes
        #Emin, Emax = 15, 600 #crab low energy
        #Emin, Emax = 20, 30 # Sco X1
        Energy_lst =  np.arange(Emin, Emax+integration_step, integration_step)  #crab

        source_type = "Crab10Mevents"
        n_total_event_sourceCollimated = 1e7 # 1Mevents from File_Emin to File_Emax, need to compute photons emmited from Emin and Emax
        ### crab spectra consts ###
        #K = 14.44  # ph/cm²/s/keV
        #alpha = 2.169
        ##########################
        #n_event_sourceCollimated = frac_photons_Crab_bin(Emin, Emax, alpha, File_Emax, File_Emin) * n_total_event_sourceCollimated
        #print(f"Photons form SOURCE PRE-OPTICS: {n_total_event_sourceCollimated}")
        #print(f"Photons from SOURCE {Emin}-{Emax}keV PRE-OPTICS: {n_event_sourceCollimated}")

        distance_dets = 1
        HED_config = 3

        # Polarimetry Constants #
        min_dist = 0.025
        angle_bin = 15
        max_dist = 100000
        angle_bin_str = str(angle_bin).replace('.','-')
        min_dist_str = str(min_dist).replace('.','-')
        max_dist_str = str(max_dist).replace('.','-')
        #                        #
        
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


        # Power law parameters
        K = 14.44  # ph/cm²/s/keV
        alpha = 2.169
        flux_func = lambda E: K * (E/1.0)**(-alpha) #
        # Integrate power law flux
        # arXiv:astro-ph/0406058v1 2 Jun 2004
        # F(E) = k (E/1kev)^(-alpha) ph/cm2/s/keV
# to write total flux for megalib
        AfterOptics_TOTAL_SourceFlux = integrate_product(area_dict, flux_func, File_Emin, File_Emax) #cnts/s
        AfterOptics_BINNED_SourceFlux = integrate_product(area_dict, flux_func, Emin, Emax) #cnts/s 
        frac_afteroptics_sourceFlux = AfterOptics_BINNED_SourceFlux/AfterOptics_TOTAL_SourceFlux
        n_event_sourceCollimated = n_total_event_sourceCollimated*frac_afteroptics_sourceFlux
        print(f"Photons from SOURCE {Emin}-{Emax}keV: {n_event_sourceCollimated}")


        # Getting Q, abs compton eff from polarimetry simulations #
        base_polarimetry_folder = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry_results'
        result_polarimetry = f"{base_polarimetry_folder}/{source_type}_{File_Emin}-{File_Emax}keV_config{HED_config}x{HED_config}_{distance_dets}cm/bin_{Emin}-{Emax}"
        folder_result_polarimetry = os.path.join(result_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
        Q, Q_uncer = compton.get_Q(folder_result_polarimetry, 'Both') 
        n_comptons = compton.get_n_comptons(folder_result_polarimetry, 'Both')
        abs_eff = n_comptons/n_event_sourceCollimated

        print(f"Q={Q}\nn_comptons={n_comptons}\nCompton_abs_eff{abs_eff}")
        


        mdp_dict = {}
        #mcrab_lst = [1, 0.1, 0.01, 0.001]
        mcrab_lst = [0.1]
        time_lst = [1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
        for idx, mcrab in enumerate(mcrab_lst):
            for delta_t in time_lst:  # Skip n=0 directly in range
                mdp_lst = []
                lst_total_area = []
                bin_crab_cnts = 0

                ################
                ##    Crab high_energy   ##
                ################
                # Integrating Crab Flux and eff_Area lenses
                for i,E in enumerate(Energy_lst):
                    if i == 0:
                        continue
                    
                    bin_min = Energy_lst[i-1]
                    bin_max = Energy_lst[i]
                    #A_eff = np.interp(E, energies, areas)
                    # We will Integrate flux over 50 kev Bin

                    #flux_func = lambda E: crab_flux_low_energy(E, mcrab) #Crab
                    #flux_func = scox1_flux #Sco X1
                    #integrated_flux_crab = mcrab * mdp.integrate_power_law(K, alpha, bin_min, bin_max) #ph/cm2/s

                    # Integrate the area then divide by bin size
                    ####total_area = mdp.integrate_area_from_dict(area_dict, bin_min, bin_max) / (bin_max-bin_min)
                    #total_area = sum_areas_in_range(area_dict, E_min, E_max)
                    #print(f'Total Sensitive Area: {total_area} cm^2')
                    
                    ####total_crab_cnts_s = integrated_flux_crab * total_area #ph/s

                    ####bin_crab_cnts = bin_crab_cnts + total_crab_cnts_s

                    total_crab_cnts_s = mcrab * integrate_product(area_dict, flux_func, bin_min, bin_max)
                    bin_crab_cnts = bin_crab_cnts + total_crab_cnts_s

                    ## for continuos source need to compute the flux and eff ares per bins of energy 
                    ## (to take into consideration the eff area with energy) then in the end sum ALL ->cnts/s 
                    ##then multiply by compton eff of that energy band and use Q of that energy band
                

                    #print(f'Total CRAB integrated flux from {bin_min} keV to {bin_max} keV: {integrated_flux_crab:.4e} ph/cm2/s')
                    #print(f'Total CRAB integrated flux from {bin_min} keV to {bin_max} keV: {total_crab_cnts_s:.4e} ph/s')
                print(f'Counts per second {bin_crab_cnts}')
                print(f' crab {mcrab}')
                
                

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

            print('##################################################')
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
        #plt.figure(figsize=(6, 5))
        #plt.title(f'Q={Q}, compt_eff={abs_eff}')
        linestyles = ['-', '-', '-', '-']
        colors = ['#000000', '#E69F00', '#56B4E9', '#009E73']  # Black, Orange, SkyBlue, BluishGreen
        colors = ['green', 'red', 'orange', 'blue']
        markers = ['s', '^', 'D', 'x']      # Square, Triangle, Diamond, X
        markers = ['s', '^', 'D', 'd']  # 'd' = filled diamond instead of 'x'
        #for delta_t, linestyle, color in zip(time_lst, linestyles, colors):
        #    for mcrab, marker in zip(mcrab_lst, markers):
        #        plt.scatter(delta_t, mdp_dict[mcrab,delta_t] * 100, color=color, label=mdp.format_flux_label(mcrab), s=80, marker=marker) # MDP %
        #        plt.plot(x,y, linewidth = 2, linestyle=linestyle, alpha=0.3, c=color)

        #plt.xlabel('Observation Time [s]', fontsize=13)
        #exp = int(np.log10(delta_t))
        #coeff = delta_t / (10**exp)
        ##plt.ylabel(f'MDP [%] (t = 10$^{{{exp}}}$ s)', fontsize=13)
        #plt.ylabel(f'MDP [%])', fontsize=13)
        #plt.yscale('log')
        #plt.xscale('log')
        #plt.axhline(5, color='darkred', linestyle='--', linewidth=2)
        ##plt.title('Minimum Detectable Polarization', fontsize=14)
        #plt.legend(fontsize=11 ,frameon=False, loc='best')

        #plt.grid(which='both', linestyle='--', linewidth=1, alpha=0.5)

        #plt.tight_layout()


        global_mdp_dict[Emin, Emax] = mdp_dict
        fig, ax = plt.subplots(figsize=(6, 5))

        print(mdp_dict)
        for delta_t, linestyle in zip(time_lst, linestyles):
            for i, (mcrab, marker, color) in enumerate(zip(mcrab_lst, markers, colors)):
                ax.scatter(delta_t, mdp_dict[(mcrab, delta_t)] * 100, 
                          color=color, edgecolor='black', s=80, marker=marker, zorder=3)
                # Connect points for this mcrab/time combo with faint line
                ax.plot(delta_t, mdp_dict[(mcrab, delta_t)] * 100, 
                        linewidth=1.5, linestyle=linestyle, alpha=0.6, 
                        color=color, zorder=1)

        
        ax.set_xlabel('Observation Time [s]', fontsize=12)
        ax.set_ylabel('MDP [%]', fontsize=12)
        #ax.set_title(f'Crab light curve, 50-400 keV\nQ={Q},' + r'$\epsilon_{Compton}$=' + f'{abs_eff:.1%}', fontsize=13, pad=5)

        ax.set_yscale('log')
        ax.set_xscale('log')

        ax.axhline(5, color='darkred', linestyle='--', linewidth=2, zorder=0)
        ax.text(0.78, 0.563, 'SR-3', transform=ax.transAxes, 
                color='darkred', fontsize=12, fontweight = 'bold', 
                bbox=dict(facecolor='white', 
                          edgecolor='darkred', pad = 5),
                verticalalignment='center', horizontalalignment='right')

        ax.text(0.26, 0.01, f'Crab Nebula, {Emin}-{Emax} keV\nQ={Q},' + r' $\epsilon_{Compton}$=' + f'{abs_eff:.1%}', 
                transform=ax.transAxes,
                color='k', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='k', alpha=0, pad = 10),
                verticalalignment='bottom',      
                horizontalalignment='center')    

        from matplotlib.lines import Line2D
        mcrab_handles = [
            Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i], 
                   markersize=8, label=mdp.format_flux_label(mcrab_lst[i]), 
                   markeredgecolor='k', markeredgewidth=0.5)
            for i in range(len(mcrab_lst))
        ]
        mcrab_legend = ax.legend(handles=mcrab_handles, loc='upper right', 
                                fontsize=11, frameon=True, framealpha=0.95, 
                                edgecolor='gray', fancybox=False)


        ax.grid(which='both', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.tick_params(axis='both', which='minor', labelsize=9)

        plt.tight_layout()
        
        #with open('../../../results/megalib_v2/MDP_CrabFlux_50-400keVbin.csv', 'w') as f:
        #    f.write(f'Notes:, time in seconds - mdp (0-1). MDP computed for Crab lightcurve. Integrated energy form 50-400 keV. Simulated 100% and random source. Determined Q100 and Compton efficiency for said energy range. Q={Q} Compton_eff={abs_eff}\n')
        #    f.write('Version,2\n')
        #    f.write('energy,mdp,mCrab\n')
        #    for delta_t, linestyle in zip(time_lst, linestyles):
        #        for i, (mcrab, marker, color) in enumerate(zip(mcrab_lst, markers, colors)):
        #            f.write(f'{delta_t},{mdp_dict[(mcrab, delta_t)]}, {mcrab}\n')

        #plt.savefig(f'../../../results/megalib_v2/MDP_CrabFlux_50-400keVbin.png', dpi=600)
        #plt.show()

    mcrab_target = 0.1

    fig, ax = plt.subplots(figsize=(6,5))

# number of energy bins (for colormap)
    n_bins = len(global_mdp_dict)
    colors = cm.rainbow_r(np.linspace(0,1,n_bins))
    hsv_colors = plt.cm.hsv(np.linspace(0, 1, 7))  # Red -> violet cycle
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray']
    colors = [
        "tab:gray",
        "#4c72b0",
        "#dd8452",
        "#55a868",
        "#c44e52",
        "#8172b3",
        "#937860",
        "#da8bc3",
    ]
    markers = ['o', 's', '^', 'D', 'v', '*', 'p']
    for i, ((Emin, Emax), mdp_dict) in enumerate(global_mdp_dict.items()):
        times = []
        mdps = []
        for (mcrab, t), mdp in mdp_dict.items():
            if mcrab == mcrab_target:
                times.append(t)
                mdps.append(mdp)

        times = np.array(times)
        mdps = np.array(mdps)
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        mdps = mdps[sort_idx]

        label = f"{Emin}-{Emax} keV"
        ax.plot(times, mdps * 100, color=colors[i], alpha=0.8)
        ax.scatter(times, mdps * 100, marker=markers[i], color=colors[i], label=label, s=80)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observation Time [s]", fontsize = 12)
    ax.set_ylabel(r"100 mCrab MDP$_{100}$ [%]", fontsize = 12)
    ax.grid(True, which="major", linestyle='-', alpha=0.7)
    ax.grid(True, which="minor", linestyle=':', alpha=0.4)
    ax.legend(title="Energy Bin")

    #ax.axhline(5, color='darkred', linestyle='--', linewidth=2, zorder=0)
    #ax.text(0.78, 0.563, 'SR-3', transform=ax.transAxes, 
    #        color='darkred', fontsize=12, fontweight = 'bold', 
    #        bbox=dict(facecolor='white', 
    #                  edgecolor='darkred', pad = 5),
    #        verticalalignment='center', horizontalalignment='right')

    plt.tight_layout()
    plt.show()



        


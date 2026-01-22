import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import numpy as np
from scipy.interpolate import interp1d

def get_n_events_total(source_folder):
    file= f'{source_folder}/parse_event_counter.txt'
    try:
        with(open(file, 'r')) as f:
             for line in f:
                 if "Total events" in line:
                     n = line.split(':')[-1].strip()
                     return int(n)
    except Exception as e: 
        print(f"ERROR: {source_folder}/AllEventsCount.txt doesnt exist - Try run polarimetry.py first - {e}")
        sys.exit(1)

def get_n_total_all(df):
    row = df[df["type"] == "ALL"]
    return int(row["total"].iloc[0]) if not row.empty else None

       # file_path_allEvents = f'{folder_source}/AllEventsCount.txt' # go get AllEventsCount that its on Polarized source!!!!
       # df_n_events = pd.read_csv(file_path_allEvents)
       # n_events = get_n_total_all(df_n_events)
if __name__ == '__main__':

    Energy_lst = [4, 10, 30, 50, 100, 150, 200, 250, 300, 350, 400]


    # Polarimetry constants
    source_type = "CollimatedBeam"
    n_event_sourceCollimated = 1e6
    distance_dets = 1.5
    HED_config = 4
    min_dist = 0.025
    angle_bin = 15
    max_dist = 100000
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    base_polarimetry_folder = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry_data'
    lst_abs_eff = []
    for source_energy in Energy_lst:
        folder_source = f"{base_polarimetry_folder}/{source_type}NonPol{source_energy}keV_config{HED_config}x{HED_config}_{distance_dets}cm"
        file_path_allEvents = f'{folder_source}/AllEventsCount.txt' # go get AllEventsCount that its on Polarized source!!!!

        #df_n_events = pd.read_csv(file_path_allEvents, comment="#")
        #n_events = get_n_total_all(df_n_events)
        n_events = get_n_events_total(folder_source)
        abs_eff = n_events/n_event_sourceCollimated
        lst_abs_eff.append(abs_eff)
    
    plt.figure()
    plt.plot(Energy_lst, lst_abs_eff)
    plt.scatter(Energy_lst, lst_abs_eff)
    plt.xlabel('Energy')
    plt.ylabel('Abs Eff')
    #output_file = "absolute_efficiency.txt"

    #with open(output_file, "w") as f:
    #    f.write("energy,abs_eff\n")
    #    for E, eff in zip(Energy_lst, lst_abs_eff):
    #        f.write(f"{E},{eff}\n")

    #print(f"Saved absolute efficiency to {output_file}")

    
    #Area lenses
    file_area_mirror = "/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/sensitivity/area_mirror.csv"
    
    area_dict_mirror = {}
    with open(file_area_mirror, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Energy' in line:
                continue
            parts = line.split(',')
            energy = float(parts[0])  
            area = float(parts[1])

            area_dict_mirror[energy] = area
    
    # Detector efficiency data
    eff_energy = np.array(Energy_lst, dtype=float)
    eff_values = np.array(lst_abs_eff, dtype=float)

    area_mirror_energies = np.array(sorted(area_dict_mirror.keys()))
    area_mirror_values = np.array([area_dict_mirror[e] for e in area_mirror_energies]) 

    eff_interp = interp1d(
        eff_energy,
        eff_values,
        kind="linear",          # physically reasonable
        bounds_error=False,     # allow outside range
        fill_value=0.0          # efficiency = 0 outside simulated range
    )
    interp_eff = eff_interp(area_mirror_energies)
    output_file = "instrument_absolute_efficiency-mirrorinterp.txt"

    with open(output_file, "w") as f:
        f.write("energy,abs_eff\n")
        for E, eff in zip(area_mirror_energies, interp_eff):
            f.write(f"{eff}\n")

    print(f"Saved absolute efficiency to {output_file}")


    # Area lenses 
    file_area_lenses = "/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/sensitivity/area_lenses.csv"
    
    area_dict_lenses = {}
    with open(file_area_lenses, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Energy' in line:
                continue
            parts = line.split(',')
            energy = float(parts[0])  
            area = float(parts[1])

            area_dict_lenses[energy] = area
    
    # Detector efficiency data
    eff_energy = np.array(Energy_lst, dtype=float)
    eff_values = np.array(lst_abs_eff, dtype=float)

# Mirror/lens energy grid
    area_lenses_energies = np.array(sorted(area_dict_lenses.keys()))
    area_lenses_values = np.array([area_dict_lenses[e] for e in area_lenses_energies]) 

    eff_interp = interp1d(
        eff_energy,
        eff_values,
        kind="linear",          # physically reasonable
        bounds_error=False,     # allow outside range
        fill_value=0.0          # efficiency = 0 outside simulated range
    )
    interp_eff = eff_interp(area_lenses_energies)
    output_file = "instrument_absolute_efficiency-lensesinterp.txt"

    with open(output_file, "w") as f:
        f.write("energy,abs_eff\n")
        for E, eff in zip(area_lenses_energies, interp_eff):
            f.write(f"{eff}\n")

    print(f"Saved absolute efficiency to {output_file}")


    #plt.figure()
    #plt.title("Interp eff with area energy points")
    #plt.scatter(area_energies, interp_eff)



    #energies = np.array(sorted(area_dict.keys()))
    #areas = np.array([area_dict[e] for e in energies])

    #plt.figure()
    #plt.plot(energies, areas)
    #plt.scatter(energies, areas)
    #plt.ylabel('Area cm2')
    #plt.xlabel('Energy')

    cosmic_background_file = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/background/Data/CosmicPhotons_Spec_600.0km_5.0deg_1100.0solarmod.dat'

    background_dict = {}
    with open(cosmic_background_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or '#' in stripped or 'IP' in stripped or 'EN' in stripped:
                continue

            parts = stripped.split()
            if len(parts) >= 3 and parts[0] == 'DP':
                energy = round(float(parts[1]),0)  #keV rounding to unit :)
                flux = float(parts[2]) # ph cm-2 s-1 kev-1 sr-1
                background_dict[energy] = flux
    
    plt.figure()
    energies = np.array(list(background_dict.keys()))
    flux = np.array(list(background_dict.values()))
    plt.plot(energies, flux)
    plt.ylabel('ph cm-2 s-1 kev-1 sr-1')
    
    # Solid angle in deg (apperture of cone) to integrate background
    angle_min_deg = 0 
    angle_max_deg = 90
    def deg_to_rad(deg):
        return deg * (np.pi/180)
    angle_max_rad = deg_to_rad(angle_max_deg) # in rad
    angle_min_rad = deg_to_rad(angle_min_deg)
    angle_rad = abs(angle_min_rad - angle_max_rad)
    solid_angle = 2*np.pi * (1 - np.cos(angle_rad))

    flux_cosmic_background = {e: v * solid_angle for e, v in background_dict.items()} # to tranform in [ph cm-2 s-1 kev-1]
    flux_cosmic_background = np.array(list(background_dict.values())) * solid_angle # [ph cm-2 s-1 kev-1]

    # put the new values back in the dict
    for e in background_dict:
        background_dict[e] *= solid_angle
   
    filtered_background = {}
    area_background = np.pi * 0.075 **2 # Only take into account the background on the laue lenses focus, 1.5mm diameter
    for E_background, bkg_flux in background_dict.items():
        filtered_background[E_background] = bkg_flux * area_background # now we have [ph s-1 kev-1]
    background_ph_s_kev = filtered_background
    energies = np.array(list(background_ph_s_kev.keys()))
    fluxes = np.array(list(background_ph_s_kev.values()))
    plt.figure()
    plt.plot(energies, fluxes)
    plt.scatter(energies, fluxes)
    plt.show()
    sys.exit(1)
    ##plt.plot(energies, areas, color='green', label='only lens + mirror')
    #plt.plot(energies, areas, color='k')  
    #plt.xlabel('Energy (keV)')
    #plt.ylabel(r'Lenses + Mirror area (cm$^2$)')
    #plt.yscale('log')
    #plt.legend()
    #plt.close()
    

    mdp_lst = []
    #mcrab_lst = [1,0.1, 0.01, 0.001]
    mcrab_lst = [1, 0.1]

    mdp_dict = {} #key:day value:mdp
    n_days = 3
    time_lst = np.arange(1000, 1e6+1000,1000)
    seconds_inDay = 60 * 60 * 24
    

    mdp_time_dict = {}
    mdp_dict = {}
    cnts_crab_dict = {}
    cnts_background_dict = {}

    for idx, mcrab in enumerate(mcrab_lst):
        for delta_t in time_lst:  # Skip n=0 directly in range
            mdp_lst = []
            lst_total_area = []
            for i,E in enumerate(Energy_lst):

                #A_eff = np.interp(E, energies, areas)
                # Power law parameters
                K = 14.44  # ph/cm²/s/keV
                alpha = 2.169
                
                half_bin = 50 # keV
                E_min = E - half_bin
                if E == 400:
                    E_max = E
                else:
                    E_max = E + half_bin

                # Integrate power law flux
                # arXiv:astro-ph/0406058v1 2 Jun 2004
                # F(E) = k (E/1kev)^(-alpha) ph/cm2/s/keV
                # We will Integrate flux over 50 kev Bin
                integrated_flux_crab = integrate_power_law(K, alpha, E_min, E_max) #ph/cm2/s

                # Integrate the area then divide by bin size
                total_area = integrate_area_from_dict(area_dict, E_min, E_max) / (half_bin*2)
                #total_area = sum_areas_in_range(area_dict, E_min, E_max)
                print(f'Total Sensitive Area: {total_area} cm^2')
                
                total_crab_cnts_s = integrated_flux_crab * total_area #ph/s


                ## for continuos source need to compute the flux and eff ares per bins of energy 
                ## (to take into consideration the eff area with energy) then in the end sum ALL ->cnts/s 
                ##then multiply by compton eff of that energy band and use Q of that energy band
            

                print(f'Total CRAB integrated flux from {E_min} keV to {E_max} keV: {integrated_flux_crab:.4e} ph/cm2/s')
                print(f'Total CRAB integrated flux from {E_min} keV to {E_max} keV: {total_crab_cnts_s:.4e} ph/s')


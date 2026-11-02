import argparse
import sys
from scipy.interpolate import interp1d
import os
import manalysis.comptons as compton
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tkinter as tk 
from tkinter import filedialog
import re
from pathlib import Path
from tqdm import tqdm


# Reference Paths
FILE_AREA = "/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/sensitivity/area_mirror_lenses.csv"
FOLDER_BASE_POLARIMETRY = "/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry_results"
COSMIC_BACKGROUND_FILE = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/background/Data/CosmicPhotons_Spec_600.0km_5.0deg_1100.0solarmod.dat'
DEBUG = False


def compute_MDP(n_source, q_100, n_background, delta_t):
   
    MDP = (4.29/(q_100*n_source)) * np.sqrt((n_source + n_background)/delta_t)
    
    if MDP < 0.00001:
        MDP = 0.00001
    if MDP > 1:
        MDP = 1

    return MDP

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


def creat_polarimetry_bins(BinCenters):
    energy_bin_edges = []
    energy_bin_widths = []
    energy_bin_centers = []  
    for i, E in enumerate(BinCenters):
        if i == 0:
            # First bin: from E - half distance to next point
            half_dist = (BinCenters[i+1] - E) / 2
            low_edge = E
            high_edge = E + half_dist
        elif i == len(BinCenters) - 1:
            # Last bin: from previous point to E + half distance
            half_dist = (E - BinCenters[i-1]) / 2
            low_edge = E - half_dist
            high_edge = E
        else:
            # Middle bins: edges halfway between energy points
            low_edge = E - (E - BinCenters[i-1]) / 2
            high_edge = E + (BinCenters[i+1] - E) / 2
        
        energy_bin_edges.append((low_edge, high_edge))
        energy_bin_widths.append(high_edge - low_edge)
        energy_bin_centers.append(E)

    return energy_bin_edges, energy_bin_widths, energy_bin_centers


def integrate_product(area_dict, E_min, E_max, source_diff_flux = None, source_center_bin = None, flux_func = None, n_points=5000):
    """
    Integrate the product of effective area and flux over energy range.
    
    Parameters:
    -----------
    area_dict : dict
        Dictionary with energy keys and effective area values
    flux_func : callable
        Function that returns flux at given energy (ph/cm2/s/keV)
    source_diff_flux, source_center_bin: lists 
        Differential flux (ph/cm2/s/kev) and the bin center of the source
    E_min, E_max : float
        Energy range for integration
    n_points : int
        Number of points for integration grid
    
    Returns:
    --------
    total_rate : float
        Integrated count rate (ph/s)
    """
    energies = np.array(list(area_dict.keys()))
    areas = np.array(list(area_dict.values()))
    
    # Sort by energy
    sort_idx = np.argsort(energies)
    energies = energies[sort_idx]
    areas = areas[sort_idx]

    if E_min < energies[0] or E_max > energies[-1]:
        print(f"Warning: Integration range [{E_min}, {E_max}] extends beyond data range [{energies[0]}, {energies[-1]}]")
        E_min = max(E_min, energies[0])
        E_max = min(E_max, energies[-1])
    if E_min >= E_max:
        return 0.0
    
    # Create interpolation function for area
    area_interp = interp1d(energies, areas, kind='linear', bounds_error=False, fill_value=0)
    
    fine_energies = np.linspace(E_min, E_max, n_points)
    fine_areas = area_interp(fine_energies)

    if flux_func is None:
        #if the source is given in differential flux
        flux_interp = interp1d(source_center_bin, source_diff_flux, kind='linear', bounds_error=False, fill_value=0)
        fine_flux = flux_interp(fine_energies)
    else:
        fine_flux = flux_func(fine_energies)
    
    # Integrate the product
    product = fine_areas * fine_flux
    total_rate = np.trapezoid(product, fine_energies)
    
    return total_rate #ph/s

def load_optics_effArea(filepath):
    area_dict = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'Energy' in line:
                continue
            parts = line.split(',')
            energy = float(parts[0])  
            area = float(parts[1])
            area_dict[energy] = area
    return area_dict


def parse_Source_name(filename):
    '''
    The nomenclature should be: <source><number>kevents_<Emin>-<Emax>keV_config<n>x<n>_<distance>cm
    At this stage only allows Mevents and kevents... 
    '''
    folder_name = Path(filename).name

    pattern = re.compile(
        r"([A-Za-z]+)"          # source name
        r"(\d+)([Mk])events_"   # number + unit (M or k)
        r"(\d+)-(\d+)keV_"      # energy range
        r"config(\d+)x\d+_"     # config
        r"(\d+)cm",              # distance
        re.IGNORECASE
    )

    match = pattern.match(folder_name)

    if match:
        source = match.group(1)
        value = int(match.group(2))
        unit = match.group(3)
        
        # Convert to integer number of events
        if unit == "M":
            events = value * 1_000_000
        elif unit == "k":
            events = value * 1_000
        else:
            raise ValueError("Unknown unit")

        emin = int(match.group(4))
        emax = int(match.group(5))
        config = int(match.group(6))
        distance = int(match.group(7))

        result = {
            "source": source,
            "events": events,
            "energy_range": (emin, emax),
            "config": config,
            "distance_cm": distance
        }
        return result

def get_flux_crab():
    '''
    arXiv:astro-ph/0406058v1 2 Jun 2004
    F(E) = k (E/1kev)^(-alpha) ph/cm2/s/keV

    Theoretically this model is from >50kev but we use it from 15keV........

    '''
    K = 14.44
    alpha = 2.169
    return lambda E: K * (E)**(-alpha)


def integrate_flux(flux_dict, E_min, E_max, solid_angle):
    
    filtered_dict_energies = {k: v for k, v in flux_dict.items() if E_min <= float(k) <= E_max}
    
    #if len(filtered_dict_energies) < 2:
    #    raise ValueError("Not enough data points in the specified energy range.")
    
    energies = np.array(sorted(filtered_dict_energies.keys()))
    fluxes = np.array([filtered_dict_energies[e] for e in energies])

    integrated_flux = np.trapezoid(fluxes, energies) 

    return integrated_flux * solid_angle # cnts cm-2 s-1

def get_background_count_s(Emin, Emax, polar_angle = 90):
    '''
    Current Background model takes into consideration Cumani LEO background paper for Cosmic Photons at 600km 5deg. It assumes photons coming from 2pi sphere and we only select photons inside PSF area. PSF diameter is 1.5mm
    '''
    background_dict = {}
    with open(COSMIC_BACKGROUND_FILE, 'r') as f:
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
    angle_max_deg = polar_angle
    def deg_to_rad(deg):
        return deg * (np.pi/180)
    angle_max_rad = deg_to_rad(angle_max_deg) # in rad
    angle_min_rad = deg_to_rad(angle_min_deg)
    angle_rad = abs(angle_min_rad - angle_max_rad)
    solid_angle = 2*np.pi * (1 - np.cos(angle_rad))

    #print(f'Solid Angle: {solid_angle} rad')
    total_flux_cosmic_background = integrate_flux(background_dict, Emin, Emax, solid_angle) #ph/cm2/s
    area_background = np.pi * 0.075 **2 # Only take into account the background on the laue lenses focus, 1.5mm diameter
    total_background_cnts_s = total_flux_cosmic_background * area_background # cnts/s

    return total_background_cnts_s

def select_source(base_path="./mdp/cosmic_sources"):

    files = [f for f in os.listdir(base_path) if f.endswith(".xcm") or f.endswith(".dat")]

    if not files:
        print("No source files found.")
        return None, None

    files.sort()

    print("\nAvailable sources:")
    for i, f in enumerate(files, start=1):
        print(f"{i}) {f}")

    while True:
        choice = input("\nSelect a source by number: ")

        if not choice.isdigit():
            print("Please enter a valid number.")
            continue

        idx = int(choice)
        if 1 <= idx <= len(files):
            selected_file = files[idx - 1]
            full_path = os.path.join(base_path, selected_file)
            return full_path, selected_file
        else:
            print("Number out of range.")

def xspec_TBabs_cutoffpl(E, NH, Gamma, Ecut, K):
    return tbabs_approx(E, NH) * cutoffpl(E, Gamma, Ecut, K) # ph s-1 cm-2 kev-1

def cutoffpl(E, Gamma, Ecut, K):
    # E in keV, Gamma is photon index, Ecut in keV, K normalization
    return K * (E**(-Gamma)) * np.exp(-E / Ecut)

def tbabs_approx(E, NH):
    # Extremely crude approximation: sigma(E) ∝ E^{-3}
    # NH in 10^22 cm^-2 (as in TBabs), E in keV
    sigma0 = 3e-22  # adjust as needed
    sigma = sigma0 * (E**-3)
    return np.exp(-NH * 1e22 * sigma)

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

def integrated_XCMmodel_flux(E_min, E_max, flux, n_points=200):
    energies = np.linspace(E_min, E_max, n_points)
    integral = np.trapezoid(flux, energies)
    return integral

if __name__ == '__main__':
    
    # Detector Configuration
    distance_dets = 1 #cm
    HED_config = 3
    Observatory_minE = 1 #keV
    Observatory_maxE = 600 #keV

    # Polarimetry Constants
    min_dist = 0.025
    angle_bin = 15
    max_dist = 100000
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-b', '--bins', action='store_true', help='If selected we will perform MDP from a MEGAlib simulated source that has been Polarimetry.py analysed. After you will be asked to provide each of the Bin_min and Bin_max, please values that were used to perform the Polarimetry.py analysis. AT THE MOMENT ONLY POSSIBLE FOR CRAB')
    parser.add_argument('-c', '--collimated', action='store_true', help='If selected will will be able to import a source flux model that will be convolved with Optics effective area. After you will need to provide the energy center bins for the energy integration. Script will use the database of Collimated Monochromatic polarimetric performace to retrieve the Q_100 and Compton_eff of the center bin and will use this to compute MDP. At this stage use energies present on database. Future: will implement interpolation for energies not in database. IMPORTANT: At this moment we assume 1e6 events generated on MEGAlib for the Collimated database.')

    args = parser.parse_args()

    Optics_effArea = load_optics_effArea(FILE_AREA) # Loadging the optics eff area
    
    plt.figure(figsize=(12,16))
    plt.subplot(3,3,1)
    plt.title('Optics Effective Area')
    plt.plot(Optics_effArea.keys(), Optics_effArea.values(), c = 'k')
    plt.scatter(Optics_effArea.keys(), Optics_effArea.values(), c = 'k')
    plt.ylabel(r'Area [$cm^2$]')
    plt.xlabel('Energy [keV]')
    plt.yscale('log')
    plt.xscale('log')

    if args.bins:
        # Pop-up window to select the source to analyse
        root = tk.Tk()
        root.withdraw()  # hide main window
        Source_filename = filedialog.askdirectory(
        title='Select MEGAlib simulated source Polarimetry.py results folder.')

        ask_bins = True
        MDP_Energy_BinEdges = []
        while ask_bins:
            print("Provide the Emin and Emax range for the energy integration. Be sure to have values concurrent with ones analysed by Polarimetry.py. Write 'stop' to stop asking.")
            emin = input("Emin: ")
            if emin == 'stop':
                ask_bins = False
                continue
            emax = input("Emax: ")
            if emax == 'stop':
                ask_bins = False
                continue
            else:
                MDP_Energy_BinEdges.append((emin, emax))
        
        Source_name_info = parse_Source_name(Source_filename)
        print(Source_name_info)
        File_Emin, File_Emax = Source_name_info['energy_range']
        NEvents_Simulated_Source = Source_name_info['events']
        source_name = Source_name_info['source']



        # For the given source and energy bins going to retrieve the Q_100 and RELATIVE Compton_eff, I need to compute the ammout of photons from source to get the absolute Compton_eff
        # RIGHT NOW ONLY AVAILBLE FOR CRAB
        if source_name == 'Crab':
            Flux_MEGAlib_Source = get_flux_crab() #ph/s/cm2/kev
        else:
            print("Currently onle works with Crab")
            sys.exit(1)

        Flux_AfterOptics_TOTAL = integrate_product(Optics_effArea, File_Emin, File_Emax, flux_func = Flux_MEGAlib_Source) #cnts/s
        
        plt.subplot(3,3,2)
        e_plot_step = 0.1 #keV
        print(MDP_Energy_BinEdges)
        E_plot = np.arange(Observatory_minE, Observatory_maxE + e_plot_step, e_plot_step)
        plt.plot(E_plot, Flux_MEGAlib_Source(E_plot), c = 'k', label=f'{source_name} flux')
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(f'{source_name} Flux [ph/s/cm2/keV]')
        plt.xlabel('Energy [keV]')

        global_mdp_dict = {}
        PLOT_Flux_AfterOptics_int = []
        PLOT_Flux_AfterOptics_Compton = [] 
        PLOT_Flux_AfterOptics_Compton_Background = []
        PLOT_Emin = []
        PLOT_Emax = []
        PLOT_E = []
        PLOT_Q = []
        PLOT_Q_Uncer = []
        PLOT_Compton_Eff = []
        for bin_edges in MDP_Energy_BinEdges:
            Emin, Emax = bin_edges
            Emin = int(Emin)
            Emax = int(Emax)
            Flux_AfterOptics_BINNED = integrate_product(Optics_effArea, Emin, Emax, flux_func=Flux_MEGAlib_Source) #cnts/s 
            frac_AfterOptics_SourceFlux = Flux_AfterOptics_BINNED/Flux_AfterOptics_TOTAL 
            n_events_Source = NEvents_Simulated_Source*frac_AfterOptics_SourceFlux

            print(f"Photons from SOURCE {Emin}-{Emax}keV: {n_events_Source}")

            result_polarimetry_folder = f"{Source_filename}/bin_{Emin:g}-{Emax:g}/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}"
            Q, Q_uncer = compton.get_Q(result_polarimetry_folder, 'Both')  # MDP for both detectors together, for further analysis play around with mdp just on one, sperate, etc...
            n_comptons = compton.get_n_comptons(result_polarimetry_folder, 'Both')
            Compton_abs_eff = n_comptons/n_events_Source

            PLOT_Q.append(Q)
            PLOT_Q_Uncer.append(Q_uncer)
            PLOT_Compton_Eff.append(Compton_abs_eff)

            print(f"Q={Q}\nn_comptons={n_comptons}\nCompton_abs_eff{Compton_abs_eff}")

            integration_step = 1 # keV
            Energy_lst = np.arange(Emin, Emax+integration_step, integration_step) # For trapezoid integration
            
            obs_time = np.logspace(1, 7, 7) 
            mcrab_lst = [1, 0.1]
            mdp_dict = {}
            
            once = True
            for mcrab in mcrab_lst:
                print(f"Computing x{mcrab} source Flux")
                for delta_t in tqdm(obs_time):
                    TOTAL_Source_Count_s = 0
                    for i, E in enumerate(Energy_lst):
                        if i == 0:
                            continue
                        bin_min = Energy_lst[i-1]
                        bin_max = Energy_lst[i]

                        Bin_Source_Count_s = mcrab * integrate_product(Optics_effArea, bin_min, bin_max, flux_func=Flux_MEGAlib_Source)
                        TOTAL_Source_Count_s = TOTAL_Source_Count_s + Bin_Source_Count_s

                    TOTAL_Background_Count_s = get_background_count_s(Emin, Emax, polar_angle=90)

                    Compton_Background_Count_s = TOTAL_Background_Count_s * Compton_abs_eff # background comptons cnt/s
                    Compton_Source_Count_s = TOTAL_Source_Count_s * Compton_abs_eff # Source comptons cnt/s

                    if mcrab == 1 and once == True:
                        once = False
                        PLOT_Flux_AfterOptics_int.append(TOTAL_Source_Count_s)
                        PLOT_Flux_AfterOptics_Compton.append(Compton_Source_Count_s)
                        PLOT_Flux_AfterOptics_Compton_Background.append(Compton_Background_Count_s)
                        E_center = (Emin + Emax)/2
                        PLOT_Emin.append(E_center - Emin)
                        PLOT_Emax.append(Emax - E_center)
                        PLOT_E.append((Emin+Emax)/2)

                    MDP = compute_MDP(Compton_Source_Count_s, Q, Compton_Background_Count_s, delta_t)
                    mdp_dict[mcrab, delta_t] = MDP

            global_mdp_dict[Emin, Emax] = mdp_dict
         
        plt.subplot(3,3,3)
        xerr = np.array([PLOT_Emin, PLOT_Emax])
        plt.errorbar(PLOT_E, PLOT_Flux_AfterOptics_int, xerr = xerr, capsize=6, label='Flux After Optics')
        plt.errorbar(PLOT_E, PLOT_Flux_AfterOptics_Compton, xerr = xerr, capsize=6, label='Compton After Optics')
        plt.errorbar(PLOT_E, PLOT_Flux_AfterOptics_Compton_Background, xerr = xerr, capsize=6, label='Background')
        plt.xlabel('Energy [keV]')
        plt.ylabel('Counts on detector [ph/s]')
        plt.yscale('log')


        plt.subplot(3,3,4)
        plt.errorbar(PLOT_E, PLOT_Q, xerr=xerr, yerr = PLOT_Q_Uncer, c='k', capsize=6)
        plt.xlabel('Energy [keV]')
        plt.ylabel(r'$Q_{100}$')

        plt.subplot(3,3,5)
        plt.errorbar(PLOT_E, PLOT_Compton_Eff, xerr=xerr, c='k', capsize=6)
        plt.xlabel('Energy [keV]')
        plt.ylabel('Absolute Compton Efficiency [0-1]')

        plt.tight_layout()
        plt.legend()
        plt.show()

    elif args.collimated:
        print("At this stage use only .dat sources, .xcm not implemented yet... .xcm only works for 1A0535 source")
        print(".dat file needs to have integrated flux over energy bin in units [ph/s/cm2]")
        source_file, source_name_full = select_source()
        print(f"\nSelected source: {source_file}")
        source_name = source_name_full.split('.')[0]
        source_file_format = source_name_full.split('.')[-1]
        
        if DEBUG:
            print(f"Source File Format: {source_file_format}")
            print(f"Source Name: {source_name}")
        
        # Asking for the binning of Energy for the MDP
        ask_bin_center = True
        MDP_Energy_BinCenter = []
        while ask_bin_center:
            print("Provide the Ebin center. Be sure to have values concurrent with ones analysed by Polarimetry.py. Write 'stop' to stop asking.")
            ebin = input("Ebin center: ")
            if ebin == 'stop':
                ask_bin_center = False
                continue
            else:
                MDP_Energy_BinCenter.append(int(ebin))
        # Binning the spectra properly
        energy_bin_edges, energy_bin_widths, energy_bin_centers = creat_polarimetry_bins(MDP_Energy_BinCenter)


        if source_file_format == 'xcm':
            ## Load the data here, just works for 1A0535 source now.... need to implement the ability to read fluxes with xspec
            model_expr, params = read_xcm_model(source_file)
            NH, Gamma, Ecut, K = params
            E = np.linspace(1, 1000, 2000)  # keV
            Flux_Source_XCM = lambda E: xspec_TBabs_cutoffpl(E, NH, Gamma, Ecut, K) # ph s-1 cm-2 kev-1
            plt.subplot(3,3,2)
            e_plot_step = 0.1 #keV
            E_plot = np.arange(Observatory_minE, Observatory_maxE + e_plot_step, e_plot_step)
            plt.plot(E_plot, Flux_Source_XCM(E_plot), c = 'k', label=f'{source_name} flux')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel(f'{source_name} Flux [ph/s/cm2/keV]')
            plt.xlabel('Energy [keV]')


        
        if source_file_format == 'dat':
            Source_Energy_BinCenter = []
            Flux_Source_Dat = []

            with open(source_file, 'r') as f:
                for line in f:
                    if 'E_min' in line:
                        continue
                    parts = line.split()
                    E_min = float(parts[0])
                    E_max = float(parts[1])
                    delta_E = E_max - E_min
                    E_center = (E_max + E_min)/2
                    flux_int = float(parts[2])  # already integrated over the bin, [ph/s/cm2]
                    flux_diff = flux_int * delta_E # [ph/s/cm2/kev]
                    
                    Source_Energy_BinCenter.append(E_center)
                    Flux_Source_Dat.append(flux_diff) # ph/s/cm/kev

            plt.subplot(3,3,2)
            plt.plot(Source_Energy_BinCenter, Flux_Source_Dat, c = 'k', label=f'{source_name} flux')
            plt.xscale('log')
            plt.yscale('log')
            plt.ylabel(f'{source_name} Flux [ph/s/cm2/keV]')
            plt.xlabel('Energy [keV]')



        obs_time = np.arange(1000, 1e6+1000,1000)
        global_mdp_dict = {}
        for i, bin_E in enumerate(MDP_Energy_BinCenter):
            Emin, Emax = energy_bin_edges[i]
            Emin = int(Emin)
            Emax = int(Emax)
            result_polarimetry_folder = f"{FOLDER_BASE_POLARIMETRY}/CollimatedBeam{bin_E}keV_config{HED_config}x{HED_config}_{distance_dets}cm/all/{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}" #using /all spectra range form polarimetry analysis
            n_events_Source = 1e6 # this is the value I use for generated photons on MEGAlib simulation for the CollimatedBeam.... need to find a way to make this consistent with the actual value used on MEGAlib simu, maybe parse the .source Megalib file???? idk
            Q, Q_uncer = compton.get_Q(result_polarimetry_folder, 'Both')  # MDP for both detectors together, for further analysis play around with mdp just on one, sperate, etc...
            n_comptons = compton.get_n_comptons(result_polarimetry_folder, 'Both')
            Compton_abs_eff = n_comptons/n_events_Source
            
            integration_step = 1 # keV
            Energy_lst = np.arange(Emin, Emax+integration_step, integration_step) # For trapezoid integration
            
            obs_time = np.logspace(1, 7, 7) 
            mcrab_lst = [1]
            mdp_dict = {}
            
            for mcrab in mcrab_lst:
                print(f"Computing x{mcrab} source Flux")
                for delta_t in tqdm(obs_time):
                    TOTAL_Source_Count_s = 0
                    for i, E in enumerate(Energy_lst):
                        if i == 0:
                            continue
                        bin_min = Energy_lst[i-1]
                        bin_max = Energy_lst[i]
                        if DEBUG:
                            print(f"\n[ENERGY BIN]")
                            print(f" File format selected: {source_file_format}")
                            print(f"  Range: {bin_min:.2f} - {bin_max:.2f} keV")

                        if source_file_format == 'xcm':
                            Bin_Source_Count_s = mcrab * integrate_product(Optics_effArea, bin_min, bin_max, flux_func = Flux_Source_XCM)
                            TOTAL_Source_Count_s = TOTAL_Source_Count_s + Bin_Source_Count_s
                            if DEBUG:
                                print(f"  Source (XCM) counts/s in bin: {Bin_Source_Count_s:.3e}")

                        if source_file_format == 'dat':
                            Bin_Source_Count_s = mcrab * integrate_product(Optics_effArea, bin_min, bin_max, source_diff_flux=Flux_Source_Dat, source_center_bin=Source_Energy_BinCenter)
                            TOTAL_Source_Count_s = TOTAL_Source_Count_s + Bin_Source_Count_s
                            if DEBUG:
                                print(f"  Source (DAT) counts/s in bin: {Bin_Source_Count_s:.3e}")
                        if DEBUG:
                            print(f"  Running TOTAL source counts/s: {TOTAL_Source_Count_s:.3e}")


                    TOTAL_Background_Count_s = get_background_count_s(Emin, Emax, polar_angle=90)
                    if DEBUG:
                        print(f"\n[BACKGROUND]")
                        print(f"  Energy range used: {Emin} - {Emax} keV")
                        print(f"  Total background counts/s: {TOTAL_Background_Count_s:.3e}")

                    Compton_Background_Count_s = TOTAL_Background_Count_s * Compton_abs_eff # background comptons cnt/s
                    Compton_Source_Count_s = TOTAL_Source_Count_s * Compton_abs_eff # Source comptons cnt/s

                    if DEBUG:
                        print(f"\n[COMPTON COUNTS]")
                        print(f"  Efficiency: {Compton_abs_eff:.3e}")
                        print(f"  Source Compton counts/s: {Compton_Source_Count_s:.3e}")
                        print(f"  Background Compton counts/s: {Compton_Background_Count_s:.3e}")
                        if Compton_Background_Count_s > 0:
                            ratio = Compton_Source_Count_s / Compton_Background_Count_s
                            print(f"  S/B ratio: {ratio:.3e}")


                    MDP = compute_MDP(Compton_Source_Count_s, Q, Compton_Background_Count_s, delta_t)
                    mdp_dict[mcrab, delta_t] = MDP
                    if DEBUG:
                        print(f"\n[MDP]")
                        print(f"[{Emin} - {Emax}] keV")
                        print(f"  Observation time: {delta_t:.2e} s")
                        print(f"  Q factor: {Q:.3f}")
                        print(f"  MDP: {MDP:.3e}")

            global_mdp_dict[Emin, Emax] = mdp_dict



     
    #plt.figure(figsize=(12, 8))
    
    # V0332
    #plot_MDP_Energy(mdp_dict, energy_bin_edges, obs_time, [1], source=f'{source_name}') 
    #breakpoint()
            
        
    mcrab_target = 1

    fig, ax = plt.subplots(figsize=(6,5))
    plt.title(f"Source {source_name}")


# number of energy bins (for colormap)
    n_bins = len(global_mdp_dict)
    #colors = cm.rainbow_r(np.linspace(0,1,n_bins))
    #hsv_colors = plt.cm.hsv(np.linspace(0, 1, 7))  # Red -> violet cycle
    #colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:gray']
    #colors = [
    #    "tab:gray",
    #    "#4c72b0",
    #    "#dd8452",
    #    "#55a868",
    #    "#c44e52",
    #    "#8172b3",
    #    "#937860",
    #    "#da8bc3",
    #]
    #markers = ['o', 's', '^', 'D', 'v', '*', 'p']
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

        print("MDPs:", mdps)
        print("Min MDP:", np.nanmin(mdps))
        print("Any <= 0:", np.any(mdps <= 0))
        print("Any NaN:", np.isnan(mdps).any())


        label = f"{Emin}-{Emax} keV"
        #ax.plot(times, mdps * 100, color=colors[i], alpha=0.8)
        ax.plot(times, mdps * 100, alpha=0.8)
        #ax.scatter(times, mdps * 100, marker=markers[i], color=colors[i], label=label, s=80)
        #ax.scatter(times, mdps * 100, color=colors[i], label=label, s=80)
        ax.scatter(times, mdps * 100, label=label, s=80)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Observation Time [s]", fontsize = 12)
    ax.set_ylabel(f"{mcrab_target}" + r" mFlux MDP$_{100}$ [%]", fontsize = 12)
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

            










    

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np

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


if __name__ == '__main__':

    E_step = 5
    Energy_lst = np.arange(15, 600 + E_step, E_step)
    
    # Binning the spectra properly
    energy_bin_edges = []
    energy_bin_widths = []
    lst_energy_centers = [] 
    lst_energy_widths = []  
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
    breakpoint()

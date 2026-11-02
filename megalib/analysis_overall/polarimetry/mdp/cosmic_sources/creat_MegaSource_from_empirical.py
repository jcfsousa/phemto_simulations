import numpy as np
from scipy.interpolate import interp1d

def get_flux_crab():
    '''
    arXiv:astro-ph/0406058v1 2 Jun 2004
    F(E) = k (E/1kev)^(-alpha) ph/cm2/s/keV

    Theoretically this model is from >50kev but we use it from 15keV........

    '''
    K = 14.44
    alpha = 2.169
    return lambda E: K * (E)**(-alpha)

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
        fluxes.append(crab_flux_high_energy(E, mcrab))
    
    integral = np.trapezoid(fluxes, energies)
    return integral  # ph/cm2/s

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


Energy_lst = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 80, 100, 125, 150, 175, 200, 250, 300, 350, 400, 450, 500, 550, 600]


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

Flux_MEGAlib_Source = get_flux_crab() #ph/s/cm2/kev 
# to write megalib sources
with open("crab_optics_spectrum.dat", "w") as f:

    f.write("IP LINLIN\n")
    for i,E in enumerate(Energy_lst):
        E_min, E_max = energy_bin_edges[i]
        #total_area = integrate_area_from_dict(area_dict, E_min, E_max)/(E_max-E_min)
        #integrated_flux_crab = integrated_crab_flux(E_min, E_max)
        #photons_per_second = integrated_flux_crab * total_area
        photons_per_second = integrate_product(area_dict, E_min, E_max, flux_func=Flux_MEGAlib_Source) #cnts/s
        dNdE = photons_per_second/(E_max-E_min) #cnts/s/kev
        f.write(f"DP {E} {dNdE}\n")
    f.write("EN")

# to write total flux for megalib
total_flux = 0
for i,E in enumerate(Energy_lst):
    E_min,E_max = energy_bin_edges[i]
    #total_area = integrate_area_from_dict(area_dict,E_min,E_max)/(E_max-E_min)
    #integrated_flux_crab = integrated_crab_flux(E_min,E_max)
    #photons_per_second = integrated_flux_crab * total_area
    photons_per_second = integrate_product(area_dict, E_min, E_max, flux_func=Flux_MEGAlib_Source) #cnts/s
    total_flux += photons_per_second
print(f"Total flux for MEGAlib: {total_flux}")

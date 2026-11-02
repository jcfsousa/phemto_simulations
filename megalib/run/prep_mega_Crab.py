import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


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

## ------------------------------------------------------------
# Change Parameters

    CdTe_matrix_size = [3]
    dists = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
    dists = [1]
    instruments_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/instruments'
    output_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/simTra_files'
    sources_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources'

    config_lst = []
    for matrix_size in CdTe_matrix_size:
        for dist in dists:
            config = f"config{matrix_size}x{matrix_size}_{dist}cm"
            config_lst.append(config)

# Energy list
#Log_E=[4,8,15,30,50,80,100,120,150,200,250,300,350,400,500,600,700]
    SourceName = 'Crab10Mevents'
    n_events = 10000000
    Emin = 15
    Emax = 600

#revan_config = 'compton.revan.cfg'
    revan_config = 'comptons_klein-abs-comptEne.cfg'

    pol = input('Polarized, non-polarized or both (pol, non, both)')

## ------------------------------------------------------------



## ------------------------------------------------------------
# PREPING THE CRAB SOURCE ACCORDING TO OPTICS EFF AREA
    path_source_crab_optics = '../sources/spectrum/PHEMTO_crab_optics_spectrum.dat'
    E_step = 5 # kev, for integration and produce .dat
    Energy_lst = np.arange(Emin, Emax + E_step, E_step)

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


# to write megalib sources
    with open(f"{path_source_crab_optics}", "w") as f:

        f.write("IP LINLIN\n")
        for i,E in enumerate(Energy_lst):
            E_min, E_max = energy_bin_edges[i]
            total_area = integrate_area_from_dict(area_dict, E_min, E_max)/(E_max-E_min)
            integrated_flux_crab = integrated_crab_flux(E_min, E_max)
            photons_per_second = integrated_flux_crab * total_area
            dNdE = photons_per_second/(E_max-E_min) #ph/s/kev
            f.write(f"DP {E} {dNdE}\n")
        f.write("EN")

# to write total flux for megalib
    total_flux_crab = 0
    lst_total_flux_crab = []
    for i,E in enumerate(Energy_lst):
        E_min,E_max = energy_bin_edges[i]
        total_area = integrate_area_from_dict(area_dict,E_min,E_max)/(E_max-E_min)
        integrated_flux_crab = integrated_crab_flux(E_min,E_max)
        photons_per_second = integrated_flux_crab * total_area
        lst_total_flux_crab.append(photons_per_second)
        total_flux_crab += photons_per_second
    print(f"Total flux for MEGAlib: {total_flux_crab} ph/s")

    # Debug plot Area and flux
    fig, ax1 = plt.subplots()
    # LEFT Y AXIS — Flux
    ax1.plot(Energy_lst, lst_total_flux_crab, c='k',label='Crab Flux at Detector')
    ax1.set_xlabel("Energy (keV)")
    ax1.set_ylabel("Flux (ph/s)")
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # RIGHT Y AXIS — Effective area
    ax2 = ax1.twinx()
    ax2.plot(energies, areas, c='b', linestyle='--', label='Optics Eff Area')
    ax2.set_ylabel("Effective Area (cm2)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2)

    plt.title("Crab Flux at Detector vs Optics Effective Area")
    plt.tight_layout()
    plt.show()


## ------------------------------------------------------------

    for config in config_lst:
        geofile=f'{instruments_path}/PHEMTO_{config}.geo.setup'

        with open(f"./runCosima{config}.sh", mode='w') as f:

            if pol == 'pol' or pol == 'both':
                # Polarized simulation
                string1= f"""# An example run for Cosima \n +
                 \nVersion          1  
                 \nGeometry         {geofile} 
                 \nPhysicsListEM    LivermorePol   // Allow polarization \n
                 \n# Output formats\n
                 \nStoreSimulationInfo       all
                 \nStoreCalibrated   true
                 \nStoreSimulationInfoIonization true
                 \nDiscretizeHits    true\n
                 \nRun {SourceName}Pol \n  // (laue, xray focus), mono for Q100
                 {SourceName}Pol.FileName              {output_path}/{SourceName}Pol_{Emin}-{Emax}keV_{config} 
                 \n{SourceName}Pol.NEvents               {n_events}\n\n
                 \n{SourceName}Pol.Source One 
                 \nOne.ParticleType        1 
                 \nOne.Beam                HomogeneousBeam  0 0 10 0 0 -1 0.075  // psf diameter = 1.5mm
                 \nOne.Spectrum            File {path_source_crab_optics} 
                 \nOne.Flux                {total_flux_crab}
                 \nOne.Polarization RelativeX 1.0 90"""

                # Polarized Source prepared
                source_file1=f'{sources_path}/{SourceName}Pol_{Emin}-{Emax}keV_{config}.source'
                sf1=open(source_file1,'w')
                sf1.write(string1)
                sf1.close()
                # For Cosima
                runCode1=f'{sources_path}/{SourceName}Pol_{Emin}-{Emax}keV_{config}.source'
                f.write(f"mdelay cosima 22; cosima -z -v 0 {runCode1} >> /dev/null & sleep 1;")
            
            if pol == 'non' or pol == 'both':
                ## Non-Polarized simulation
                string2= f"""# An example run for Cosima \n 
                 \nVersion          1  
                 \nGeometry         {geofile}
                 \nPhysicsListEM    LivermorePol   // Allow polarization \n
                 \n# Output formats\n
                 \nStoreSimulationInfo       all
                 \nStoreCalibrated   true
                 \nStoreSimulationInfoIonization true
                 \nDiscretizeHits    true\n
                 \nRun {SourceName}NonPol 
                 \n{SourceName}NonPol.FileName              {output_path}/{SourceName}NonPol_{Emin}-{Emax}keV_{config} 
                 \n{SourceName}NonPol.NEvents               {n_events}
                 \n#{SourceName}NonPol.Time                   100000\n\n
                 \n{SourceName}NonPol.Source One 
                 \nOne.ParticleType        1 
                 \n
                 \nOne.Beam                HomogeneousBeam  0 0 10 0 0 -1 0.075 // psf diameter = 1.5mm 
                 \nOne.Spectrum            File {path_source_crab_optics} 
                 \nOne.Flux                {total_flux_crab}
                 \nOne.Polarization Random """

                # Non-Polarized Source prepared
                source_file2=f'{sources_path}/{SourceName}NonPol_{Emin}-{Emax}keV_{config}.source'
                sf2=open(source_file2,'w')
                sf2.write(string2)
                sf2.close()
                # For Cosima
                runCode2=f'{sources_path}/{SourceName}NonPol_{Emin}-{Emax}keV_{config}.source'
                f.write(f"mdelay cosima 22; cosima -z -v 0 {runCode2} >> /dev/null  & sleep 1;")
            f.write('\necho "Waiting for Cosima to run...."')
            f.write('\nwait')


# Prep Revan
    for config in config_lst:
        geofile=f'{instruments_path}/PHEMTO_{config}.geo.setup'
        with open(f"./runRevan{config}.sh", mode='w') as f:
            if pol == 'pol' or pol == 'both':
                source_file1=f'{output_path}/{SourceName}Pol_{Emin}-{Emax}keV_{config}'
                f.write(f"revan -c {revan_config} -a -n -f {source_file1}.inc1.id1.sim.gz -g {geofile} \n")
            if pol == 'non' or pol == 'both':
                source_file2=f'{output_path}/{SourceName}NonPol_{Emin}-{Emax}keV_{config}'
                f.write(f"revan -c {revan_config} -a -n -f {source_file2}.inc1.id1.sim.gz -g {geofile} \n")



# Prep Run
    with open(f"./runAll.sh", mode='w') as f:
        for config in config_lst:
            f.write(f'bash ./runCosima{config}.sh && echo "Cosima ended... Starting Revan..." && bash ./runRevan{config}.sh \n')
        f.close()

    print('Run ./runAll.sh to run Cosima and Revan')

#with open(f"./runAll.sh", mode='w') as f:
#    for config in config_lst:
#        f.write(f'bash ./runRevan{config}.sh \n')
#
#    f.close()



# Prep Polarimetry
    output_parser = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry_data'
    output_polarimetry = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/polarimetry_results'
    base_path_analysis_overall = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall'

    with open(f"./runPolarimetry.sh", mode='w') as f:
        f.write(f'source {base_path_analysis_overall}/env-phemto/bin/activate ;') # Activate python3 virtual environment
        for config in config_lst:
                ###### Parsing .tra -> .t3pa ######
                # Polarized source
                tra_gz_filePathPol=f'{output_path}/{SourceName}Pol_{Emin}-{Emax}keV_{config}.inc1.id1.tra.gz'
                runPol_name = f"{SourceName}Pol_{Emin}-{Emax}keV_{config}"
                output_path_configPol = f"{output_parser}/{runPol_name}"
                f.write(f'python3 {base_path_analysis_overall}/parse.py -f {tra_gz_filePathPol} -o {output_path_configPol}; ') #Polarized Source
                # Non-Polarized source
                tra_gz_filePathNonPol=f'{output_path}/{SourceName}NonPol_{Emin}-{Emax}keV_{config}.inc1.id1.tra.gz'
                runNonPol_name = f"{SourceName}NonPol_{Emin}-{Emax}keV_{config}"
                output_path_configNonPol = f"{output_parser}/{runNonPol_name}"
                f.write(f'python3 {base_path_analysis_overall}/parse.py -f {tra_gz_filePathNonPol} -o {output_path_configNonPol};') # Non-Polarized Source


                ###### Pre_process ######
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/pre_process.py -i {output_path_configNonPol};') # Pre_process, single, double, mult, 
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/pre_process.py -i {output_path_configPol};') # Pre_process, single, double, mult, 
                ###### SpecMaker ######
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/specMaker.py -i {output_path_configNonPol} -o {output_path_configNonPol} -l;')
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/specMaker.py -i {output_path_configPol} -o {output_path_configPol} -l;')



                ###### Polarimetry Analysis ######
                ## perform polarimetry per energy bin
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/polarimetry.py -o {output_polarimetry} -ip {output_path_configPol} -inp {output_path_configNonPol} -b 50-400;')
                ## From 15-600keV:
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/polarimetry.py -o {output_polarimetry} -ip {output_path_configPol} -inp {output_path_configNonPol} -b {Emin}-{Emax};')
                # From 15-50keV:
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/polarimetry.py -o {output_polarimetry} -ip {output_path_configPol} -inp {output_path_configNonPol} -b 15-50;')
                # From 50-150keV:
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/polarimetry.py -o {output_polarimetry} -ip {output_path_configPol} -inp {output_path_configNonPol} -b 50-150 ;')
                # From 150-250keV:
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/polarimetry.py -o {output_polarimetry} -ip {output_path_configPol} -inp {output_path_configNonPol} -b 150-250 ;')
                # From 250-350keV:
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/polarimetry.py -o {output_polarimetry} -ip {output_path_configPol} -inp {output_path_configNonPol} -b 250-350 ;')
                # From 350-450keV
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/polarimetry.py -o {output_polarimetry} -ip {output_path_configPol} -inp {output_path_configNonPol} -b 350-450 ;')
                # From 450-600keV
                f.write(f'python3 {base_path_analysis_overall}/polarimetry/polarimetry.py -o {output_polarimetry} -ip {output_path_configPol} -inp {output_path_configNonPol} -b 450-600 ;')
        f.close()

    print('Run ./runPolarimetry.sh to run .tra.gz -> .t3pa parser, SpecMaker and Polarimetry analysis.')



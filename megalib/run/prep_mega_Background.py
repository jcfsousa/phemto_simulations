import numpy as np 
import sys

def integrate_flux(flux_dict, E_min, E_max, solid_angle):
    
    filtered_dict_energies = {k: v for k, v in flux_dict.items() if E_min <= float(k) <= E_max}
    
    #if len(filtered_dict_energies) < 2:
    #    raise ValueError("Not enough data points in the specified energy range.")
    
    energies = np.array(sorted(filtered_dict_energies.keys()))
    fluxes = np.array([filtered_dict_energies[e] for e in energies])

    integrated_flux = np.trapezoid(fluxes, energies) 

    return integrated_flux * solid_angle # cnts cm-2 s-1

Emin = 1.6
Emax = 400

# cm-2 s-1 kev-1 sr-1
cosmic_background_file = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/background/Data/CosmicPhotons_Spec_600.0km_5.0deg_1100.0solarmod.dat'

background_dict = {}
with open(cosmic_background_file, 'r') as f:
    for line in f:
        stripped = line.strip()
        if not stripped or '#' in stripped or 'IP' in stripped or 'EN' in stripped:
            continue

        parts = stripped.split()
        if len(parts) >= 3 and parts[0] == 'DP':
            energy = float(parts[1])  #keV
            flux = float(parts[2]) # cm-2 s-1 kev-1 sr-1
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
total_flux_cosmic_background = integrate_flux(background_dict, Emin, Emax, solid_angle) #ph/cm2/s
print(f'Total BACKGROUND integrated flux from {Emin} keV to {Emax} keV {solid_angle} sr: {total_flux_cosmic_background:.20e} ph/s/cm^2')


## ------------------------------------------------------------
# Change Parameters

CdTe_matrix_size = [4]
dists = [1.5]
instruments_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/instruments'
output_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/simTra_files'
sources_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources'

SourceName = 'Background10ksec'

config_lst = []
for matrix_size in CdTe_matrix_size:
    for dist in dists:
        config = f"config{matrix_size}x{matrix_size}_{dist}cm"
        config_lst.append(config)

# Energy list
#Log_E=[4,8,15,30,50,80,100,120,150,200,250,300,350,400,500,600,700]

revan_config = 'comptons_klein-abs-comptEne.cfg'

## ------------------------------------------------------------

for config in config_lst:
    geofile=f'{instruments_path}/PHEMTO_collimator_{config}.geo.setup'

    with open(f"./runCosima{config}.sh", mode='w') as f:

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
         \nRun {SourceName} \n  // Gauss (laue, xray focus), mono for Q100
         {SourceName}.FileName              {output_path}/{SourceName}_{Emin}-{Emax}keV_{config} 
         \n{SourceName}.Time               10000\n\n
         \n{SourceName}.Source One 
         \nOne.ParticleType        1 
         \nOne.Beam                FarFieldAreaSource 0 90 0 360  // 
         \nOne.Spectrum            File {cosmic_background_file}
         \nOne.Flux                {total_flux_cosmic_background}"""
            
        # Polarized Source prepared
        source_file1=f'{sources_path}/{SourceName}_{Emin}-{Emax}keV_{config}.source'
        sf1=open(source_file1,'w')
        sf1.write(string1)
        sf1.close()
        # For Cosima
        runCode1=f'{sources_path}/{SourceName}_{Emin}-{Emax}keV_{config}.source'
        #f.write(f"mdelay cosima 22; cosima -z -v 0 {runCode1} >> /dev/null & sleep 1;")
        f.write(f"cosima -z {runCode1}")
        
        #f.write('\necho "Waiting for Cosima to run...."')
        #f.write('\nwait')


# Prep Revan
for config in config_lst:
    geofile=f'{instruments_path}/PHEMTO_collimator_{config}.geo.setup'
    with open(f"./runRevan{config}.sh", mode='w') as f:
        source_file2=f'{output_path}/{SourceName}_{Emin}-{Emax}keV_{config}'
        f.write(f"revan -c {revan_config} -a -n -f {source_file2}.inc1.id1.sim.gz -g {geofile} \n")



# Prep Run
with open(f"./runAll.sh", mode='w') as f:
    for config in config_lst:
        f.write(f'bash ./runCosima{config}.sh && echo "Cosima ended... Starting Revan..." && bash ./runRevan{config}.sh \n')
    f.close()

print('Run ./runAll.sh to run Cosima and Revan')


# Prep Parse for event effs
output_parser = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/background_data'
output_polarimetry = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/background_results'
base_path_analysis_overall = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall'

with open(f"./runParser.sh", mode='w') as f:
    f.write(f'source {base_path_analysis_overall}/env-phemto/bin/activate ;') # Activate python3 virtual environment
    for config in config_lst:
            ###### Parsing .tra -> .t3pa / Also countes events to compute bkg ph/s/kev ######
            # Background
            tra_gz_filePathPol=f'{output_path}/{SourceName}_{Emin}-{Emax}keV_{config}.inc1.id1.tra.gz'
            runPol_name = f"{SourceName}_{Emin}-{Emax}keV_{config}"
            output_path_configPol = f"{output_parser}/{runPol_name}"
            f.write(f'python3 {base_path_analysis_overall}/parse.py -f {tra_gz_filePathPol} -o {output_path_configPol} -p 0.075; ') # -p 0.075, 0.075cm PSF radius

print('Run ./runParser.sh to run .tra.gz -> .t3pa parser')

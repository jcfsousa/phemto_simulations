## ------------------------------------------------------------
# Change Parameters

CdTe_matrix_size = [4]
dists = [1.5]
instruments_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/instruments'
output_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/simTra_files'
sources_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources'
SourceName = 'CollimatedBeam'

config_lst = []
for matrix_size in CdTe_matrix_size:
    for dist in dists:
        config = f"config{matrix_size}x{matrix_size}_{dist}cm"
        config_lst.append(config)

# Energy list
#Log_E=[4,8,15,30,50,80,100,120,150,200,250,300,350,400,500,600,700]
Log_E=[4, 10, 30, 50, 100, 150, 200, 250, 300, 350, 400]

revan_config = ['EffectiveArea.revan.cfg', 'revan.cfg', 'Test.revan.cfg']

pol = input('Polarized, non-polarized or both (pol, non, both)')

## ------------------------------------------------------------

for config in config_lst:
    geofile=f'{instruments_path}/PHEMTO_{config}.geo.setup'

    with open(f"./runCosima{config}.sh", mode='w') as f:

        for myene in Log_E:
            
            if pol == 'pol' or pol == 'both':
                # Polarized simulation
                string1= f"""# An example run for Cosima \n +
                 \nVersion          1  
                 \nGeometry         {geofile} 
                 \nPhysicsListEM    LivermorePol   // Allow polarization \n
                 \n# Output formats\n
                 StoreSimulationInfo       all\n
                 StoreCalibrated   true\n
                 StoreSimulationInfoIonization true\n
                 DiscretizeHits    true\n
                 \nRun {SourceName}Pol \n  // Gauss (laue, xray focus), mono for Q100
                 {SourceName}Pol.FileName              {output_path}/{SourceName}Pol{myene}keV_{config} \n
                 {SourceName}Pol.NEvents               1000000\n\n\n
                 {SourceName}Pol.Source One \n
                 One.ParticleType        1 \n
                 One.Beam                HomogeneousBeam  0 0 20 0 0 -1 0.075  // psf diameter = 1.5mm\n
                 One.Spectrum            Mono  {myene}\n
                 One.Flux                1\n 
                 One.Polarization RelativeX 1.0 90"""

                # Polarized Source prepared
                source_file1=f'{sources_path}/{SourceName}Pol{myene}keV_{config}.source'
                sf1=open(source_file1,'w')
                sf1.write(string1)
                sf1.close()
                # For Cosima
                runCode1=f'{sources_path}/{SourceName}Pol{myene}keV_{config}.source'
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
                 \n{SourceName}NonPol.FileName              {output_path}/{SourceName}NonPol{myene}keV_{config} 
                 \n{SourceName}NonPol.NEvents               1000000
                 \n{SourceName}NonPol.Source One 
                 \nOne.ParticleType        1 
                 \n
                 \nOne.Beam                HomogeneousBeam  0 0 20 0 0 -1 0.075 // psf diameter = 1.5mm 
                 \nOne.Spectrum            Mono  {myene}
                 \nOne.Flux                1
                 \nOne.Polarization Random """

                # Non-Polarized Source prepared
                source_file2=f'{sources_path}/{SourceName}NonPol{myene}keV_{config}.source'
                sf2=open(source_file2,'w')
                sf2.write(string2)
                sf2.close()
                # For Cosima
                runCode2=f'{sources_path}/{SourceName}NonPol{myene}keV_{config}.source'
                f.write(f"mdelay cosima 22; cosima -z -v 0 {runCode2} >> /dev/null  & sleep 1;")
        f.write('\necho "Waiting for Cosima to run...."')
        f.write('\nwait')


# Prep Revan
for config in config_lst:
    geofile=f'{instruments_path}/PHEMTO_{config}.geo.setup'
    with open(f"./runRevan{config}.sh", mode='w') as f:
        for myene in Log_E:
            if pol == 'pol' or pol == 'both':
                source_file1=f'{output_path}/{SourceName}Pol{myene}keV_{config}'
                f.write(f"revan -c revan.cfg -a -n -f {source_file1}.inc1.id1.sim.gz -g {geofile} \n")
            if pol == 'non' or pol == 'both':
                source_file2=f'{output_path}/{SourceName}NonPol{myene}keV_{config}'
                f.write(f"revan -c revan.cfg -a -n -f {source_file2}.inc1.id1.sim.gz -g {geofile} \n")



# Prep Run
with open(f"./runAll.sh", mode='w') as f:
    for config in config_lst:
        f.write(f'bash ./runCosima{config}.sh && echo "Cosima ended... Starting Revan..." && bash ./runRevan{config}.sh \n')
    f.close()

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
            for myene in Log_E:
                    ###### Parsing .tra -> .t3pa ######
                    # Polarized source
                    tra_gz_filePathPol=f'{output_path}/{SourceName}Pol{myene}keV_{config}.inc1.id1.tra.gz'
                    runPol_name = f"{SourceName}Pol{myene}keV_{config}"
                    output_path_configPol = f"{output_parser}/{runPol_name}"
                    f.write(f'python3 {base_path_analysis_overall}/parse.py -f {tra_gz_filePathPol} -o {output_path_configPol}; ') #Polarized Source
                    # Non-Polarized source
                    tra_gz_filePathNonPol=f'{output_path}/{SourceName}NonPol{myene}keV_{config}.inc1.id1.tra.gz'
                    runNonPol_name = f"{SourceName}NonPol{myene}keV_{config}"
                    output_path_configNonPol = f"{output_parser}/{runNonPol_name}"
                    f.write(f'python3 {base_path_analysis_overall}/parse.py -f {tra_gz_filePathNonPol} -o {output_path_configNonPol};') # Non-Polarized Source


                    ###### Polarimetry Analysis ######
                    f.write(f'python3 {base_path_analysis_overall}/polarimetry/polarimetry.py -o {output_polarimetry} -ip {output_path_configPol} -inp {output_path_configNonPol} ;')
    f.close()




def float_range(start, stop, step):
    x = start
    while x < stop - 1e-9:
        yield round(x, 10)
        x += step
## ------------------------------------------------------------
# Change Parameters

CdTe_matrix_size = [4, 5, 6, 7, 8, 9]
dists = list(float_range(0.5, 10.5, 1))
instruments_path = '/local/home/jf285468/Documents/phemto_simulations/megalib/instruments'
config_lst = []
for matrix_size in CdTe_matrix_size:
    for dist in dists:
        config = f"config{matrix_size}x{matrix_size}_{dist}cm"
        config_lst.append(config)

# Energy list
#Log_E=[4,8,15,30,50,80,100,120,150,200,250,300,350,400,500,600,700]
Log_E=[50,100,150,200,250,300,350,400,500,600,700]

## ------------------------------------------------------------

for config in config_lst:
    geofile=f'{instruments_path}/PHEMTO_{config}.geo.setup'

    with open(f"./runCosima{config}.sh", mode='w') as f:
       for myene in Log_E:

             ## Polarized simulation
             string1= """# An example run for Cosima \n +
             \nVersion          1  
             \nGeometry         %s // Update this to your path  \n
             PhysicsListEM    LivermorePol   // Allow polarization \n
             \n# Output formats\n
             StoreSimulationInfo       all\n
             StoreCalibrated   true\n
             StoreSimulationInfoIonization true\n
             DiscretizeHits    true\n
             \nRun GaussBeamPol \n  // Gauss (laue, xray focus), mono for Q100
             GaussBeamPol.FileName              /media/jf285468/SAUVEGARDES/phemto_simulations/simTra_files/GaussBeamPol%dkeV_%s \n
             GaussBeamPol.NEvents               1000000\n\n\n
             GaussBeamPol.Source One \n
             One.ParticleType        1 \n
             One.Beam                GaussianConeBeam  0 0 20.025 0 0 -1 0.3638 0.18  // FWHM=0.15cm at z=0.025 cm - surface of Si \n
             One.Spectrum            Mono  %i\n
             One.Flux                1\n 
             One.Polarization RelativeX 1.0 90"""%(geofile, myene, config, myene)

             # Polarized Source prepared
             source_file1='../sources/GaussBeamPol%dkeV_%s.source'%(myene, config)
             sf1=open(source_file1,'w')
             sf1.write(string1)
             sf1.close()
             # For Cosima
             runCode1='../sources/GaussBeamPol%dkeV_%s.source'%(myene, config)
             f.write("cosima -z {};".format(runCode1))
             
             ## Non-Polarized simulation
             string2= """# An example run for Cosima \n 
             \nVersion          1  
             \nGeometry         %s // Update this to your path  \n
             PhysicsListEM    LivermorePol   // Allow polarization \n
             \n# Output formats\n
             StoreSimulationInfo       all\n
             StoreCalibrated   true\n
             StoreSimulationInfoIonization true\n
             DiscretizeHits    true\n
             \nRun GaussBeamPol // Gauss (laue, xray focus), mono for Q100 \n
             GaussBeamPol.FileName              /media/jf285468/SAUVEGARDES/phemto_simulations/simTra_files/GaussBeamNonPol%dkeV_%s \n
             GaussBeamPol.NEvents               1000000\n\n\n
             GaussBeamPol.Source One \n
             One.ParticleType        1 \n
             One.Beam                GaussianConeBeam  0 0 20.025 0 0 -1 0.3638 0.18 // FWHM=0.15cm at z=0.025 cm - surface of Si \n
             One.Spectrum            Mono  %i\n
             One.Flux                1\n 
             One.Polarization Random """%(geofile, myene, config, myene)

             # Non-Polarized Source prepared
             source_file2='../sources/GaussBeamNonPol%dkeV_%s.source'%(myene, config)
             sf2=open(source_file2,'w')
             sf2.write(string2)
             sf2.close()
             # For Cosima
             runCode2='../sources/GaussBeamNonPol%dkeV_%s.source'%(myene, config)
             f.write("cosima -z {};".format(runCode2))


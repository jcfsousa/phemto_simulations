## ------------------------------------------------------------
# Change Parameters

config_list = ['config1', 'config2', 'config3']

for config in config_list:
    geofile=f'/local/home/jf285468/Documents/PHEMTO/new/instruments/PHEMTO_{config}.geo.setup'
    #Log_E=[1,4,8,30,50,80,100,120,150,200,250,300,400,500,600]
    E_init = 50
    Log_E=[]
    while E_init <= 693.5:
        Log_E.append(E_init)
        E_init = E_init + 6.5
## ------------------------------------------------------------

    with open(f"./runCosima{config}.sh", mode='w') as f:
       for myene in Log_E:
             
             string1= "# An example run for Cosima \n\nVersion          1 \nGeometry         %s // Update this to your path  \nPhysicsListEM    LivermorePol \n\n# Output formats\nStoreSimulationInfo       all\nStoreCalibrated   true\nStoreSimulationInfoIonization true\nDiscretizeHits    true\n\nRun MyHomogeneousBeam \nMyHomogeneousBeam.FileName              ../sources/simTra_files/HomogeneousBeam%dkeV_%s \nMyHomogeneousBeam.NEvents               1000000\n\n\nMyHomogeneousBeam.Source One \nOne.ParticleType        1 \nOne.Beam                HomogeneousBeam  0 0 20 0 0 -1 0.075  \nOne.Spectrum            Mono  %i\nOne.Flux                1\n"%(geofile, myene, config, myene)
             source_file1='../sources/HomogeneousBeam%dkeV_%s.source'%(myene, config)
             sf1=open(source_file1,'w')
             sf1.write(string1)
             sf1.close()
             
             
             runCode1='../sources/HomogeneousBeam%dkeV_%s.source'%(myene, config)
             f.write("cosima -z {};".format(runCode1))
             
             



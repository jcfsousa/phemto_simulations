## ------------------------------------------------------------
# Change Parameters
geofile='/local/home/jf285468/Documents/PHEMTO/new/instruments/PHEMTO_config1.geo.setup'
Log_E=[8,30,50,80,100,120,150,200,250,300,400,500,600]

## ------------------------------------------------------------

geofile_name = geofile.split('/')[-1]
geofile_onlyname = geofile_name.split('.')[0]

with open("./runCosima.sh", mode='w') as f:
   for myene in Log_E:
         
         string1= "# An example run for Cosima \n\nVersion          1 \nGeometry         %s // Update this to your path  \nPhysicsListEM    LivermorePol \n\n# Output formats\nStoreSimulationInfo       all\nStoreCalibrated   true\nStoreSimulationInfoIonization true\nDiscretizeHits    true\n\nRun MyHomogeneousBeam \nMyHomogeneousBeam.FileName              ./sources/simTra_files/HomogeneousBeam%dkeV_%s \nMyHomogeneousBeam.NEvents               1000000\n\n\nMyHomogeneousBeam.Source One \nOne.ParticleType        1 \nOne.Beam                HomogeneousBeam  0 0 20 0 0 -1 0.075  \nOne.Spectrum            Mono  %i\nOne.Flux                1\n"%(geofile, myene, geofile_onlyname, myene)
         source_file1='sources/HomogeneousBeam%dkeV_%s.source'%(myene,geofile_onlyname)
         sf1=open(source_file1,'w')
         sf1.write(string1)
         sf1.close()
         
         
         runCode1='sources/HomogeneousBeam%dkeV_%s.source'%(myene,geofile_onlyname)
         f.write("cosima -z {};".format(runCode1))
         
         



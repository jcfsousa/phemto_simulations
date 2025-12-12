## ------------------------------------------------------------
# Change Parameters
geofile='/mnt/c/Users/utente/OneDrive/Work/Ferrara/ASTENAMegalib/Geometries/NFT_Pathfinder_4L.geo.setup'
Log_E=[80,100,120,150,200,250,300,400,500,600]

## ------------------------------------------------------------


with open("./runCosima.sh", mode='w') as f:
   for myene in Log_E:
         
         string1= "# An example run for Cosima \n\nVersion          1 \nGeometry         %s // Update this to your path  \nPhysicsListEM    LivermorePol \n\n# Output formats\nStoreSimulationInfo       all\n\nRun FFPS \nFFPS.FileName              Pol%dkeV \nFFPS.NEvents               1000000 \n\n\nFFPS.Source One \nOne.ParticleType        1 \nOne.Beam                GaussianConeBeam  0 0 200 0 0 -1 0.0365 0.0183  \nOne.Spectrum            Mono  %i\nOne.Flux                1\nOne.Polarization RelativeX 1.0 90 "%(geofile, myene, myene)
         source_file1='Pol%dkeV.source'%(myene)
         sf1=open(source_file1,'w')
         sf1.write(string1)
         sf1.close()
         
         string2= "# An example run for Cosima \n\nVersion          1 \nGeometry         %s // Update this to your path  \nPhysicsListEM    LivermorePol \n\n# Output formats\nStoreSimulationInfo       all\n\nRun FFPS \nFFPS.FileName              NoPol%dkeV \nFFPS.NEvents               1000000 \n\n\nFFPS.Source One \nOne.ParticleType        1 \nOne.Beam                GaussianConeBeam  0 0 200 0 0 -1 0.0365 0.0183 \nOne.Spectrum            Mono  %i\nOne.Flux                1\nOne.Polarization   Random "%(geofile, myene, myene)
         source_file2='NoPol%dkeV.source'%(myene)
         sf2=open(source_file2,'w')
         sf2.write(string2)
         sf2.close()
         
         runCode1='Pol%dkeV.source'%(myene)
         f.write("cosima -z {};".format(runCode1))
         runCode2='NoPol%dkeV.source'%(myene)
         f.write("cosima -z {};".format(runCode2))
         
         



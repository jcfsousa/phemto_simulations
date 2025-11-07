

## ------------------------------------------------------------
# Change Parameters
geofile='/local/home/jf285468/Documents/PHEMTO/new/PHEMTO_config1.geo.setup'
Log_E=[80,100,120,150,200,250,300,400,500,600]
## ------------------------------------------------------------



with open("./runRevan.sh", mode='w') as f:
    for myene in Log_E:

            source_file1='Pol%dkeV'%(myene)
            f.write("revan -a -n -f {}.inc1.id1.sim.gz -g {} -c revan.cfg\n".format(source_file1,geofile))
            source_file2='NoPol%dkeV'%(myene)
            f.write("revan -a -n -f {}.inc1.id1.sim.gz -g {} -c revan.cfg\n".format(source_file2,geofile))



## ------------------------------------------------------------
# Change Parameters
geofile='/local/home/jf285468/Documents/PHEMTO/new/instruments/PHEMTO_config1.geo.setup'
Log_E=[8,30,50,80,100,120,150,200,250,300,400,500,600]
## ------------------------------------------------------------

geofile_name = geofile.split('/')[-1]
geofile_onlyname = geofile_name.split('.')[0]

with open("./runRevan.sh", mode='w') as f:
    for myene in Log_E:

            source_file1='./sources/simTra_files/HomogeneousBeam%dkeV_%s'%(myene, geofile_onlyname)
            f.write("revan -a -n -f {}.inc2.id1.sim.gz -g {} -c revan.cfg\n".format(source_file1,geofile))

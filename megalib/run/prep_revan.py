

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

    with open(f"./runRevan{config}.sh", mode='w') as f:
        for myene in Log_E:

                source_file1='../sources/simTra_files/HomogeneousBeam%dkeV_%s'%(myene, config)
                f.write("revan -a -n -f {}.inc1.id1.sim.gz -g {} -c revan.cfg\n".format(source_file1,geofile))

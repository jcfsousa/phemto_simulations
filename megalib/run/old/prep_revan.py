def float_range(start, stop, step):
    x = start
    while x < stop - 1e-9:
        yield round(x, 10)
        x += step
## ------------------------------------------------------------
# Change Parameters

CdTe_matrix_size = [4, 5, 6, 7, 8, 9] # consistent with geometries
dists = list(float_range(0.5, 10.5, 1)) # consistent with geometries
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
# Change Parameters

for config in config_lst:
    geofile=f'{instruments_path}/PHEMTO_{config}.geo.setup'

    with open(f"./runRevan{config}.sh", mode='w') as f:
        for myene in Log_E:

                source_file1='/media/jf285468/SAUVEGARDES/phemto_simulations/simTra_files/GaussBeamPol%dkeV_%s'%(myene, config)
                f.write("revan -a -n -f {}.inc1.id1.sim.gz -g {} -c revan.cfg\n".format(source_file1,geofile))

                source_file2='/media/jf285468/SAUVEGARDES/phemto_simulations/GaussBeamNonPol%dkeV_%s'%(myene, config)
                f.write("revan -a -n -f {}.inc1.id1.sim.gz -g {} -c revan.cfg\n".format(source_file2,geofile))

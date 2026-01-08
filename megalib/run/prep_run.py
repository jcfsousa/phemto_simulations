def float_range(start, stop, step):
    x = start
    while x < stop - 1e-9:
        yield round(x, 10)
        x += step

CdTe_matrix_size = [4, 5, 6, 7, 8, 9] # consistent with geometries
dists = list(float_range(0.5, 10.5, 1)) # consistent with geometries
instruments_path = '/local/home/jf285468/Documents/phemto_simulations/megalib/instruments'
config_lst = []
for matrix_size in CdTe_matrix_size:
    for dist in dists:
        config = f"config{matrix_size}x{matrix_size}_{dist}cm"
        config_lst.append(config)


#with open(f"./runAll.sh", mode='w') as f:
    #    for config in config_lst:
        #        f.write(f'bash ./runCosima{config}.sh && bash ./runRevan{config}.sh \n')
#
#    f.close()
with open(f"./runAll.sh", mode='w') as f:
    for config in config_lst:
        f.write(f'bash ./runRevan{config}.sh \n')

    f.close()


config_list = ['config1', 'config2', 'config3']


with open(f"./runAll.sh", mode='w') as f:
    for config in config_list:
        f.write(f'bash ./runCosima{config}.sh && bash ./runRevan{config}.sh \n')

    f.close()

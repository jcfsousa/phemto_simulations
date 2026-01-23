import matplotlib.pyplot as plt
import csv
import numpy as np

energy_lst_config1 = []
sensitivity_lst_config1 = []
area_eff_lst_config1 = []

energy_lst_config2 = []
sensitivity_lst_config2 = []
area_eff_lst_config2 = []

energy_lst_config3 = []
sensitivity_lst_config3 = []
area_eff_lst_config3 = []

list_config = ['config1', 'config2', 'config3']

dict_configTothickness = {'config1': 'CdTe 2mm',
                          'config2': 'CZT 5mm',
                          'config3': 'CZT 10mm'}

for config in list_config:
    energy_lst= []
    sensitivity_lst = []
    with open (f'./{config}/PHEMTO_{config}_sensitivity_simLaue.csv', mode='r') as f:
        lines= f.readlines()
        a=0
        for line in lines:
            if a == 0:
                a = 1
                continue
            line = line.split('\n')[0]
            energy = float(line.split(',')[0])
            sensitivity = float(line.split(',')[1])
            energy_lst.append(energy) 
            sensitivity_lst.append(sensitivity)

        f.close()

    energy_lst_old = []
    sensitivity_lst_old = []
    with open ('./old/PHEMTO_old_sensitivity.csv', mode='r') as f:
        lines= f.readlines()
        a=0
        for line in lines:
            if a == 0:
                a = 1
                continue
            line = line.split('\n')[0]
            energy = float(line.split(',')[0])
            sensitivity = float(line.split(',')[1])
            energy_lst_old.append(energy) 
            sensitivity_lst_old.append(sensitivity)

        f.close()

    energy_lst_old = np.array(energy_lst_old)
    energy_lst = np.array(energy_lst)
    sensitivity_lst= np.array(sensitivity_lst)
    sensitivity_lst_old = np.array(sensitivity_lst_old)


#plt.plot(energy_lst_old*0.001, sensitivity_lst_old, label = 'Lauret', c='r')
    plt.plot(energy_lst*0.001, sensitivity_lst, label = f'{dict_configTothickness[config]}', color = 'k')
    plt.xlabel('Energy (MeV)')
    plt.ylabel(r'Sensitivity (erg cm$^{-2}$ s$^{-1}$)')

    plt.legend()
#plt.xlim(1e-4, 1e11)
#plt.ylim(1e-16, 1e-7)
    plt.yscale('log')
    plt.xscale('log')

    plt.savefig(f'./{config}/{config}_sensitivity.png', dpi=700, transparent=True)
    plt.close()

    plt.plot(energy_lst_old*0.001, sensitivity_lst_old, label = 'Lauret', c='r')
    plt.plot(energy_lst*0.001, sensitivity_lst, label = f'{dict_configTothickness[config]}', color = 'k')
    plt.xlabel('Energy (MeV)')
    plt.ylabel(r'Sensitivity (erg cm$^{-2}$ s$^{-1}$)')

    plt.legend()
#plt.xlim(1e-4, 1e11)
#plt.ylim(1e-16, 1e-7)
    plt.yscale('log')
    plt.xscale('log')

    plt.savefig(f'./{config}/{config}_sensitivity_vrsLauret.png', dpi=700, transparent=True)
    plt.close()

    plt.plot(energy_lst*0.001, sensitivity_lst, label = f'{dict_configTothickness[config]}', color = 'k', linewidth = 3)
    plt.xlabel('Energy (MeV)')
    plt.ylabel(r'Sensitivity (erg cm$^{-2}$ s$^{-1}$)')

    plt.legend()
    plt.xlim(1e-4, 1e11)
    plt.ylim(1e-16, 1e-7)
    plt.yscale('log')
    plt.xscale('log')

    plt.savefig(f'./{config}/{config}_sensitivity_several_instruments.png', dpi=700, transparent=True)
    plt.close()


    energy_lst = []
    area_lst = []
    area_eff_lst = []
    with open (f'./{config}/eff_area_{config}.csv', mode='r') as f:
        lines= f.readlines()
        a=0
        for line in lines:
            if a == 0:
                a = 1
                continue
            line = line.split('\n')[0]
            energy = float(line.split(',')[0])
            area = float(line.split(',')[1])
            eff = float(line.split(',')[2])
            area_eff = area*eff
            energy_lst.append(energy) 
            area_lst.append(area)
            area_eff_lst.append(area_eff)

        f.close()

    energy_lst = np.array(energy_lst)
    area_eff_lst = np.array(area_eff_lst)
    area_lst = np.array(area_lst)

    plt.plot(energy_lst*0.001, area_lst, color = 'green', label = 'only lens + mirror')
    plt.plot(energy_lst*0.001, area_eff_lst, color='k', label = f'{dict_configTothickness[config]} (mirro + lens + det)')
    plt.xlabel('Energy (MeV)')
    plt.ylabel(r'Effective area (cm$^2$)')
    plt.yscale('log')
    plt.legend()
#plt.show()
    plt.savefig(f'./{config}/{config}_effectiveArea.png', dpi=700)
    plt.close()



    if config == 'config1':
        energy_lst_config1 = energy_lst
        sensitivity_lst_config1 = sensitivity_lst
        area_eff_lst_config1 = area_eff_lst
    if config == 'config2':
        energy_lst_config2 = energy_lst
        sensitivity_lst_config2 = sensitivity_lst
        area_eff_lst_config2 = area_eff_lst
    if config == 'config3':
        energy_lst_config3 = energy_lst
        sensitivity_lst_config3 = sensitivity_lst
        area_eff_lst_config3 = area_eff_lst


 

plt.plot(energy_lst_config1*0.001, sensitivity_lst_config1, label = f'{dict_configTothickness['config1']}', color = 'k')
plt.plot(energy_lst_config2*0.001, sensitivity_lst_config2, label = f'{dict_configTothickness['config2']}', color = 'r')
plt.plot(energy_lst_config3*0.001, sensitivity_lst_config3, label = f'{dict_configTothickness['config3']}', color = 'g')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Energy (MeV)')
plt.ylabel(r'Sensitivity (erg cm$^{-2}$ s$^{-1}$)')
plt.legend()
plt.savefig('./sensitivity_configs_comparison.png', dpi = 700)
plt.close()


plt.plot(energy_lst_config1*0.001, area_eff_lst_config1, label = f'{dict_configTothickness['config1']}', color = 'k')
plt.plot(energy_lst_config2*0.001, area_eff_lst_config2, label = f'{dict_configTothickness['config2']}', color = 'r')
plt.plot(energy_lst_config3*0.001, area_eff_lst_config3, label = f'{dict_configTothickness['config3']}', color = 'g')
plt.yscale('log')
plt.xscale('log')
plt.xlabel('Energy (MeV)')
plt.ylabel(r'Effective area (cm$^2$)')
plt.legend()
plt.savefig('effarea_configs_comparison.png', dpi = 700)
plt.close()

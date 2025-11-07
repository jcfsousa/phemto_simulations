import gzip
import matplotlib.pyplot as plt
import numpy as np

geofile = '/local/home/jf285468/Documents/PHEMTO/new/instruments/PHEMTO_config1.geo.setup'
log_E=[8,30,50,80,100,120,150,200,250,300,400,500,600]

geofile_name = geofile.split('/')[-1]
geofile_onlyname = geofile_name.split('.')[0]

source_basename = 'HomogeneousBeam'

spectra_dict = {}

for energy in log_E:
    trafile = f'./sources/simTra_files/{source_basename}{energy}keV_{geofile_onlyname}.inc2.id1.tra.gz'
    print(f'reading {trafile} ...')
    with gzip.open(trafile,'rt') as f:
        spectra_file = []
        tra_content = f.read()
        tra_chunck_split = tra_content.split('FT START')[0]
        tra_events = tra_chunck_split.split('SE')[1:]
        for event in tra_events:
            split_event = event.splitlines()
            if split_event[1] == 'ET PH':
               event_id = split_event[2].split(' ')[-1]
               interaction_time = float(split_event[3].split(' ')[-1]) # seconds
               event_energy = float(split_event[4].split(' ')[-1]) #keV
               event_x = float(split_event[5].split(' ')[1]) #cm
               event_y = float(split_event[5].split(' ')[2]) #cm
               event_z = float(split_event[5].split(' ')[3]) #cm
               spectra_file.append(event_energy)
            if split_event[1] == 'ET CO':
               event_id = split_event[2].split(' ')[-1]
               interaction_time = float(split_event[3].split(' ')[-1]) # seconds
               ch0_event_data = split_event[11]
               ch0_event_energy = float(ch0_event_data.split(' ')[5])
               ch0_event_x = float(ch0_event_data.split(' ')[2])
               ch0_event_y = float(ch0_event_data.split(' ')[3])
               ch0_event_z = float(ch0_event_data.split(' ')[4])
               ch1_event_data = split_event[12]
               ch1_event_energy = float(ch1_event_data.split(' ')[5])
               ch1_event_x = float(ch1_event_data.split(' ')[2])
               ch1_event_y = float(ch1_event_data.split(' ')[3])
               ch1_event_z = float(ch1_event_data.split(' ')[4])
               event_energy = ch0_event_energy + ch1_event_energy
               spectra_file.append(event_energy)
            else:
                continue
    spectra_dict[f'{energy}'] = spectra_file


for energy in log_E:
    print(energy)
    energy_list = spectra_dict[f'{energy}']

    xmin = 1
    xmax = 700
    steps = xmax-xmin
    bins = np.linspace(xmin, xmax, steps)
    hist, bins = np.histogram(energy_list, bins=bins)
    bins_centered = (bins[:-1] + bins[1:])/2

    plt.plot(bins_centered, hist)
    plt.yscale('log')
    plt.show()


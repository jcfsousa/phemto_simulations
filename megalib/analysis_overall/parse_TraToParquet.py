import gzip
from scipy.signal import find_peaks
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.colors import LogNorm

def check_event_detector(z):
   if z > -1: # si 
       return 0
   else:
       return 1
def coordinate_transform(x, instrument_size):
    '''
    Apply coordinate transfrom from x,y = 0,0 in the center of the geometry to the left bottom corner of the detector
    '''
    return x + (instrument_size/2)

def get_pixel_id(x, pixel_size):
    '''
    Returns pix ID, for a given x,y coordinate. Need to compute coordinate_transform() first to be on the correct reference frame.
    '''
    ix = np.floor(x / pixel_size).astype(int) -1
    return ix

def get_matrix_id(x_pix, y_pix, matrix_size):
    '''
    matrix_size: number of pixels in x direction (assuming y direction has the same ammount of pix)
    '''
    pix_id = y_pix * matrix_size + x_pix
    return pix_id
    

def check_mega_event_insideDet(x, y, instrument_size):
    '''
    Sometimes megalib gives events that are ouside the detector. Check HomogeneousBeam661keV_config1.inc1.id1.tra.gz event ID 313, coordinat on CdTe is x=-5cm with reference on center of detector, its outside cdte detector
    x,y, on the detector ref frame
    '''
    if x > instrument_size or x < 0:
        return False
    elif y>instrument_size or y<0:
        return False
    else:
        return True

if __name__ == '__main__':
    
    config_list = ['config1']
    path_base_tra = "/local/home/jf285468/documents/phd/phd-MEGAlib/PHEMTO/v1.1/sources/simTra_files"
    #log_E=[4,8,30,50,80,100,120,150,200,250,300,400,500,600]
    
    E_init = 50
    Log_E=[]
    while E_init <= 693.5:
        Log_E.append(E_init)
        E_init = E_init + 6.5
        
    Log_E = [661]
    source_basename = 'HomogeneousBeam'
    
    instrument_size_cdte = 6.4 #6.4 x 6.4 cm2, 4x4 MC2
    pix_size_cdte = 0.025 # cm = 0.25mm
    matrix_size_cdte = 256 # 256x256
    
    #Si: https://www.sciencedirect.com/science/article/pii/S0168900219310812
    instrument_size_si = 6.656 #6.4 x 6.4 cm2, 4x4 MC2
    pix_size_si = 0.013 # cm = 0.13mm
    matrix_size_si = 512 # 512x512
    for config in config_list:
        spectra_dict = {}
        for energy in Log_E:
            event_lst = []
            time_lst = []
            energy_lst = []
            x_lst = []
            xpix_lst = []
            y_lst = []
            ypix_lst = []
            matrixID_lst = []
            z_lst = []
            chip_id = [] #0=Si, 1=CdTe
            source_name = f"{source_basename}%dkeV_{config}"%(energy)
            output_folder = f"./input_polarimetry/{source_name}"
            output_folder = f'./input_polarimetry/pol_300keV'
            trafile = f'{path_base_tra}/{source_name}.inc1.id1.tra.gz'
            trafile = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/pol_200keV.inc1.id1.tra.gz'
            trafile = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/input_polarimetry/pol_300keV/pol_300keV.inc2.id1.tra.gz'
            print(f'reading {trafile} ...')
            with gzip.open(trafile,'rt') as f:
                tra_content = f.read()
                tra_chunck_split = tra_content.split('FT START')[0]
                tra_events = tra_chunck_split.split('SE')[1:]
                event_cnt = 0
                max_event_cnt = 1e5
                ouput_file_cntr = 0
                for event in tra_events:
                    split_event = event.splitlines()
                    if split_event[1] == 'ET PH':
                        event_id = split_event[2].split(' ')[-1]
                        interaction_time = float(split_event[3].split(' ')[-1]) * 10e9 # nano seconds
                        event_energy = float(split_event[4].split(' ')[-1]) #keV
                        event_x = float(split_event[5].split(' ')[1]) #cm
                        event_y = float(split_event[5].split(' ')[2]) #cm
                        event_z = float(split_event[5].split(' ')[3]) #cm
                        
                        if check_event_detector(event_z) == 1:
                            instrument_size = instrument_size_cdte
                            pix_size = pix_size_cdte
                            matrix_size = matrix_size_cdte

                        else:
                            instrument_size = instrument_size_si
                            pix_size = pix_size_si
                            matrix_size = matrix_size_si

                        #transforming x,y coordinate to CdTe detector reference frame, 0,0 at bottom left
                        event_x_detRef = coordinate_transform(event_x, instrument_size)
                        event_y_detRef = coordinate_transform(event_y, instrument_size)
                        if check_mega_event_insideDet(event_x_detRef, event_y_detRef, instrument_size) == False:
                            print("Megalib Event outside detector, skipping event")
                            continue
                        else:
                            #getting pixel coordinate 
                            event_xpix_detRef = get_pixel_id(event_x_detRef,  pix_size)
                            event_ypix_detRef = get_pixel_id(event_y_detRef,  pix_size)
                            xpix_lst.append(event_xpix_detRef)
                            ypix_lst.append(event_ypix_detRef)
                            #computing pixel id, 0-256 row of pixels (left to right), 0-256 collumn of pixels (bottom to up)
                            matrix_id = get_matrix_id(event_xpix_detRef, event_ypix_detRef, matrix_size)
                            matrixID_lst.append(matrix_id)
                            #print(f"event_x_detRef: {event_x_detRef:.3f} cm")
                            #print(f"event_y_detRef: {event_y_detRef:.3f} cm")
                            #print(f"xpix: {event_xpix_detRef}, ypix: {event_ypix_detRef}")
                            #print(f"matrix_id: {matrix_id}")
                              

                            event_lst.append(int(event_id))
                            time_lst.append(interaction_time)
                            energy_lst.append(event_energy)
                            x_lst.append(event_x)
                            y_lst.append(event_y)
                            z_lst.append(event_z)
                            chip_id.append(check_event_detector(event_z))
                            event_cnt += 1


                    elif split_event[1] == 'ET CO':
                        event_id = split_event[2].split(' ')[-1]
                        interaction_time = float(split_event[3].split(' ')[-1]) * 10e9 # nano seconds
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
                        
                        #cluster 1
                        if check_event_detector(ch0_event_z) == 1:
                            instrument_size = instrument_size_cdte
                            pix_size = pix_size_cdte
                            matrix_size = matrix_size_cdte

                        else:
                            instrument_size = instrument_size_si
                            pix_size = pix_size_si
                            matrix_size = matrix_size_si

                        ch0_event_x_detRef = coordinate_transform(ch0_event_x, instrument_size)
                        ch0_event_y_detRef = coordinate_transform(ch0_event_y, instrument_size)

                        if check_mega_event_insideDet(ch0_event_x_detRef, ch0_event_y_detRef, instrument_size) == False:
                            print("Megalib Event outside detector, skipping event")
                            continue
                        else:
                            ch0_event_xpix_detRef = get_pixel_id(ch0_event_x_detRef, pix_size)
                            ch0_event_ypix_detRef = get_pixel_id(ch0_event_y_detRef, pix_size)
                            ch0_matrix_id = get_matrix_id(ch0_event_xpix_detRef, ch0_event_ypix_detRef, matrix_size)
                            #print(f"\n=== COMPTON EVENT {event_id} ===")
                            #print(f"Z: {ch0_event_z:.3f} cm ")
                            #print(f"instrument_size_si: {instrument_size_si}")
                            #print(f"event_x : {ch0_event_x:.3f} cm")
                            #print(f"event_y : {ch0_event_y:.3f} cm")
                            #print(f"event_x_detRef: {ch0_event_x_detRef:.3f} cm")
                            #print(f"event_y_detRef: {ch0_event_y_detRef:.3f} cm")
                            #print(f"xpix: {ch0_event_xpix_detRef}, ypix: {ch0_event_ypix_detRef}")
                            #print(f"matrix_id: {ch0_matrix_id}")
                            #input("COMPTON event - press Enter to continue...")
                            xpix_lst.append(ch0_event_xpix_detRef)
                            ypix_lst.append(ch0_event_ypix_detRef)
                            matrixID_lst.append(ch0_matrix_id)
                            event_lst.append(int(event_id))
                            time_lst.append(interaction_time)
                            energy_lst.append(ch0_event_energy)
                            x_lst.append(ch0_event_x)
                            y_lst.append(ch0_event_y)
                            z_lst.append(ch0_event_z)
                            chip_id.append(check_event_detector(ch0_event_z))
                       

                        #cluster 2
                        if check_event_detector(ch1_event_z) == 1:
                            instrument_size = instrument_size_cdte
                            pix_size = pix_size_cdte
                            matrix_size = matrix_size_cdte
                        else:
                            instrument_size = instrument_size_si
                            pix_size = pix_size_si
                            matrix_size = matrix_size_si

                        ch1_event_x_detRef = coordinate_transform(ch1_event_x, instrument_size)
                        ch1_event_y_detRef = coordinate_transform(ch1_event_y, instrument_size)
                        if check_mega_event_insideDet(ch1_event_x_detRef, ch1_event_y_detRef, instrument_size) == False:
                            print("Megalib Event outside detector, skipping event")
                            continue
                        else:
                            ch1_event_xpix_detRef = get_pixel_id(ch1_event_x_detRef, pix_size)
                            ch1_event_ypix_detRef = get_pixel_id(ch1_event_y_detRef, pix_size)
                            ch1_matrix_id = get_matrix_id(ch1_event_xpix_detRef, ch1_event_ypix_detRef, matrix_size)
                            #print(f"\n=== COMPTON EVENT {event_id} ===")
                            #print(f"Z: {ch1_event_z:.3f} cm ")
                            #print(f"instrument_size_si: {instrument_size_si}")
                            #print(f"event_x : {ch1_event_x:.3f} cm")
                            #print(f"event_y : {ch1_event_y:.3f} cm")
                            #print(f"event_x_detRef: {ch1_event_x_detRef:.3f} cm")
                            #print(f"event_y_detRef: {ch1_event_y_detRef:.3f} cm")
                            #print(f"xpix: {ch1_event_xpix_detRef}, ypix: {ch1_event_ypix_detRef}")
                            #print(f"matrix_id: {ch1_matrix_id}")
                            #input("COMPTON event - press Enter to continue...")
                            xpix_lst.append(ch1_event_xpix_detRef)
                            ypix_lst.append(ch1_event_ypix_detRef)
                            matrixID_lst.append(ch1_matrix_id)
                            event_lst.append(int(event_id))
                            time_lst.append(interaction_time)
                            energy_lst.append(ch1_event_energy)
                            x_lst.append(ch1_event_x)
                            y_lst.append(ch1_event_y)
                            z_lst.append(ch1_event_z)
                            chip_id.append(check_event_detector(ch1_event_z))
                        event_cnt += 1
                    
                    else:
                        print(split_event)
                        input()
                        print(f'No single, No compton double, other event type: {split_event[1]}')
                    
                    if event_cnt == max_event_cnt:
                        df = pd.DataFrame()
                        df['Event'] = event_lst
                        df['Matrix Index'] = matrixID_lst
                        df['ToT (keV)'] = energy_lst
                        df['X'] = xpix_lst
                        df['Y'] = ypix_lst
                        df['Ns'] = time_lst
                        df['Overflow'] = chip_id
                        df.to_csv(f"{output_folder}/_{ouput_file_cntr}.t3pa", index=False) #the _ is mandatory atm 15dec2025
                        ouput_file_cntr += 1
                        event_cnt = 0
                        event_lst = []
                        time_lst = []
                        energy_lst = []
                        x_lst = []
                        xpix_lst = []
                        y_lst = []
                        ypix_lst = []
                        matrixID_lst = []
                        z_lst = []
                        chip_id = [] #0=Si, 1=CdTe

        
        #print(df)

        df_cdte = df[df['Overflow'] == 1] ## CdTe
        pixel_energy = df_cdte.groupby(['X', 'Y'])['ToT (keV)'].sum().reset_index()

# Create energy map
        energy_map = np.zeros((matrix_size_cdte, matrix_size_cdte))
        for _, row in pixel_energy.iterrows():
            x = int(row['X'])
            y = int(row['Y'])
            energy_map[y, x] = row['ToT (keV)']  # Fill with summed energy

        print(f"Max pixel energy (grouped): {energy_map.max():.1f} keV")
        print(f"Unique pixels with energy: {len(pixel_energy)}")

        plt.figure(figsize=(16, 12))
        plt.subplot(2,2,1)
        plt.imshow(energy_map, cmap='jet', origin='lower', interpolation='nearest')
        plt.colorbar(label='Total energy deposited (keV)')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.title('CdTe: Energy Summed per Pixel (GroupBy)')

        plt.subplot(2,2,3)
        im = plt.imshow(energy_map, cmap='jet', origin='lower', 
                        norm=LogNorm(vmin=energy_map[energy_map > 0].min(), 
                                    vmax=energy_map.max()),
                        interpolation='nearest')
        plt.colorbar(im, label='Total energy deposited (keV)')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        plt.title('CdTe: Energy Summed per Pixel (Log Scale)')



        df_si = df[df['Overflow'] == 0] # Si detector
        pixel_energy_si = df_si.groupby(['X', 'Y'])['ToT (keV)'].sum().reset_index()
        energy_map_si = np.zeros((matrix_size_si, matrix_size_si))
        for _, row in pixel_energy_si.iterrows():
            x = int(row['X'])
            y = int(row['Y'])
            energy_map_si[y, x] = row['ToT (keV)']
        
        print(f"Si - Max pixel energy: {energy_map_si.max():.1f} keV")
        print(f"Si - Unique pixels: {len(pixel_energy_si)}")
        
        # Linear scale
        plt.subplot(2,2,2)
        plt.imshow(energy_map_si, cmap='jet', origin='lower', interpolation='nearest')
        plt.colorbar(label='Total energy (keV)')
        plt.title('Si: Linear Scale')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')
        
        # Log scale
        plt.subplot(2,2,4)
        im = plt.imshow(energy_map_si, cmap='jet', origin='lower',
                        norm=LogNorm(vmin=energy_map_si[energy_map_si > 0].min(),
                                    vmax=energy_map_si.max()),
                        interpolation='nearest')
        plt.colorbar(im, label='Total energy (keV)')
        plt.title('Si: Log Scale')
        plt.xlabel('X pixel')
        plt.ylabel('Y pixel')

        plt.tight_layout()
        plt.show()

import os
import subprocess
import manalysis.specLib as specLib
import subprocess
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
from calibration.calibration import Calibration

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm

import manalysis.comptons as compton
from datetime import datetime


def compute_max_dist(x_max, x_min, y_max, y_min):
    
    x_max = x_max - (int(chip_id) * 256) - 1
    x_min = x_min - (int(chip_id) * 256) - 1
    beam_limits = [x_max, x_min, y_max, y_min]

    limit_compton_dist_dict = {
        'x_distance_to_Rboarder': 255 - beam_limits[0],  
        'x_distance_to_Lboarder': beam_limits[1],       
        'y_distance_to_TBoarder': 255 - beam_limits[2], 
        'y_distance_to_BBoarder': beam_limits[3]       
        }
    
    min_dist_value = min(limit_compton_dist_dict.values())
    min_dist_value = min_dist_value - 4 # remove 4 pixels form the edge
    min_dist_value_mm = min_dist_value * 0.055
    
    return min_dist_value_mm


def detect_sudden_changes(data, threshold=2.0):
    """
    Detects sudden changes in a 1D array using the first derivative.
    
    Parameters:
        data (array-like): 1D array containing energy counts (e.g., x_energy_deposited).
        threshold (float): Threshold for detecting a significant change (in standard deviations).
        
    Returns:
        List of indices where sudden changes occur.
    """
    # Calculate the first derivative (rate of change)
    diff = np.diff(data)
    #print(diff)

    std_dev = np.std(diff)
    #print(std_dev)

    normalized_diff = diff / std_dev if std_dev != 0 else diff
    #print(normalized_diff)

    sudden_changes = np.where(np.abs(normalized_diff) > threshold)[0]


    return sudden_changes.tolist()


def find_beam_location(output_folder_base, source, chip_id):

    
    resultFolder = os.path.join(specLib.global_config.output_folder, source)
    resultFolder_parquet = os.path.join(resultFolder, 'parquet')    

    singles_folder = os.path.join(resultFolder_parquet,'singles')
    doubles_folder = os.path.join(resultFolder_parquet, 'doubles')
    multiples_folder = os.path.join(resultFolder_parquet, 'multiples')
    masked_folder = os.path.join(resultFolder_parquet, 'masked')

    
    files = specLib.get_list_files(masked_folder, endswith='.parquet')
    pixel_matrix = np.ones((256,1024))
    i = 0

    pixel_matrix_filepath = f'{output_folder}/pixel_matrix.csv'

    if not os.path.exists(pixel_matrix_filepath):
        for file in files:
            df = pd.read_parquet(f'{masked_folder}/{file}', columns=['X', 'Y', 'ToT (keV)', 'Overflow'])
            
            df = df[df['Overflow'] == chip_id]

            df['X'] = abs(df['X'])
            df['Y'] = abs(df['Y'])

            grouped_df = df.groupby(['X', 'Y'])['ToT (keV)'].sum().reset_index()
            

            for _, row in grouped_df.iterrows():
                x, y, tot = int(row['X']), int(row['Y']), row['ToT (keV)']
                if x == 530 and y == 32:
                    continue
                pixel_matrix[y, x] += tot

        np.savetxt(f"{output_folder}/pixel_matrix.csv", pixel_matrix, delimiter=',', fmt="%.2f")
    else:
        pixel_matrix = np.loadtxt(pixel_matrix_filepath, delimiter=',')

    x_energy_deposited = np.sum(pixel_matrix, axis=0)
    y_energy_deposited = np.sum(pixel_matrix, axis=1)


    x_interesting_points = detect_sudden_changes(x_energy_deposited, threshold=5.0)
    y_interesting_points = detect_sudden_changes(y_energy_deposited, threshold=5.0)

    print(f"x_interesting_points: {x_interesting_points}")
    print(f"y_interesting_points: {y_interesting_points}")
    
    x_max = max(x_interesting_points) + 1
    x_min = min(x_interesting_points)

    y_max = max(y_interesting_points) + 2
    y_min = min(y_interesting_points) - 1

    return x_max, x_min, y_max, y_min


def perform_beam_img(output_folder_base, source, chip_id):
    resultFolder = os.path.join(specLib.global_config.output_folder, source)
        

    os.makedirs(resultFolder, exist_ok=True)
    resultFolder_parquet = os.path.join(resultFolder, 'parquet')    

    singles_folder = os.path.join(resultFolder_parquet,'singles')
    doubles_folder = os.path.join(resultFolder_parquet, 'doubles')
    multiples_folder = os.path.join(resultFolder_parquet, 'multiples')
    masked_folder = os.path.join(resultFolder_parquet, 'masked')

    
    files = specLib.get_list_files(masked_folder, endswith='.parquet')
    pixel_matrix = np.ones((256,1024))
    i = 0
    
    pixel_matrix_filepath = f'{resultFolder}/pixel_matrix.csv'

    if not os.path.exists(pixel_matrix_filepath):
        for file in files:
            df = pd.read_parquet(f'{masked_folder}/{file}', columns=['X', 'Y', 'ToT (keV)', 'Overflow'])
            
            df = df[df['Overflow'] == chip_id]

            df['X'] = abs(df['X'])
            df['Y'] = abs(df['Y'])

            grouped_df = df.groupby(['X', 'Y'])['ToT (keV)'].sum().reset_index()
            

            for _, row in grouped_df.iterrows():
                x, y, tot = int(row['X']), int(row['Y']), row['ToT (keV)']
                if x == 530 and y == 32:
                    continue
                pixel_matrix[y, x] += tot

        np.savetxt(f"{resultFolder}/pixel_matrix.csv", pixel_matrix, delimiter=',', fmt="%.2f")
    else:
        pixel_matrix = np.loadtxt(pixel_matrix_filepath, delimiter=',')
        

    x_energy_deposited = np.sum(pixel_matrix, axis=0)
    y_energy_deposited = np.sum(pixel_matrix, axis=1)


    x_interesting_points = detect_sudden_changes(x_energy_deposited, threshold=5.0)
    y_interesting_points = detect_sudden_changes(y_energy_deposited, threshold=5.0)

    x_max = max(x_interesting_points) + 1
    x_min = min(x_interesting_points)

    y_max = max(y_interesting_points) + 2
    y_min = min(y_interesting_points) - 1
        

    x_matrix_Ledge = int(chip_id) * 256
    x_matrix_Redge = int(chip_id) * 256 + 255

    plt.figure(figsize=(12, 12))


# Main matrix plot
    ax_img = plt.gcf().add_axes([0.1, 0.1, 0.7, 0.7])
    ax_img.imshow(pixel_matrix, origin='lower', cmap='jet', aspect='auto', 
               norm=LogNorm(vmin=1000, vmax=np.max(pixel_matrix)))
    #ax_img._colorbars(label='Cumulative ToT (keV)', orientation='horizontal')
    #ax_img.set_xlim(x_matrix_Ledge, x_matrix_Redge)
    ax_img.set_xlim(x_min-70, x_max+70)
    ax_img.set_ylim(y_min-70, y_max+70)
    ax_img.set_xlabel('X Pixel')
    ax_img.set_ylabel('Y Pixel')
    #ax_img.set_title(f'Cumulative Pixel Energy Heatmap for Source: {source}')

# Add vertical lines as markers
    for x in [256, 512, 768]:
        plt.axvline(x=x, color='red', linestyle='--', linewidth=1, label=f'x={x}')
    
    plt.axvline(x=x_min, color = 'red', linewidth=2)
    plt.axvline(x=x_max, color = 'red', linewidth=2)
    plt.axhline(y=y_min, color = 'red', linewidth=2)
    plt.axhline(y=y_max, color = 'red', linewidth=2)

# Add inset for x-energy histogram
    ax_x = plt.gcf().add_axes([0.1, 0.8, 0.7, 0.1])  # [left, bottom, width, height]
    #ax_x.bar(range(len(x_energy_deposited)), x_energy_deposited, color='black', alpha=0.7)
    ax_x.fill_between(range(len(x_energy_deposited)), x_energy_deposited, color='black', alpha=0.5)
    #x_heatmap = np.expand_dims(x_energy_deposited, axis=0)  # Convert to 2D for imshow
    #ax_x.imshow(x_heatmap, aspect='auto', cmap='Greys', extent=[0, len(x_energy_deposited), 0, 1], norm=LogNorm(vmin=1, vmax=np.max(x_heatmap)))
    ax_x.set_xlim(x_min-40, x_max+40)
    #ax_x.set_yticks([])
    ax_x.set_yscale('log')
    ax_x.set_ylabel('Energy (keV)')
    ax_x.set_title(f'Energy Deposition Map for Source: {source}')

# Add inset for y-energy histogram
    ax_y = plt.gcf().add_axes([0.80, 0.1, 0.1, 0.7])  # [left, bottom, width, height]
    #ax_y.barh(range(len(y_energy_deposited)), y_energy_deposited, color='black', alpha=0.7)
    ax_y.fill_between(y_energy_deposited, range(len(y_energy_deposited)), color='black', alpha=0.5)
    ax_y.set_ylim(y_min-40, y_max+40)
    ax_y.set_xscale('log')
    ax_y.set_xlabel('Energy (keV)')
    #ax_y.set_title('Y-Axis Energy Distribution', rotation=90)

    plt.savefig(f"{resultFolder}/beam_image_{specLib.global_config.chip_dict[f'{chip_id}']}")
    plt.close()

    #plt.show()


    return 


if __name__ == '__main__':
    
    start_time = datetime.now()

    config_file = "/home/josesousa/Documents/thor/detector/detSoftware/detanalysis/polarimetry/config_LabCol.json"
    

    specLib.global_config = specLib.Config(config_file)
    
    #automatic update the chip config on the calibration.py script (its hard coded idk why)
    result = subprocess.run(['./update_chip_config.sh'],text=True, input = specLib.global_config.config_chips)
    


    sources = specLib.global_config.sources
    sources_peaks = specLib.global_config.sources_peaks
    abct_folder = specLib.global_config.abct_folder
    output_folder_base = specLib.global_config.output_folder
    input_folder = specLib.global_config.input_dir

    chip = 'F04-W0060'
    chip_id = specLib.get_chip_id(chip)


   
    fontsize = 14
    plt.rcParams['figure.figsize'] = ((4/3)*5.5, 5.5)
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['axes.titlesize'] = fontsize+4
    plt.rcParams['figure.titlesize'] = fontsize+4
    plt.rcParams['axes.titlepad'] = fontsize
    plt.rcParams['axes.axisbelow'] = True
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['axes.labelsize'] = fontsize
    plt.rcParams['xtick.labelsize'] = fontsize 
    plt.rcParams['ytick.labelsize'] = fontsize
    plt.rcParams['axes.grid'] = True
    plt.rcParams['savefig.dpi'] = 300


    plt.rcParams['grid.linestyle'] = 'dotted'
    
    for source in sources:

        data_folder = os.path.join(input_folder, source)
        _  = None
        calib = Calibration(output_folder_base, _)

        args = (source, output_folder_base, sources_peaks)
        specLib.pre_process_source(args)

        output_folder = os.path.join(output_folder_base, source)


        energy, x_center_pixel, y_center_pixel = calib.set_up(data_folder, 'beam_energy', 'x_center_pixel', 'y_center_pixel')
        print(f"going to run: {sources}")

    
        print("Performing beam image....")
        perform_beam_img(output_folder_base, source, chip_id)
        print("Done....")

        print("Localizing beam position....")
        x_max, x_min, y_max, y_min = find_beam_location(output_folder_base, source, chip_id)
        print("Done....")

        print("Computing max distance of Compton interaction")
        #max_dist_computed = compute_max_dist(x_max, x_min, y_max, y_min) 
        max_dist_computed = 4.18
        print("Done....")
       
  
        min_dist_start = 0.055  # mm
        min_dist_end = 3        # mm
        min_dist_step = 0.055

        min_dist_list = list(np.arange(min_dist_start, min_dist_end + min_dist_step , min_dist_step))
        angle_bin_list = [x for x in range(1, 37) if 360 % x == 0]

        #max_dist_list = list(np.arange(min_dist_end+3*0.055 , max_dist_computed, min_dist_step))
        max_dist_list = [max_dist_computed]
        max_dist_on_list = max_dist_list[-1]

        print(f'iterations: {len(min_dist_list)*len(angle_bin_list)}')


        compton.identify_compton(output_folder, abct_folder, energy, x_center_pixel, y_center_pixel, chip)
        
        print(f'Counting number of event type, single, double, multiple, compton...')
        compton.count_nEvents_allTypes(output_folder, energy, chip)

        polarimetry_task = [(min_dist, angle_bin, output_folder, x_center_pixel, y_center_pixel, energy, chip, max_dist, abct_folder) for min_dist, angle_bin, max_dist in product(min_dist_list, angle_bin_list, max_dist_list)]
        with Pool() as pool:
            for _ in tqdm(pool.imap_unordered(compton.polarimetry_task_residual, polarimetry_task), total=len(polarimetry_task), desc='Compton Polarimetry'):
                pass
        

        #max_merit, best_min_dist, best_angle_bin = compton.plot_figureMeritMap(output_folder, min_dist_list, angle_bin_list, max_dist_list)
       # 
       # print(f'best_min_dist: {best_min_dist}')
       # print(f'best_angle_bin: {best_angle_bin}')
       # print(f'max_merit: {max_merit}')


        #compton.plot_QvrsRadius(output_folder, min_dist_list, [best_angle_bin], max_dist_on_list) # with fixed angle bin!!
        #compton.plot_QvrsBin(output_folder, [best_min_dist], angle_bin_list, max_dist_on_list) # with fixed angle bin!!
        #compton.plot_EffvrsRadius(output_folder, min_dist_list, [best_angle_bin], max_dist_on_list) # with fixed angle bin!!
        #compton.plot_QvrsMaxdist(output_folder, best_min_dist, best_angle_bin, max_dist_list)

    #compton.plot_rotationMeasurements(output_folder_base, sources, [0.55], [1])

    print('') 
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    end_time = datetime.now()
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate total run time
    total_time = end_time - start_time
    print(f"Total Run Time: {total_time}")



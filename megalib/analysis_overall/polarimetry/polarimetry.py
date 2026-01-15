import os
import sys
import matplotlib.patches as patches
import colorcet as cc
import subprocess

import manalysis.specLib as specLib
import manalysis.comptons as compton
import manalysis.configlib as configlib
import manalysis.pathlib as pathlib

import subprocess
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
from calibration.calibration import Calibration
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from datetime import datetime

fontsize = 20
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (8,8)
plt.rcParams['font.size'] = fontsize
plt.rcParams['axes.titlesize'] = fontsize + 4 
plt.rcParams['figure.titlesize'] = fontsize + 6
plt.rcParams['axes.labelsize'] = fontsize + 6
plt.rcParams['axes.titlepad'] = fontsize
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['legend.facecolor'] = 'white'
plt.rcParams['legend.edgecolor'] = 'black'
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.fontsize'] = fontsize - 4 
plt.rcParams['axes.labelsize'] = fontsize + 2
plt.rcParams['xtick.labelsize'] = fontsize 
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.grid'] = True
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = fontsize*0.35
plt.rcParams['ytick.major.size'] = fontsize*0.35
plt.rcParams['xtick.minor.size'] = fontsize*0.175
plt.rcParams['ytick.minor.size'] = fontsize*0.175
plt.rcParams['xtick.major.width'] = fontsize*0.1
plt.rcParams['ytick.major.width'] = fontsize*0.1
plt.rcParams['xtick.minor.width'] = fontsize*0.1
plt.rcParams['ytick.minor.width'] = fontsize*0.1
plt.rcParams['axes.linewidth'] = fontsize/fontsize
plt.rcParams['grid.linestyle'] = 'dotted'

def compute_max_dist(x_max, x_min, y_max, y_min, chip_id = 2):
    
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



def find_beam_location(output_folder, source, chip_id):

    
    resultFolder = os.path.join(specLib.global_config.output_folder, source)
    resultFolder_parquet = os.path.join(resultFolder, 'parquet')    

    singles_folder = os.path.join(resultFolder_parquet,'singles')
    doubles_folder = os.path.join(resultFolder_parquet, 'doubles')
    multiples_folder = os.path.join(resultFolder_parquet, 'multiples')
    masked_folder = os.path.join(resultFolder_parquet, 'masked')

    
    files = pathlib.get_list_files(masked_folder, endswith='.parquet')
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


    x_interesting_points = compton.detect_sudden_changes(x_energy_deposited, threshold=5.0)
    y_interesting_points = compton.detect_sudden_changes(y_energy_deposited, threshold=5.0)

    #print(f"x_interesting_points: {x_interesting_points}")
    #print(f"y_interesting_points: {y_interesting_points}")
    
    x_max = max(x_interesting_points) + 1
    x_min = min(x_interesting_points)
    
    y_max = max(y_interesting_points) + 2
    y_min = min(y_interesting_points) - 1

    return x_max, x_min, y_max, y_min


def perform_beam_img(output_folder_base, source, chip_id):
    resultFolder = os.path.join(specLib.global_config.output_folder, source)
        
    #print(f"CHIP IF INSIDE BEAM IMG: {chip_id}")
    os.makedirs(resultFolder, exist_ok=True)
    resultFolder_parquet = os.path.join(resultFolder, 'parquet')    

    singles_folder = os.path.join(resultFolder_parquet,'singles')
    doubles_folder = os.path.join(resultFolder_parquet, 'doubles')
    multiples_folder = os.path.join(resultFolder_parquet, 'multiples')
    masked_folder = os.path.join(resultFolder_parquet, 'masked')

 
    files = pathlib.get_list_files(masked_folder, endswith='.parquet')
    pixel_matrix = np.ones((256,1024))
    i = 0
    
    pixel_matrix_filepath = f'{output_folder}/pixel_matrix.csv'

    if not os.path.exists(pixel_matrix_filepath):
        for file in files:
            df = pd.read_parquet(f'{masked_folder}/{file}', columns=['X', 'Y', 'ToT (keV)', 'Overflow'])
            print(df)
            
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
        pixel_matrix = np.loadtxt(pixel_matrix_filepath, delimiter=',')
        pixel_matrix = np.roll(pixel_matrix, shift=-512, axis=1) 

    x_energy_deposited = np.sum(pixel_matrix, axis=0)
    y_energy_deposited = np.sum(pixel_matrix, axis=1)


    x_interesting_points = compton.detect_sudden_changes(x_energy_deposited, threshold=1.0)
    y_interesting_points = compton.detect_sudden_changes(y_energy_deposited, threshold=1.0)

    x_max = max(x_interesting_points) + 1
    x_min = min(x_interesting_points)

    y_max = max(y_interesting_points) + 2
    y_min = min(y_interesting_points) - 1
        

    x_matrix_Ledge = int(chip_id) * 256
    x_matrix_Redge = int(chip_id) * 256 + 255

    fig = plt.figure(figsize=(7, 7))
    print(np.max(x_energy_deposited))
    max_E = np.max(x_energy_deposited)

    colors = sum([cc.glasbey_cool[i:i + 4] for i in range(5, 50, 10)], [])
    cmap = ListedColormap(colors)

    energy_source = 206#compton.get_energy_from_source_name(source)

# Main matrix plot
    ax_img = plt.gcf().add_axes((0.13, 0.1, 0.7, 0.7))
    im = ax_img.imshow(pixel_matrix, origin='lower', cmap='jet' , aspect='auto', 
               norm=LogNorm(vmin=1, vmax=max_E))
    
    #cbar = plt.colorbar(im, ax=ax_img, orientation='vertical', pad=0.01)
    #cbar.set_label(f'{energy_source} keV')
    rect = patches.Rectangle((x_max + 47, y_max + 66), 25, 10, 
                             linewidth=1.5, edgecolor='black', facecolor='white', alpha=1)
    ax_img.add_patch(rect)

# Annotate inside the rectangle
    ax_img.annotate(f'{energy_source} keV', (x_max + 50, y_max + 70), 
                    color='black', fontsize=12, ha='left', va='center')

    cbar_ax = fig.add_axes((0.931, 0.105, 0.03, 0.696))  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Intensity (keV)')
    vmin, vmax = 1, max_E  # Your current LogNorm bounds

# Choose the number of decades (log steps) you want
    num_ticks = 6  # Adjust based on your needs
    ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=num_ticks)

# Set the ticks and format them properly
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{tick:.1f}" for tick in ticks])
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)  # Enables scientific notation
    cbar.formatter.set_powerlimits((0, 0))  # Force scientific notation for all ticks
    cbar.update_ticks()  # Apply changes
# Add colorbar
    #cbar.set_label(f'{energy_source} keV')
    #ax_img.imshow(
    #pixel_matrix, 
    #origin='lower', 
    #cmap='tab20c', 
    #aspect='auto', 
    #vmin=200, 
    #vmax=30e3
  # )

    #ax_img._colorbars(label='Cumulative ToT (keV)', orientation='horizontal')
    #ax_img.set_xlim(x_matrix_Ledge, x_matrix_Redge)
    ax_img.set_xlim(x_min-80, x_max+80)
    ax_img.set_ylim(y_min-80, y_max+80)
    ax_img.set_xlabel(r'X-position [pixel \#]')
    ax_img.set_ylabel(r'Y-position [pixel \#]')
    #ax_img.set_title(f'Cumulative Pixel Energy Heatmap for Source: {source}')
    ax_img.grid(False)

# Add vertical lines as markers
    for x in [256, 512, 768]:
        plt.axvline(x=x, color='red', linestyle='--', linewidth=1, label=f'x={x}')
    
    plt.axvline(x=x_min, color = '#C6B800', linewidth=2)
    plt.axvline(x=x_max, color ='#F6E800' , linewidth=2)
    plt.axhline(y=y_min, color = '#F6E800', linewidth=2)
    plt.axhline(y=y_max, color = '#F6E800', linewidth=2)

# Add inset for x-energy histogram
    ax_x = plt.gcf().add_axes((0.13, 0.8, 0.7, 0.1))  # [left, bottom, width, height]
    #ax_x.bar(range(len(x_energy_deposited)), x_energy_deposited, color='black', alpha=0.7)
    ax_x.fill_between(range(len(x_energy_deposited)), x_energy_deposited, color='black', alpha=0.5)
    #x_heatmap = np.expand_dims(x_energy_deposited, axis=0)  # Convert to 2D for imshow
    #ax_x.imshow(x_heatmap, aspect='auto', cmap='Greys', extent=[0, len(x_energy_deposited), 0, 1], norm=LogNorm(vmin=1, vmax=np.max(x_heatmap)))
    ax_x.set_xlim(x_min-50, x_max+50)
    #ax_x.set_yticks([])
    ticks = []
    ax_x.set_xticks(ticks)
    ax_x.set_yticks(ticks)
    ax_x.set_yscale('log')
    ax_x.set_ylabel('(keV)', fontsize = 15)
    #ax_x.set_title(f'Energy Deposition Map for Source: {source}')
    ax_x.grid(False)

# Add inset for y-energy histogram
    ax_y = plt.gcf().add_axes((0.83, 0.1, 0.1, 0.7))  # [left, bottom, width, height]
    #ax_y.barh(range(len(y_energy_deposited)), y_energy_deposited, color='black', alpha=0.7)
    ax_y.fill_between(y_energy_deposited, range(len(y_energy_deposited)), color='black', alpha=0.5)
    ax_y.set_ylim(y_min-50, y_max+50)
    ax_y.set_xticks(ticks)
    ax_y.set_yticks(ticks)
    ax_y.set_xscale('log')
    ax_y.set_xlabel('(keV)', fontsize = 15)
    #ax_y.set_title('Y-Axis Energy Distribution', rotation=90)

    plt.grid(False)

    plt.subplots_adjust(left = 1.1, right=2)
    plt.savefig(f"{resultFolder}/beam_image_{specLib.global_config.chip_dict[chip_id]}", bbox_inches='tight')
    #plt.show()
    plt.close()

    #plt.show()


    return 


if __name__ == '__main__':
    
    start_time = datetime.now()

    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    
    #Ask user for what config to use
    selected_config = configlib.initialconfig(parent_dir)
    
    #Load the configuration
    specLib.global_config = specLib.Config(selected_config)
    
    print('Chip configuration loaded:')
    with open(specLib.global_config.config_chips) as f:
        text = f.readlines()
        print(text)

    sources = specLib.global_config.sources
    sources_peaks = specLib.global_config.sources_peaks
    output_folder_base = specLib.global_config.output_folder
    input_folder = specLib.global_config.input_dir

   
    #chip = 'K10-W0060'
    chip = 'cdte'
    chip_id = specLib.get_chip_id(chip)
    
    print(f"At this stage we are using the chip {chip_id}, {chip}")
    
    # Ask User if its analysis of non-Polarized source or Polarized source

    #user_chips = configlib.query_user_chips() 
    #print(f"CHIPS TO USE: {user_chips}")

    #print(user_chips)

    list_max_dist = []

   # for source in sources:

   #     data_folder = os.path.join(input_folder, source)
   #     _ = None
   #     calib = Calibration(output_folder_base, _)

   # #    specLib.pre_process_source(source) # aslo needs to runs the multiplicity, maybe remove this and tell the user to run the pre_process.py script before running poarimetry.py 08/08/2025
   # #    specLib.process_event_multiplicity(source)

   #     output_folder = os.path.join(output_folder_base, source)

   #     source_peaks = specLib.global_config.sources_peaks

   #     try:
   #         peak_energy_list = source_peaks[source]['e0']
   #     except:
   #         print("ERROR: Please provide the source theoretical peak energy to perform polarimetry. Put it on the /config folder sources_database.json file")
   #         sys.exit(1)

   #     print(f"going to run source: {sources}")

   # 
   #     print("Performing beam image....")
   #     #try:
   #     perform_beam_img(output_folder_base, source, chip_id)
   #     #    print("Done....")
   #     #except Exception as e:
   #     #    print(f"WARNING: Did not perform beam energy, probably too little statistics. {e}")

   #     print("Localizing beam position....")
   #     try:
   #         x_max, x_min, y_max, y_min = find_beam_location(output_folder_base, source, chip_id)
   #         print("Done....")
   #     except Exception as e:
   #         print(f"WARNING: Was not able to localize the beam. {e}")
   #         user_awns_beam_pos = input(f"Do you want to give expected beam position, pixel number? y/n \n")
   #         if user_awns_beam_pos == 'y':
   #             x_min = int(input(" x_pix_min: "))
   #             x_max = int(input(" x_pix_max: "))
   #             y_min = int(input(" y_pix_min: "))
   #             y_max = int(input(" x_pix_max: "))
   #         else:
   #             print('Provide more statistics then... goodbye...')
   #             sys.exit(1)

   #     print("Computing max distance of Compton interaction")
   #     max_dist_computed = compute_max_dist(x_max, x_min, y_max, y_min, chip_id = chip_id) 
   #     print(f'ssssssssshit: {max_dist_computed}')
   #     print("Done....")

    #    list_max_dist.append(max_dist_computed)

    #print(list_max_dist)
    #print(f'min max dist = {min(list_max_dist)}')
    #minimum_maxdist = min(list_max_dist)

    

    #minimum_maxdist = 4.18 ## just running it once for 100kev, delete after

    print(f"going to run source: {sources}")

    for source in sources:

        print(source)
        
        energy = float(compton.get_energy_from_source_name(source))

        pol_type = compton.get_pol_type_from_source_name(source)
        if pol_type == 'NonPol':
            continue
        source_pol = source
        source_Nonpol = source.replace('Pol', 'NonPol')


        folder_input_polarimetry_pol = os.path.join(output_folder_base, source_pol)
        folder_input_polarimetry_Nonpol = os.path.join(output_folder_base, source_Nonpol)

        result_polarimetry_base = os.path.join(output_folder_base, 'result_polarimetry')
        pathlib.creat_dir(result_polarimetry_base)
        
        source_analysis = source.replace('Pol', '')

        result_polarimetry = os.path.join(result_polarimetry_base, source_analysis)
        pathlib.creat_dir(result_polarimetry)


        # Detectors geometry constants depending on HED geometry
        # CdTe
        z_cdte = -float((source.split("_")[-1]).split('c')[0]) #distance form source name, negative value
        cdte_matrix = int((source.split("_")[1]).split('x')[-1])   # cdte matrix from source name, ex:GaussBeamPol50keV_config4x4_0.5cm
        cdte_single_det_size = 1.6 # 1.6x1.6 cm2
        cdte_detSize = cdte_single_det_size * cdte_matrix # cm
        cdte_pixSize = 0.025 # cm
        # Si
        z_si = 0 # position of Si detector
        si_detSize = 6.656 # cm
        si_pixSize = 0.013 # cm
        #############################

        # Polarimetry constants
        min_dist_list = [0.025, 0.05, 0.075]
        angle_bin_list = [1, 5, 10, 15, 36]  #bin size for polarimetry
        max_dist_list = [100000]  # max dist between compton events, cm
        max_dist_list = np.round(max_dist_list, 3)
        max_dist_on_list = max_dist_list[-1]
        #############################


        calib = Calibration(output_folder_base, None) # useless remove this feature, past artfact...
        
        # Identify single, double and multiple, identify event multiplicity (spacial cluster)
        specLib.pre_process_source(source_pol) 
        specLib.pre_process_source(source_Nonpol)

        specLib.process_event_multiplicity(source_pol) # writes multiplicity .parquets
        specLib.process_event_multiplicity(source_Nonpol) # writes multiplicity .parquets

        compton.identify_compton(source_pol, energy, z_cdte, cdte_detSize, cdte_pixSize, z_si, si_detSize, si_pixSize)
        compton.identify_compton(source_Nonpol, energy, z_cdte, cdte_detSize, cdte_pixSize, z_si, si_detSize, si_pixSize)
       
        print('Counting events.... wait')
        compton.count_nEvents_allTypes(folder_input_polarimetry_pol, energy, chip)
        compton.count_nEvents_allTypes(folder_input_polarimetry_Nonpol, energy, chip)

        polarimetry_task = [(folder_input_polarimetry_pol, folder_input_polarimetry_Nonpol, result_polarimetry, min_dist, angle_bin, energy, max_dist, z_cdte, z_si, cdte_detSize, si_detSize) for min_dist, angle_bin, max_dist in product(min_dist_list, angle_bin_list, max_dist_list)]
        #with Pool() as pool:
        #    for _ in tqdm(pool.imap_unordered(compton.polarimetry_task, polarimetry_task), total=len(polarimetry_task), desc='Compton Polarimetry'):
        #        pass
        for task in tqdm(polarimetry_task,
                         total=len(polarimetry_task),
                         desc='Compton Polarimetry'):
            compton.polarimetry_task(task)
        #max_merit, best_min_dist, best_angle_bin, sigma_max_merit = compton.plot_figureMeritMap(output_folder, min_dist_list, angle_bin_list, max_dist_list)
        
        #print(f'best_min_dist: {best_min_dist}')
        #print(f'best_angle_bin: {best_angle_bin}')
        #print(f'max_merit: {max_merit}')


    #compton.plot_rotationMeasurements(output_folder_base, sources, [0.55], [1])

    print('') 
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    end_time = datetime.now()
    print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Calculate total run time
    total_time = end_time - start_time
    print(f"Total Run Time: {total_time}")



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import compton_equations
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
import random
import math
from tqdm import tqdm
import time
import os

def uniform_distribute_energy(matrix, center, energy, radius):
    """Distribute energy in a circular cluster around the center with a normal distribution."""
    cx, cy = center
    size = matrix.shape[0]
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    circle = np.sqrt((x-cx)**2 + (y-cy)**2)
    list_radius = []
    list_radius_x = []
    list_radius_y = []
    _pixelCount = 0
    total_energy = 0

    for i in range(0,256,1):
        for j in range(0,256,1):
            if circle[i,j] <= radius:
                """ Get x,y coordinates of the circle """
                list_radius.append(circle[i,j])
                list_radius_x.append(j)
                list_radius_y.append(i)
                _pixelCount += 1

    for i in range(_pixelCount):
        matrix[list_radius_x[i],list_radius_y[i]] = energy/_pixelCount
        total_energy += (energy/_pixelCount)

    #print(_pixelCount)
    #print(total_energy)

def generate_random_weights(count, min_value=0):
    # Create random values between 0 and 1
    random_weights = np.random.random(count)
    adjusted_weights = random_weights * (1 - count * min_value) + min_value
    normalized_weights = adjusted_weights / np.sum(adjusted_weights)
    #print(f"normalized_wheights: {normalized_weights}")
    return normalized_weights  
    

def random_distribute_energy(matrix, center, energy, radius):
    """Distribute energy in a circular cluster around the center with a normal distribution."""
    cx, cy = center
    size = matrix.shape[0]
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    circle = np.sqrt((x-cx)**2 + (y-cy)**2)

    list_radius_x = []
    list_radius_y = []

    _pixelCount = 0

    #print("\n")

    for i in range(0,256,1):
        for j in range(0,256,1):
            if circle[i,j] <= radius:
                """ Get x,y coordinates of the circle """
                #print(f"radius = {radius}")
                #print(f"circle[i,j] = {circle[i,j]}")

                list_radius_x.append(j)
                list_radius_y.append(i)
                _pixelCount += 1

    #print(f"list_radius_x: {list_radius_x}")
    #print(f"list_radius_y: {list_radius_y}")

    
    normalized_weights = generate_random_weights(_pixelCount)     

    #print(normalized_weights)

    energy_distribution = normalized_weights * energy

    #print(f'energy_distribution = {energy_distribution}')

    hit_pixels = []

    for i in range(_pixelCount):
        x = list_radius_x[i]
        y = list_radius_y[i]
        matrix[y, x] += energy_distribution[i]
        hit_pixels.append((x, y, energy_distribution[i]))
    
    #print(f'hit_pixels = {hit_pixels}')

    return hit_pixels

def dect_config(conf) -> None:
    if conf == "single":
        h_pixels = 256
        v_pixels = 256
        matrix = np.zeros((h_pixels, v_pixels), dtype=float)
        
    elif conf == "quad":
        h_pixels = 1024
        v_pixels = 256
        matrix = np.zeros((h_pixels, v_pixels), dtype=float)

    else:
        raise Exception("No valid config. Available configs: 'single' 'quad'")
    return matrix, h_pixels, v_pixels

def get_random_photon(MeanEnergy, sigma):
    Energy_photon = np.random.normal(loc=MeanEnergy, scale=sigma)

    while Energy_photon <= 30: # 30keV min photon energy
        Energy_photon = np.random.normal(loc=MeanEnergy, scale=sigma)

    random_theta = np.random.randint(40,180) # not sure about this range
    return Energy_photon, random_theta

def compute_E_electron(E_incoming_photon, theta):
    E_electron = round(compton_equations.get_energy_electron(E_incoming_photon, theta),1)
    return E_electron

def compute_E_photon(E_incoming_photon,theta):
    E_photon = round(compton_equations.compton_photon(E_incoming_photon, theta),1)
    return E_photon

def get_cluster_centerPos(h_pixels, v_pixels, electron_cluster_size, photon_cluster_size):
    """Avoids placing clusters near the borders and ensures no cluster overlap."""
    
    border_limit = 15
    max_x_pos = h_pixels - border_limit
    max_y_pos = v_pixels - border_limit

    def get_valid_position():
        pos = np.random.choice(h_pixels * v_pixels, size=1, replace=False)
        x_pos, y_pos = divmod(pos[0], h_pixels)
        while not (border_limit <= x_pos <= max_x_pos and border_limit <= y_pos <= max_y_pos):
            pos = np.random.choice(h_pixels * v_pixels, size=1, replace=False)
            x_pos, y_pos = divmod(pos[0], h_pixels)
        return x_pos, y_pos

    elect_xPos, elect_yPos = get_valid_position()
    photon_xPos, photon_yPos = get_valid_position()

    min_distance_between_clusters = 3  # Ensure clusters don't overlap
    cluster_size = max(electron_cluster_size, photon_cluster_size)
    min_center_distance = 2 * cluster_size + min_distance_between_clusters

    while np.sqrt((elect_xPos - photon_xPos) ** 2 + (elect_yPos - photon_yPos) ** 2) < min_center_distance:
        photon_xPos, photon_yPos = get_valid_position()

    return (elect_xPos, elect_yPos), (photon_xPos, photon_yPos)


def determine_cluster_size(MeanEnergy, sigma, eventEnergy) -> 2:

    energy_range = 1000

    if eventEnergy > energy_range:
        energy_range = eventEnergy
    
    #print(f"EventEnergy: {eventEnergy} keV")
    energy_parameter = math.log(eventEnergy)/math.log(energy_range)
    energy_parameter = eventEnergy/energy_range
    max_cluster_size = 7
    cluster_size = max_cluster_size * energy_parameter

    return cluster_size

def main(config, interactions, MeanEnergy, sigma):
    matrix, h_pixels, v_pixels = dect_config(config)

    Index = 0
    ToA = 0
    counter = 0

    if not os.path.exists("./outputs"):
    # If it doesn't exist, create it
        os.makedirs("./outputs")

    output_file_name = f"compton_generator_{interactions}Comptons_{MeanEnergy}keV-{sigma}sigma.t3pa"

    output_file = os.path.join("outputs", output_file_name)

    with open(output_file, 'w') as f:
        f.write("Index MatrixIndex ToA ToT(keV) FToA Overflow Ephoton Eelectron theta event\n")

        start_time = time.time()  

        for i in tqdm(range(interactions), desc="Processing", unit="compton"):
            #print("\n")
            ToA_step = int(round(np.random.normal(loc=10**4, scale=50), 0)) # clock cycles
            min_ToA_step = 10**3 #clock cycles

            counter += 1

            if ToA_step <= min_ToA_step:
                ToA_step = min_ToA_step

            electron_dist = []
            photon_dist = []

            E_incoming_photon, random_theta = get_random_photon(MeanEnergy, sigma)
            E_electron = compute_E_electron(E_incoming_photon, random_theta)
            E_compton_photon = compute_E_photon(E_incoming_photon, random_theta)

            electron_cluster_size = determine_cluster_size(MeanEnergy, sigma, E_electron)
            photon_cluster_size = determine_cluster_size(MeanEnergy, sigma, E_compton_photon)

            electron_position, photon_position = get_cluster_centerPos(h_pixels, v_pixels, electron_cluster_size, photon_cluster_size)
            Cluster_elect_xPos, Cluster_elect_yPos = electron_position
            Cluster_photon_xPos, Cluster_photon_yPos = photon_position

            #print(f'ex: {Cluster_elect_xPos} ey: {Cluster_elect_yPos}')
            #print(f'phx: {Cluster_photon_xPos} phy: {Cluster_photon_yPos}')

            electron_dist = random_distribute_energy(matrix, (Cluster_elect_xPos, Cluster_elect_yPos), E_electron, electron_cluster_size)
            photon_dist = random_distribute_energy(matrix, (Cluster_photon_xPos, Cluster_photon_yPos), E_compton_photon, photon_cluster_size)

            ToA = ToA + random.randrange(min_ToA_step,ToA_step)
            ToA_wiggleRoom = 2  # clocks
            
            for hit in electron_dist:
                x, y, energy = hit
                matrix_index = y * h_pixels + x  
                Index += 1
                f.write(f"{Index} {matrix_index} {ToA + random.randrange(ToA_wiggleRoom)} {round(energy, 1)} {random.randrange(10)} 0 {E_compton_photon} {E_electron} {random_theta} e\n")

            for hit in photon_dist:
                x, y, energy = hit
                matrix_index = y * h_pixels + x  
                Index += 1
                f.write(f"{Index} {matrix_index} {ToA + random.randrange(ToA_wiggleRoom)} {round(energy, 1)} {random.randrange(10)} 0 {E_compton_photon} {E_electron} {random_theta} ph\n")

            

        end_time = time.time()
        elapsed_time = end_time - start_time

    current_directory = os.getcwd()

    outFile_dir = os.path.join(current_directory, output_file)

    print(f"Data saved on {outFile_dir} ")
    
    # Plotting the matrix
    plt.imshow(matrix, cmap='nipy_spectral', interpolation='none', origin='lower', norm='linear')
    plt.colorbar(label='keV')
    plt.title('256 x 256 Matrix with Random Values')
    plt.show()

     

if __name__ == '__main__':

    config = "single" # at this stage we can only use single condiguration
    print("The script generates photons with a gaussian energy distribuiton (Eph>30keV). Plese input the parameters.")
    minEnergy = int(input("Input mean energy incoming photon Energy(keV): \n>> "))
    maxEnergy = int(input("Input sigma (keV): \n>> "))
    interactions = int(input("Input nº of Comptons to generate: \n>> "))

    main(config, interactions, minEnergy, maxEnergy)

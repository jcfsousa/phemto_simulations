from matplotlib.cm import colors
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def create_scatter_plot(data, x_wheighed, y_wheighed):
    grouped_data = data.groupby(['X', 'Y'])['ToT (keV)'].sum().reset_index()
    x, y, intensity = grouped_data['X'], grouped_data['Y'], grouped_data['ToT (keV)']
  
    intensity_matrix = np.zeros((256, 1024))  # Adjust dimensions if needed

    for xi, yi, inten in zip(x, y, intensity):
        intensity_matrix[yi - 1, xi - 1] = inten 

    plt.figure(figsize=(10, 10))
    img = plt.imshow(intensity_matrix, cmap='jet', norm=mcolors.LogNorm(), origin='lower')
    #scatter = plt.scatter(x, y, c=intensity, cmap='Reds', s=2, norm=mcolors.LogNorm())
    cbar = plt.colorbar(img, label='Energy (keV)')
    cbar.set_label('Energy (keV)')

    x_min = x.min()-3
    x_max = x.max()+1

    y_min = y.min()-3
    y_max = y.max()+1

    plt.vlines(x_wheighed, y_min, y_max, colors = 'red', linestyles='solid')
    plt.hlines(y_wheighed, x_min, x_max, colors = 'red', linestyles='solid')
    
    cluster_energy = data['ToT (keV)'].sum()
    cluster_id = data['Cluster'].max()
    event_id = data['Event'].max()

    plt.title(f'Event ID: {event_id} Cluster: {cluster_id} Energy:{cluster_energy} keV')
    # Set labels and title
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    #plt.xlim(x_min, x_max)
    #plt.ylim(y_min, y_max)
        
    #plt.show()
    
    del data

    
def barycenter(cluster_df):
    cluster_energy = cluster_df['ToT (keV)'].sum()

    n_collumns_df = len(cluster_df.axes[1])

    cluster_df.insert(n_collumns_df, 'X_wheight', (cluster_df['X']*cluster_df['ToT (keV)'])/cluster_energy)
    cluster_df.insert(n_collumns_df, 'Y_wheight', (cluster_df['Y']*cluster_df['ToT (keV)'])/cluster_energy)
    
    print(f'Cluster Energy: {cluster_energy} keV')
    print(f'n_collumns_df: {n_collumns_df}')

    print(f'cluster_df with wheighed X,Y:\n {cluster_df}')

    x_barycenter = cluster_df['X_wheight'].sum() - 1
    y_barycenter = cluster_df['Y_wheight'].sum() - 1

    create_scatter_plot(cluster_df, x_barycenter, y_barycenter)

    print(f'x_barycenter = {x_barycenter}')
    print(f'y_barycenter = {y_barycenter}')

    return x_barycenter, y_barycenter


def energies_data_frames_or_dict(data, dict = False):
    """
    This function determines the energy of each cluster, E1 and E2, where E1 is the most energetic cluster and       the E2 is the least energenic cluster on the event. 
    The function also tries to determine the pixel where the interaction happened.

    """

    # Initialize dictionaries to store the E1 and E2 for each event
    E1 = {}
    E2 = {}

    # Iterate over each event
    for event, group_df in data:
        # Group by 'Cluster' and sum the 'Log_energy' for each cluster
        cluster_energy_sums = group_df.groupby('Cluster')['ToT (keV)'].sum()
        
        # Identify the highest and lowest summed energy clusters
        highest_energy_cluster = cluster_energy_sums.idxmax()
        lowest_energy_cluster = cluster_energy_sums.idxmin()

        highest_energy = cluster_energy_sums.max()
        lowest_energy = cluster_energy_sums.min()

        # Get the most energetic hit's coordinates for the highest and lowest energy clusters
        highest_energy_center_hit = group_df[group_df['Cluster'] == highest_energy_cluster].sort_values('ToT (keV)', ascending=False).iloc[0]
        lowest_energy_center_hit = group_df[group_df['Cluster'] == lowest_energy_cluster].sort_values('ToT (keV)', ascending=False).iloc[0]
       
        data_cluster_high_energy = group_df[group_df['Cluster'] == highest_energy_cluster]
        data_cluster_low_energy = group_df[group_df['Cluster'] == lowest_energy_cluster]
        print("------------------")
        print(f'selected cluster 1:\n {data_cluster_high_energy}')
        print(f'selected cluster 2:\n {data_cluster_low_energy}')

       # print(f'event = {event}')
      #  print(f'group_df =\n{group_df}')
     #   print(f'highest_energy_center_hit = {highest_energy_center_hit}')
    #    print(f'lowest_energy_center_hit = {lowest_energy_center_hit}')

        
        x_highest_energy_cluster, y_highest_energy_cluster = barycenter(data_cluster_high_energy)
        x_lowest_energy_cluster, y_lowest_energy_cluster = barycenter(data_cluster_low_energy)

        print("------------------")

        input("press ENTER to continue")





    E1_df = pd.DataFrame.from_dict(E1, orient='index').reset_index().rename(columns={'index': 'Event'})
    E2_df = pd.DataFrame.from_dict(E2, orient='index').reset_index().rename(columns={'index': 'Event'})

    if dict:
        return E1, E2
    else:
        return E1_df, E2_df



if __name__ == '__main__':
    
    double_events_in_peak_df = pd.read_csv('double_events_in_peak_df.csv')
    grouped_double_events_in_peak_df = double_events_in_peak_df.groupby('Event')
        
    E1, E2 = energies_data_frames_or_dict(grouped_double_events_in_peak_df, dict=True)[0],energies_data_frames_or_dict(grouped_double_events_in_peak_df, dict=True)[1]


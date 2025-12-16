import os, subprocess
from matplotlib.colors import LogNorm
from lmfit import Model
import gc
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import math
from scipy.linalg import polar
from scipy.optimize import curve_fit
from calibration.calibration import Calibration as calib
from manalysis import specLib
import manalysis.pathlib as pathlib
import manalysis.comptons as compton
import aquisition.tpx_analysis3 as tpx_analysis

# Define your custom function using numpy
def polarimetry_fit(phi, A, B, C):
    #return A + B * np.sin(np.radians(phi) - C)**2
    return A * np.cos(2*(np.radians(phi) - B)) + C

def fit_binned_counts(data, angle_bin):
    
    # Prepare data for fitting
    phi = data['Phi_bin'].astype(float).values
    counts_norm = data['Counts_Norm'].values
    error = data['Error'].values

    # Use curve_fit to fit the custom function to your data
    popt, pcov = curve_fit(polarimetry_fit, phi, counts_norm)
        
    # Extract the standard deviations (errors) of the fitted parameters
    perr = np.sqrt(np.diag(pcov))
    #popt[1] = math.degrees(popt[1])

    n_div = round((360/angle_bin) * 100)

    try:
        chi2 = tpx_analysis.chi2_red(counts_norm, polarimetry_fit(phi, *popt), error, len(phi)-len(popt))
    except (RuntimeError, RuntimeWarning, ValueError):
        chi2 = 0

    # Generate fitted data
    phi_fit = np.linspace(0, 360, n_div)
    counts_fit = polarimetry_fit(phi_fit, *popt)
    del data
    return phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error


def polarimetry_fit_residual(phi, A, B, C, D, E):
    #return A + B * np.sin(np.radians(phi) - C)**2
    #return A + B * (np.sin(np.radians(phi)-phi_c))**2 + C * (np.sin(2*(np.radians(phi)-phi_s)))**2
    return A*np.cos(2*(np.radians(phi) - B)) + C *np.cos(4*(np.radians(phi) - D)) + E

def fit_binned_counts_residual(data, angle_bin):
    
    # Prepare data for fitting
    phi = data['Phi_bin'].astype(float).values
    counts_norm = data['Counts_Norm'].values
    error = data['Error'].values
    
    lower_bounds = [0.00001, 0, 0.000001, 0, 0.00001]  
    upper_bounds = [10, 360, 10, 361, 10] 

    popt, pcov = curve_fit(polarimetry_fit_residual, phi, counts_norm, bounds=(lower_bounds, upper_bounds))
    perr = np.sqrt(np.diag(pcov))

    phi = np.array(phi)
    print(phi)
    ###
    #gmodel = Model(polarimetry_fit_residual)
    #params = gmodel.make_params(A=1, B=180, C=0, D=180, E=1)  # Initial guesses

# Set bounds:
    #params['A'].set(min=0, max=10)
    #params['B'].set(min=0, max=361)
    #params['C'].set(min=0, max=10)
    #params['D'].set(min=0, max=361)
    #params['E'].set(min=0, max=10)
    #result = gmodel.fit(data['Counts_Norm'], params, phi = data['Phi_bin'])
        
    #popt = list(result.best_values.values())
    #perr = np.sqrt(np.diag(result.covar)) if result.covar is not None else [None] * len(popt)
    ###
        
 
    print(f'popt: {popt}')
    print(f'perr: {perr}')
    #print(f'pcov: {pcov}')

    n_div = round((360/angle_bin) * 100)
    
    try:
        chi2 = tpx_analysis.chi2_red(counts_norm, polarimetry_fit_residual(phi, *popt), error, len(phi)-len(popt))
    except (RuntimeError, RuntimeWarning, ValueError):
        chi2 = 0
    #print(f'chi2: {chi2}')

    # Generate fitted data
    phi_fit = np.linspace(0, 360, n_div)
    counts_fit = polarimetry_fit_residual(phi_fit, *popt)
    del data
    return phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error

def plot_fit_binned_counts_polar(outputFolder, data, phi_fit, counts_fit, popt, perr, counts_norm, phi, energy, angle_bin, min_dist, chi2, max_dist = None, multiple_plots = False, rot_list = None):
 
    outputFolder_polarimetry = os.path.join(outputFolder, 'photonPolarimetry')
    pathlib.creat_dir(outputFolder_polarimetry)
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    source_energy = compton.get_energy_from_source_name(outputFolder)

    outputFolder_polarimetry_binDist = os.path.join(outputFolder_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')

    if multiple_plots:
        fig, axs = plt.subplots(len(data), 1, figsize=(10, 5 * len(data)), subplot_kw={'projection': 'polar'})
        fig.suptitle(f'{source_energy} keV')
        fig.tight_layout()  # Adjust spacing between subplots

        if len(data) == 1:
            axs = [axs]

        for i, (data_subset, phi_fit_subset, counts_fit_subset, counts_norm_subset, phi_subset, popt_subset, perr_subset, chi2_subset) in enumerate(zip(data, phi_fit, counts_fit, counts_norm, phi, popt, perr, chi2)):

            phi_radians_fit = np.deg2rad(phi_fit_subset)
            phi_radians = np.deg2rad(phi_subset)

            axs[i].bar(phi_radians, counts_norm_subset, width=np.deg2rad(angle_bin), color='firebrick', edgecolor='firebrick', alpha = 0.9, linewidth = 1)
            axs[i].grid(True, alpha=0.9)

            # Plot fit curve
            axs[i].plot(
                phi_radians_fit, counts_fit_subset, color = 'k',
                label=(
                    r"$N(\phi) = A \cdot \cos2(\phi - B) + C$"
                    f"\nA = {popt_subset[0]:.4f} ± {perr_subset[0]:.4f}"
                    f"\nB = {math.degrees(popt_subset[1]):.2f}° ± {math.degrees(perr_subset[1]):.2f}°"
                    f"\nC = {popt_subset[2]:.4f} ± {perr_subset[2]:.4f}\n"
                    r"$\chi^2_{red} = $" + f"${chi2_subset:.2f}$",
                ),
                linestyle = '-',
                linewidth = 3)
   
            axs[i].set_theta_zero_location('E')  # Set 0 degrees to the right
            axs[i].set_theta_direction(1)  # Set the direction of increasing avngles (clockwise)

            # Set radial ticks (intensity scale) along the 90º line
            max_count_norm = data_subset['Counts_Norm'].max()
            radial_ticks = np.linspace(0, max_count_norm, 0)
            axs[i].set_rticks(radial_ticks)  # Set the number of ticks
            axs[i].set_yticklabels([f'{tick:.2f}' for tick in radial_ticks], verticalalignment='top')  # Place tick labels
            axs[i].legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)

            plt.subplots_adjust(hspace=0)

        axs[-1].set_xlabel(r'Azimuthal Angle, $\phi$ (°)')

    else:
        phi_radians_fit = np.deg2rad(phi_fit)

        # Plotting
        plt.figure(figsize=(10,6))
        ax = plt.subplot(111, polar=True)
        bars = ax.bar(np.deg2rad(phi), counts_norm, width=np.deg2rad(angle_bin), color='firebrick', edgecolor='firebrick', linewidth = 1, alpha = 0.9)
        #ax.plot(np.deg2rad(phi), counts_norm, drawstyle = 'steps-mid', color = 'k', linewidth = 2)
        ax.grid(True, alpha=0.9)

        # Plot the fitted curve
        ax.plot(phi_radians_fit, counts_fit, color='k', label=
                r"$N(\phi) = A \cdot \cos2(\phi - B) + C$"
                f"\nA = {popt[0]:.4f} ± {perr[0]:.4f}"
                f"\nB = {math.degrees(popt[1]):.2f}° ± {math.degrees(perr[1]):.2f}°"
                f"\nC = {popt[2]:.4f} ± {perr[2]:.4f}\n"
                r"$\chi^2_{red} = $" + f"${chi2:.2f}$",
                linestyle = '-',
                linewidth = 3
                )

                
        # Customize the plot to have 0 degrees to the right
        ax.set_theta_zero_location('E')  # Set 0 degrees to the right
        ax.set_theta_direction(1)  # Set the direction of increasing avngles (clockwise)

        # Set radial ticks (intensity scale) along the 90º line
        max_count_norm = data['Counts_Norm'].max()
        radial_ticks = np.linspace(0, max_count_norm, 0)
        ax.set_rticks(radial_ticks)  # Set the number of ticks
        ax.set_yticklabels([f'{tick:.2f}' for tick in radial_ticks], verticalalignment='top')  # Place tick labels


        ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.tight_layout()


    if multiple_plots:
        path_components = outputFolder.split(os.sep)
        root_dir = os.sep.join(path_components[:-1])
        grenoble_conlcusions_folder = f'{root_dir}/3-GrenobleGeneralConclusions'
        plt.savefig(f'{grenoble_conlcusions_folder}/polarimetry_{source_energy}kev_examplePOLAR.png', bbox_inches='tight')
    else:
        plt.savefig(f'{outputFolder}/{energy}keV_BESTPOLAR_{angle_bin_str}bin_md{min_dist_str}.png', bbox_inches='tight')
    plt.close()

    del data

def plot_fit_binned_counts(outputFolder, data, phi_fit, counts_fit, popt, perr, counts_norm, phi, energy, angle_bin, min_dist, chi2, max_dist = None, multiple_plots = False, rot_list = None):
 
    #outputFolder_polarimetry = os.path.join(outputFolder, 'photonPolarimetry')
    #specLib.creat_dir(outputFolder_polarimetry)
    #angle_bin_str = str(angle_bin).replace('.','-')
    #min_dist_str = str(min_dist).replace('.','-')
    #max_dist_str = str(max_dist).replace('.','-')

    #outputFolder_polarimetry_binDist = os.path.join(outputFolder_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')

    ## Plot the data and the fit
    #plt.figure(figsize=(10, 6))
    #plt.errorbar(data['Phi_bin'], data['Counts_Norm'], yerr=data['Error'], fmt='none', color='lightgray', ecolor='black', capsize=3, zorder = 1)
    ##plt.plot(data['Phi_bin'], data['Counts_Norm'], drawstyle='steps-mid', color='k' )
    #plt.scatter(phi, counts_norm, color='red', edgecolor='black', label='Data', zorder = 2)
    #plt.plot(
    #    phi_fit, counts_fit,
    #    linewidth = 3,
    #    label=(
    #        r"$N(\phi) = A \cdot \cos2(\phi - B) + C$"
    #        f"\nA = {popt[0]:.2f} ± {perr[0]:.2f}"
    #        f"\nB = {math.degrees(popt[1]):.2f}° ± {math.degrees(perr[1]):.2f}°"
    #        f"\nC = {popt[2]:.2f} ± {perr[2]:.2f}\n"
    #        r"$\chi^2_{red} = $" + f"${chi2:.2f}$"
    #        ),
    #    color='blue',
    #    zorder = 3
    #)
    #plt.xlabel(r'$\phi$ (°)')
    #plt.ylabel('Normalized Counts')    
    #plt.legend(loc="lower right")
    #plt.grid(False)
    #plt.tight_layout()
    ##plt.title(f'{energy} {angle_bin} Comptons Radial Bin Count ({angle_bin}° BinsCounts)')
    #plt.show()
    ##plt.savefig(f'{outputFolder_polarimetry_binDist}/{energy}keV_Comptons_Bin_Count_{angle_bin_str}bin_md{min_dist_str}')

    #plt.close()

    #del data

    outputFolder_polarimetry = os.path.join(outputFolder, 'photonPolarimetry')
    pathlib.creat_dir(outputFolder_polarimetry)
    angle_bin_str = str(angle_bin).replace('.', '-')
    min_dist_str = str(min_dist).replace('.', '-')
    max_dist_str = str(max_dist).replace('.', '-')

    source_energy = compton.get_energy_from_source_name(outputFolder)

    outputFolder_polarimetry_binDist = os.path.join(outputFolder_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')

    if multiple_plots:
        # Create subplots stacked vertically
        fig, axs = plt.subplots(len(data), 1, figsize=(10, 5 * len(data)), sharex=True)
        fig.suptitle(f'{source_energy} keV')
        fig.tight_layout()  # Adjust spacing between subplots

        # If only one subplot is created, axs will not be a list, so we convert it to a list
        if len(data) == 1:
            axs = [axs]

        for i, (data_subset, phi_fit_subset, counts_fit_subset, counts_norm_subset, phi_subset, popt_subset, perr_subset, chi2_subset) in enumerate(zip(data, phi_fit, counts_fit, counts_norm, phi, popt, perr, chi2)):
            # Plot error bars
            axs[i].errorbar(data_subset['Phi_bin'], data_subset['Counts_Norm'], yerr=data_subset['Error'], fmt='none', color='lightgray', ecolor='black', capsize=3, zorder=1)

            # Plot scatter points
            axs[i].scatter(phi_subset, counts_norm_subset, color='red', edgecolor='black', label='Data', zorder=2)

            # Plot fit curve
            axs[i].plot(
                phi_fit_subset, counts_fit_subset,
                linewidth=3,
                label=(
                    r"$N(\phi) = A \cdot \cos2(\phi - B) + C$"
                    + fr"\n$A = {popt_subset[0]:.4f} \pm {perr_subset[0]:.4f}$"
                    + fr"\n$B = {math.degrees(popt_subset[1]):.2f}^\circ \pm {math.degrees(perr_subset[1]):.2f}^\circ$"
                    + fr"\n$C = {popt_subset[2]:.4f} \pm {perr_subset[2]:.4f}$"
                    + fr"\n$\chi^2_{{red}} = {chi2_subset:.2f}$"
                ),
                color='blue',
                zorder=3
            )
            #axs[i].annotate(xy=[300,1.75], xytext=[315,1.65], text=f'Detector \nPhysical \nRotation: {rot_list[i]}°', fontsize=14)

            axs[i].annotate(xy=[300,1.75], xytext=[315,1.65], text=fr'$\Theta_{{rot}}$ = {round(rot_list[i],0)}°', fontsize=16)

            axs[i].legend(loc="lower right")
            axs[i].minorticks_on()
            axs[i].tick_params(axis='both', which='both', top=True, bottom=True, right=True)
            axs[i].grid(False)
            axs[i].set_ylim(0,2)

            plt.subplots_adjust(hspace=0)

        
        axs[-1].minorticks_on()
        axs[-1].tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True)
        axs[int(round(i/2,0))].set_ylabel('Normalized Counts')
        axs[-1].set_xlabel(r'Azimuthal Angle, $\phi$ (°)')

    else:
        # Single plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(data['Phi_bin'], data['Counts_Norm'], yerr=data['Error'], fmt='none', color='lightgray', ecolor='black', capsize=3, zorder=1)
        plt.scatter(phi, counts_norm, color='red', edgecolor='black', label='Data', zorder=2)
        plt.plot(
            phi_fit, counts_fit,
            linewidth=3,
            label=(
                fr"$N$($\phi$) = $A$ $\cdot$ $\cos$2($\phi$ - $B$) + $C$" + fr"\\$A$ = {popt[0]:.4f} $\pm$ {perr[0]:.4f}"
                + fr"\\$B$ = {math.degrees(popt[1]):.2f}$^\circ$ $\pm$ {math.degrees(perr[1]):.2f}$^\circ$"
                + fr"\\$C$ = {popt[2]:.4f} $\pm$ {perr[2]:.4f}"
                + fr"\\$\chi^2_{{red}}$ = {chi2:.2f}"
            ),
            color='blue',
            zorder=3
        )
        plt.ylim(0,2)
        plt.xlabel(r'Azimuthal Angle, $\phi$ (°)')
        plt.ylabel('Normalized Counts')
        plt.title(f'{source_energy} keV')
        plt.minorticks_on()
        plt.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True)
        plt.legend(loc="lower right", fontsize = 15)
        plt.grid(False)
        plt.tight_layout()

    # Save or show the plot
    if multiple_plots:
        path_components = outputFolder.split(os.sep)
        root_dir = os.sep.join(path_components[:-1])
        grenoble_conlcusions_folder = f'{root_dir}/3-GrenobleGeneralConclusions'
        plt.savefig(f'{grenoble_conlcusions_folder}/polarimetry_{source_energy}kev_example.png', bbox_inches='tight')
    else:
        plt.savefig(f'{outputFolder}/{energy}keV_BEST_{angle_bin_str}bin_md{min_dist_str}.png')
    plt.close()

    del data


def plot_fit_binned_counts_residual_polar(sourceFolder, data, phi_fit, counts_fit, popt, perr, counts_norm, phi, energy, angle_bin, min_dist, chi2):
 
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')

    # Convert phi_fit to radians for polar plot
    phi_radians_fit = np.deg2rad(phi_fit)

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(np.deg2rad(phi), counts_norm, width=np.deg2rad(angle_bin), color='firebrick', alpha = 0.9, edgecolor='firebrick', linewidth=1)
    ax.grid(True, alpha=0.9)

    # Plot the fitted curve
    ax.plot(phi_radians_fit, counts_fit, color='black', label=(
        r"$N(\phi) = A \cdot cos[2(\phi - B)] + C \cdot cos[4(\phi -D)] + E$"
        f"\\A = {popt[0]:.4f} ± {perr[0]:.4f}"
        f"\\B = {popt[1]:.2f}°± {perr[1]:.2f}°"
        f"\\C = {popt[2]:.4f} ± {perr[2]:.4f}"
        f"\\" + r"D =" + f" {popt[3]:.2f}°± {perr[3]:.2f}°"
        f"\\" + r"E =" + f" {popt[4]:.4f}± {perr[4]:.4f}\n"
        r"$\chi^2_{red} = $" + f"$ {chi2:.2f}$"
    ))  

            
    # Customize the plot to have 0 degrees to the right
    ax.set_theta_zero_location('E')  # Set 0 degrees to the right
    ax.set_theta_direction(1)  # Set the direction of increasing avngles (clockwise)

    # Set radial ticks (intensity scale) along the 90º line
    max_count_norm = data['Counts_Norm'].max()
    radial_ticks = np.linspace(0, max_count_norm, 0)
    ax.set_rticks(radial_ticks)  # Set the number of ticks
    ax.set_yticklabels([f'{tick:.2f}' for tick in radial_ticks], verticalalignment='top')  # Place tick labels

    ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
    plt.tight_layout()
    #plt.title(f'{energy}keV Events Radial Bin Count ({angle_bin_str}° BinsCounts)')
    plt.savefig(f'{sourceFolder}/residualPolarimetry_{angle_bin_str}bin_md{min_dist_str}-POLAR.png', bbox_inches='tight' )
    #plt.savefig(f'{outputFolder_polarimetry_binDist}/{energy}keV_Events_Radial_Bin_Count_{angle_bin_str}bin_md{min_dist_str}-residual')
    plt.close()
    
    gc.collect()
    del data, phi_fit, counts_fit, popt, perr, counts_norm, phi


def plot_fit_binned_counts_residual(sourceFolder, data, phi_fit, counts_fit, popt, perr, counts_norm, phi, energy, angle_bin, min_dist, chi2):
 
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')

    # Plot the data and the fit
    plt.figure(figsize=(10, 6))
    plt.errorbar(data['Phi_bin'], data['Counts_Norm'], yerr=data['Error'], fmt='none', color='lightgray', ecolor='black', capsize=3, zorder=1)
    plt.scatter(phi, counts_norm, color='red', edgecolor='black', label='Data', zorder=2)
    plt.plot(
        phi_fit, counts_fit,
        label=(
            r"$N(\phi) = A \cdot cos[2(\phi - B)] + C \cdot cos[4(\phi -D)] + E$"
            f"\nA = {popt[0]:.4f} ± {perr[0]:.4f}"
            f"  , B = {popt[1]:.2f}° ± {perr[1]:.2f}°"
            f"  , C = {popt[2]:.4f} ± {perr[2]:.4f}"
            f"\n " + r"D =" + f" {popt[3]:.2f}° ± {perr[3]:.2f}°"
            f" , " + r"E =" + f" {popt[4]:.4f} ± {perr[4]:.4f}\n"
            r"$\chi^2_{red} = $" + f"$ {chi2:.2f}$"
            ),
        color='blue', linewidth=3, zorder=3
    )
    plt.xlabel(r'Azimuthal Angle, $\phi$ (°)')
    plt.ylabel('Normalized Counts')    
    #plt.legend(loc="lower right")
    plt.legend(
        loc='upper center',          
        ncol=2,                                
        fontsize = 15
    )

    plt.ylim(0.7, 1.5)
    plt.minorticks_on()
    plt.tick_params(direction="in", axis='both', which='both', top=True, bottom=True, right=True)
    plt.grid(False)
    plt.tight_layout()
    #plt.title(f'{energy} {angle_bin} Comptons Radial Bin Count ({angle_bin}° BinsCounts)')
    plt.savefig(f'{sourceFolder}/residualPolarimetry_{angle_bin_str}bin_md{min_dist_str}.png', bbox_inches='tight')
    #plt.savefig(f'{outputFolder_polarimetry_binDist}/{energy}keV_Comptons_Bin_Count_{angle_bin_str}bin_md{min_dist_str}-residual')

    plt.close()

    del data

def plot_radial_bin_count(data, angle_bin, energy):
    # Convert bin centers to radians for polar plot
    phi_radians = np.deg2rad(data['Phi_bin'].astype(float))

    # Plotting
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    bars = ax.bar(phi_radians, data['Counts_Norm'], width=np.deg2rad(angle_bin), color='red', edgecolor='black')

    plt.title(f'{energy}keV Events Radial Bin Count ({angle_bin}° BinsCounts)')

    plt.close()
    plt.close()


def limit_dist_photon_travel(data, min_dist, max_dist):    
    """
    cluster data points for each event and assign globally unique cluster ids.
    
    parameters:
    - data (pd.dataframe): dataframe containing the data to be clustered.
    - global_cluster_id (int): initial global cluster id to start with.

    returns:
    - cluster_dict (dict): a dictionary mapping index to global cluster id.
    - global_cluster_id (int): updated global cluster id after clustering.
    """		

    data['distance_mm'] = np.sqrt(
        (data['E_photon_X0'] - data['E_elect_X0'])**2 +
        (data['E_photon_Y0'] - data['E_elect_Y0'])**2
    )
        
    mindist_filtered_df = data[data['distance_mm'] >= min_dist]

    maxDist_filtered_df = mindist_filtered_df[mindist_filtered_df['distance_mm'] <= max_dist]

    final_filtered_df = maxDist_filtered_df
    

    return final_filtered_df

def compute_polarimetry_phi(data):    
    """
    uses the barycenter of both photoelectorn and compton photon, in a way it recenters the interaction to a central pixel and computes the phi angle.
    """    
    all_phi = []

    #center pixel
    print(data)
    breakpoint()
    for event, line in data.iterrows():
        if line['E_photon=E1'] == 'yes' and line['E_elect=E2'] == 'yes':

            #if (line['E1_x0'] - x_center_pixel) == 0 and (line['E1_y0'] - y_center_pixel) > 0:
            if (line['E1_X0'] - line['E2_X0']) == 0 and (line['E1_Y0'] - line['E2_Y0']) > 0:
                phi = math.pi/2 * (180/math.pi) 

                all_phi.append(phi)
                continue
            
            #elif (line['E1_X0'] - x_center_pixel) == 0 and (line['E1_y0'] - y_center_pixel) < 0:
            elif (line['E1_X0'] - line['E2_X0']) == 0 and (line['E1_Y0'] - line['E2_Y0']) < 0:
                phi = 3 * (math.pi/2) * (180/math.pi)

                all_phi.append(phi)
                continue
            
            else:            

                #phi = math.atan2(line['e1_y0'] - y_center_pixel , line['e1_x0'] - x_center_pixel)
                phi = math.atan2(line['E1_Y0'] - line['E2_Y0'] , line['E1_X0'] - line['E2_X0'])

                phi_deg = math.degrees(phi)

                if phi_deg < 0:
                    phi_deg += 360
            
                all_phi.append(round(phi_deg,2))
        
        elif line['E_photon=E2'] == 'yes' and line['E_elect=E1'] == 'yes':

            #if (line['e2_x0'] - x_center_pixel) == 0 and (line['e2_y0'] - y_center_pixel) > 0:
            if (line['E2_X0'] - line['E1_X0']) == 0 and (line['E2_Y0'] - line['E1_Y0']) > 0:
                phi = math.pi/2 * (180/math.pi)

                all_phi.append(phi)
            
            #elif (line['e2_x0'] - x_center_pixel) == 0 and (line['e2_y0'] - y_center_pixel) < 0:
            elif (line['E2_X0'] - line['E1_X0']) == 0 and (line['E2_Y0'] - line['E1_Y0']) < 0:
                phi = 3 * (math.pi/2) * (180/math.pi)

                all_phi.append(phi)

            else:
                #arg = (line['e2_y0'] - line['e1_y0']) / (line['e2_x0'] - line['e1_x0'])

                #phi = math.atan2(line['e2_y0'] - y_center_pixel, line['e2_x0'] - x_center_pixel) 
                phi = math.atan2(line['E2_Y0'] - line['E1_Y0'], line['E2_X0'] - line['E1_X0']) 

                phi_deg = math.degrees(phi)

                if phi_deg < 0:
                    phi_deg += 360
                    
                all_phi.append(round(phi_deg,2))
        
    data['Phi'] = all_phi
    return data


def plot_all_events_3d(df_xyz, sample=None):
    """3D scatter of all E_photon/E_elect points (optionally subsample)."""
    if sample is not None and sample < len(df_xyz):
        dfp = df_xyz.sample(sample, random_state=0)
    else:
        dfp = df_xyz

    # Half-sizes in cm
    half_cdte = 6.4 / 2.0      # 3.2 cm
    half_si   = 6.656 / 2.0    # 3.328 cm
    z_cdte = -5.0  
    z_si   = 0.0

    # Corners for CdTe plane at z_cdte
    x_cdte = np.array([-half_cdte,  half_cdte,  half_cdte, -half_cdte, -half_cdte])
    y_cdte = np.array([-half_cdte, -half_cdte,  half_cdte,  half_cdte, -half_cdte])
    z_cdte_arr = np.full_like(x_cdte, z_cdte)

    # Corners for Si plane at z_si
    x_si = np.array([-half_si,  half_si,  half_si, -half_si, -half_si])
    y_si = np.array([-half_si, -half_si,  half_si,  half_si, -half_si])
    z_si_arr = np.full_like(x_si, z_si)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # E_photon points (blue)
    ax.scatter(dfp['E_photon_X0'], dfp['E_photon_Y0'], dfp['E_photon_Z0'],
               c='tab:blue', s=20, alpha=0.7, label='Photon (E_photon)', 
               edgecolors='darkblue', linewidth=0.5)

    # E_elect points (orange)
    ax.scatter(dfp['E_elect_X0'], dfp['E_elect_Y0'], dfp['E_elect_Z0'],
               c='tab:orange', s=20, alpha=0.7, label='Electron (E_elect)',
               edgecolors='darkred', linewidth=0.5)

    # Lines between photon and electron for each event
    #for _, r in dfp.iterrows():
    #    ax.plot([r['E_photon_X0'], r['E_elect_X0']],
    #            [r['E_photon_Y0'], r['E_elect_Y0']],
    #            [r['E_photon_Z0'], r['E_elect_Z0']],
    #            color='grey', alpha=0.3, linewidth=1)
    

    # Detector planes
    ax.plot(x_cdte, y_cdte, z_cdte_arr, color='tab:blue', lw=3, label='CdTe plane (z=0)')
    ax.plot(x_si,   y_si,   z_si_arr,   color='tab:green', lw=3, label='Si plane (z=5)')

    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'All events: Photon(blue)/Electron(orange) positions\n(N={len(dfp)} events)')
    ax.legend(loc='upper left')
    
    # Set reasonable view limits
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5.5, 0.5])

    plt.tight_layout()

#filtered_df = compton.limit_dist_photon_travel(compton_data_phi, min_dist, max_dist)
def add_xyz(df, z_si=5.0, z_cdte=0.0):
    """Add X,Y,Z positions for E_photon and E_elect based on overflow flags."""
    df = df.copy()
    
    # Map overflow → z (user‑adjustable)
    z_photon = np.where(df['E_photon_overflow'] == 1, z_cdte, z_si)
    z_elect  = np.where(df['E_elect_overflow'] == 1, z_cdte, z_si)
    
    df['E_photon_Z0'] = z_photon
    df['E_elect_Z0']  = z_elect
    
    return df


def plot_detector_hits(df, detector='both', sample=None, bins=50, log_color=True):
    """
    Plot 2D spatial energy distribution on CdTe (z=0) and/or Si (z=5) detectors.
    
    Parameters:
    - detector: 'cdte', 'si', or 'both'
    - sample: max events to plot (faster for large datasets)
    - bins: number of histogram bins
    """
    if sample is not None and sample < len(df):
        dfp = df.sample(sample, random_state=0)
    else:
        dfp = df
    
    fig, axes = plt.subplots(1, 1 if detector != 'both' else 2, 
                           figsize=(8 if detector != 'both' else 16, 6))
    if detector != 'both':
        axes = [axes]
    
    # CdTe hits (z=0)
    if detector in ['cdte', 'both']:
        cdte_mask = dfp['E_photon_Z0'] == -5.0
        if cdte_mask.sum() > 0:
            cdte_hits = dfp[cdte_mask]
            axes[0].hist2d(cdte_hits['E_photon_X0'], cdte_hits['E_photon_Y0'],
                          weights=cdte_hits['E_photon'], bins=bins,
                          cmap='hot', range=[[-3.5, 3.5], [-3.5, 3.5]], norm=LogNorm() if log_color else None)
            axes[0].set_title(f'CdTe (z=0) Energy Distribution\n(N={cdte_mask.sum():,} hits)')
            axes[0].set_xlabel('X (cm)')
            axes[0].set_ylabel('Y (cm)')
            plt.colorbar(axes[0].collections[0], ax=axes[0], label='Energy (keV)')
    
    # Si hits (z=5)
    if detector in ['si', 'both']:
        si_mask = dfp['E_photon_Z0'] == 0.0
        if si_mask.sum() > 0:
            si_hits = dfp[si_mask]
            ax_idx = 0 if detector == 'si' else 1
            axes[ax_idx].hist2d(si_hits['E_photon_X0'], si_hits['E_photon_Y0'],
                               weights=si_hits['E_photon'], bins=bins,
                               cmap='viridis', range=[[-3.5, 3.5], [-3.5, 3.5]], norm=LogNorm() if log_color else None)
            axes[ax_idx].set_title(f'Si (z=5) Energy Distribution\n(N={si_mask.sum():,} hits)')
            axes[ax_idx].set_xlabel('X (cm)')
            axes[ax_idx].set_ylabel('Y (cm)')
            plt.colorbar(axes[ax_idx].collections[0], ax=axes[ax_idx], label='Energy (keV)')
    
    plt.tight_layout()
def fit_radial_plot(outputFolder, energy, angle_bin, min_dist, max_dist, residual=False):
    
    outputFolder_parquet = os.path.join(outputFolder, 'parquet')
    outputFolder_parquet_doubles = os.path.join(outputFolder_parquet, 'doubles')
    outputFolder_parquet_doublesPeak = os.path.join(outputFolder_parquet_doubles, 'inPeak')
    outputFolder_parquet_doublesPeakCompton = os.path.join(outputFolder_parquet_doublesPeak, 'comptons')


    outputFolder_polarimetry = os.path.join(outputFolder, 'photonPolarimetry')
    pathlib.creat_dir(outputFolder_polarimetry)
    angle_bin_str = str(angle_bin).replace('.','-')
    min_dist_str = str(min_dist).replace('.','-')
    max_dist_str = str(max_dist).replace('.','-')

    outputFolder_polarimetry_binDist = os.path.join(outputFolder_polarimetry, f'{angle_bin_str}bin_md{min_dist_str}_maxd{max_dist_str}')
    pathlib.creat_dir(outputFolder_polarimetry_binDist)
   
    final_filtered_df = []
    for filename in os.listdir(f'{outputFolder_parquet_doublesPeakCompton}/'):
        if filename.startswith('comptons') and filename.endswith(".parquet"):
            filepath = os.path.join(outputFolder_parquet_doublesPeakCompton, filename)
            
            df = pd.read_parquet(filepath)

            print(df)
            
            compton_data_phi = compton.compute_polarimetry_phi(df)
            ## limit xmax and xmin

            compton_data_phi = limit_dist_photon_travel(compton_data_phi, 0.05, 2)
            print(compton_data_phi)


            filtered_df = add_xyz(compton_data_phi, z_si=0, z_cdte=-5)
            #plot_all_events_3d(filtered_df) 

            #plot_all_events_3d(filtered_df)
            final_filtered_df.append(filtered_df)

     
    concatenated_df = pd.concat(final_filtered_df, ignore_index=True)

    plot_detector_hits(concatenated_df, detector='both', bins = 100)
    plot_all_events_3d(concatenated_df) 
    plt.show()


    data_concatenated_df = compton.function_to_idk(concatenated_df, energy, angle_bin)
    
    #data = data_concatenated_df.groupby('Phi_bin', as_index=True).sum().reset_index()
    data = data_concatenated_df.groupby('Phi_bin', as_index=False, observed=False).sum()
    
    if residual:
        phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error = fit_binned_counts_residual(data, angle_bin)
        gc.collect()

        plot_data = pd.DataFrame({
        'Phi_bin': phi,  # Bin centers
        'Counts_Norm': counts_norm,  # Normalized counts
        'Error': error,  # Errors
    })
        
        plot_data_fit = pd.DataFrame({
        'Phi_fit': phi_fit,  # Fitted phi values
        'Counts_fit': counts_fit  # Fitted counts
    })
        plot_data.to_csv(
            f'{outputFolder_polarimetry_binDist}/{energy}keV_Comptons_Bin_Count_{angle_bin_str}bin_md{min_dist_str}_plot_data-residual.csv',
            index=False  # Do not save the index
        )
        plot_data_fit.to_csv(
            f'{outputFolder_polarimetry_binDist}/{energy}keV_Comptons_Bin_Count_{angle_bin_str}bin_md{min_dist_str}_plot_data-FIT-residual.csv',
            index=False  # Do not save the index
        )

        plot_fit_binned_counts_residual(outputFolder_polarimetry_binDist, data, phi_fit, counts_fit, popt, perr, counts_norm, phi, energy, angle_bin, min_dist, chi2)
        
        A = popt[0]
        A_err = perr[0]

        B = popt[1]
        B_err = perr[1]

        C = popt[2]
        C_err = perr[2]
        
        phi_c = popt[3]
        phi_c_err = perr[3]

        phi_s = popt[4]
        phi_s_err = perr[4]

        #Q = B / (2*A + B)
        #Q = A / C
        #Q_err = (A/C)*np.sqrt((A_err/A)**2 + (C_err/C)**2)

        #PA = B-90
        #PA_err = B_err

        with open(f'{outputFolder_polarimetry_binDist}/Fit_Values-residual.txt', 'w') as f:
            f.write(f'A = {A} ± {A_err}\nB = {B} ± {B_err}\nC = {C} ± {C_err}\nphi_c = {np.degrees(phi_c)} ± {np.degrees(phi_c_err)}\nphi_s = {np.degrees(phi_s)} ± {np.degrees(phi_s_err)}')

        # Convert phi_fit to radians for polar plot
        phi_radians_fit = np.deg2rad(phi_fit)

        # Plotting
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111, polar=True)
        bars = ax.bar(np.deg2rad(phi), counts_norm, width=np.deg2rad(angle_bin), color='red', alpha = 0.9, edgecolor='black')
        ax.grid(True, alpha=0.3)

        # Plot the fitted curve
        ax.plot(phi_radians_fit, counts_fit, color='black', label=(
            fr'N($\phi$) = $A + B\\sin^2(\phi - \phi_c) + C\\sin^2(\phi - \phi_s)$\n'
            f'A = {popt[0]:.2f} ± {perr[0]:.2f}\n'
            f'B = {popt[1]:.2f} ± {perr[1]:.2f}\n'
            f'C = {popt[2]:.2f} ± {perr[2]:.2f}\n'
            f'$\\phi_r$ = {np.degrees(phi_c):.2f} ± {np.degrees(phi_c_err):.2f}\n'
            f'$\\phi_s$ = {np.degrees(phi_s):.2f} ± {np.degrees(phi_s_err):.2f}\n'
            f'$\\chi^2_{{red}}$ = {chi2:.2f}'
        ))  

                
        # Customize the plot to have 0 degrees to the right
        ax.set_theta_zero_location('E')  # Set 0 degrees to the right
        ax.set_theta_direction(1)  # Set the direction of increasing avngles (clockwise)

        # Set radial ticks (intensity scale) along the 90º line
        max_count_norm = data['Counts_Norm'].max()
        radial_ticks = np.linspace(0, max_count_norm, 5)
        ax.set_rticks(radial_ticks)  # Set the number of ticks
        ax.set_yticklabels([f'{tick:.2f}' for tick in radial_ticks], verticalalignment='top')  # Place tick labels

        ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.tight_layout()
        #plt.title(f'{energy}keV Events Radial Bin Count ({angle_bin_str}° BinsCounts)')
        plt.savefig(f'{outputFolder_polarimetry_binDist}/{energy}keV_Events_Radial_Bin_Count_{angle_bin_str}bin_md{min_dist_str}-residual')
        plt.close()
        
        gc.collect()
        del data, phi_fit, counts_fit, popt, perr, counts_norm, phi
    else:
        phi_fit, counts_fit, popt, perr, counts_norm, phi, chi2, error = fit_binned_counts(data, angle_bin)

   # counts_fit = fit_binned_counts(data)[1]
   # popt = fit_binned_counts(data)[2]
   # perr = fit_binned_counts(data)[3]
   # counts_norm = fit_binned_counts(data)[4]
   # phi = fit_binned_counts(data)[5]
        gc.collect()

        plot_data = pd.DataFrame({
        'Phi_bin': phi,  # Bin centers
        'Counts_Norm': counts_norm,  # Normalized counts
        'Error': error,  # Errors
    })
        
        plot_data_fit = pd.DataFrame({
        'Phi_fit': phi_fit,  # Fitted phi values
        'Counts_fit': counts_fit  # Fitted counts
    })
        plot_data.to_csv(
            f'{outputFolder_polarimetry_binDist}/{energy}keV_Comptons_Bin_Count_{angle_bin_str}bin_md{min_dist_str}_plot_data.csv',
            index=False  # Do not save the index
        )
        plot_data_fit.to_csv(
            f'{outputFolder_polarimetry_binDist}/{energy}keV_Comptons_Bin_Count_{angle_bin_str}bin_md{min_dist_str}_plot_data-FIT.csv',
            index=False  # Do not save the index
        )

        plot_fit_binned_counts(outputFolder_polarimetry_binDist, data, phi_fit, counts_fit, popt, perr, counts_norm, phi, energy, angle_bin, min_dist, chi2)
        
        A = popt[0]
        A_err = perr[0]

        B = math.degrees(popt[1])
        B_err = math.degrees(perr[1])

        C = popt[2]
        C_err = perr[2]

        #Q = B / (2*A + B)
        Q = A / C
        Q_err = (A/C)*np.sqrt((A_err/A)**2 + (C_err/C)**2)

        PA = B-90
        PA_err = B_err

        with open(f'{outputFolder_polarimetry_binDist}/Fit_Values.txt', 'w') as f:
            f.write(f'A = {A} ± {A_err}\nB = {B} ± {B_err}\nC = {C} ± {C_err}\nQ = {Q} ± {Q_err}\nPA = {PA} ± {PA_err}')

        # Convert phi_fit to radians for polar plot
        phi_radians_fit = np.deg2rad(phi_fit)

        # Plotting
        plt.figure(figsize=(10, 6))
        ax = plt.subplot(111, polar=True)
        bars = ax.bar(np.deg2rad(phi), counts_norm, width=np.deg2rad(angle_bin), color='red', alpha = 0.9, edgecolor='black')
        ax.grid(True, alpha=0.3)

        # Plot the fitted curve
        ax.plot(phi_radians_fit, counts_fit, color='black', label=fr'N($\phi$) = $A$ $\cdot$ $\cos$2($\phi$ - $B$) + $C$\\$A$ = {popt[0]:.2f} $\pm$ {perr[0]:.2f}\\$B$ = {math.degrees(popt[1]):.2f} $\pm$ {math.degrees(perr[1]):.2f}$^\circ$ \\$C$ = {popt[2]:.2f} $\pm$ {perr[2]:.2f}' + '\n' + r"$\chi^2_{red}$ = " + f"{chi2:.2f}")

                
        # Customize the plot to have 0 degrees to the right
        ax.set_theta_zero_location('E')  # Set 0 degrees to the right
        ax.set_theta_direction(1)  # Set the direction of increasing avngles (clockwise)

        # Set radial ticks (intensity scale) along the 90º line
        max_count_norm = data['Counts_Norm'].max()
        radial_ticks = np.linspace(0, max_count_norm, 5)
        ax.set_rticks(radial_ticks)  # Set the number of ticks
        ax.set_yticklabels([f'{tick:.2f}' for tick in radial_ticks], verticalalignment='top')  # Place tick labels

        ax.legend(bbox_to_anchor=(1.04, 1), borderaxespad=0)
        plt.tight_layout()
        #plt.title(f'{energy}keV Events Radial Bin Count ({angle_bin_str}° BinsCounts)')
        plt.savefig(f'{outputFolder_polarimetry_binDist}/{energy}keV_Events_Radial_Bin_Count_{angle_bin_str}bin_md{min_dist_str}.png')
        plt.close()
        
        gc.collect()
        del data, phi_fit, counts_fit, popt, perr, counts_norm, phi

    return concatenated_df

    


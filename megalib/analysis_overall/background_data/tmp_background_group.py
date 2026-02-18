#!/usr/bin/env python3
"""
Script to concatenate all CdTe and Si spectra from Background folders and plot them
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import re

def find_spectra_files(base_path="."):
    """
    Find all CdTe and Si spectra files in the Background folders
    """
    # Get all Background folders
    bg_folders = glob.glob("Background500sec_2-400keV_config4x4_1.5cm_inc*")
    bg_folders.sort()  # Sort to process in order
    
    cdte_files = []
    si_files = []
    
    for folder in bg_folders:
        # Construct the expected paths
        folder_name = os.path.basename(folder)
        
        # CdTe file path
        cdte_path = os.path.join(folder, "parquet", "masked", "spectra", 
                                 f"{folder_name}_spec_hist-masked_Det-CdTe.txt")
        if os.path.exists(cdte_path):
            cdte_files.append(cdte_path)
        else:
            print(f"Warning: CdTe file not found: {cdte_path}")
        
        # Si file path
        si_path = os.path.join(folder, "parquet", "masked", "spectra",
                               f"{folder_name}_spec_hist-masked_Det-Si.txt")
        if os.path.exists(si_path):
            si_files.append(si_path)
        else:
            print(f"Warning: Si file not found: {si_path}")
    
    return cdte_files, si_files

def read_spectrum_file(filepath):
    """
    Read a spectrum file with header and return energy and counts
    File format:
    - First line: # ObsTime(ns)	[value]
    - Second line: # Bin size 	[value] keV
    - Third line: # Bin_center(keV),cnts(total)
    - Then data: energy,counts
    """
    try:
        # Read metadata from header
        with open(filepath, 'r') as f:
            header_lines = [f.readline().strip() for _ in range(3)]
        
        # Extract bin size from second header line
        bin_size = None
        for line in header_lines:
            if 'Bin size' in line:
                # Extract number using regex
                match = re.search(r'Bin size\s+([\d.]+)', line)
                if match:
                    bin_size = float(match.group(1))
                    print(f"  Bin size: {bin_size} keV")
        
        # Read data (skip first 3 header lines)
        data = np.loadtxt(filepath, skiprows=3, delimiter=',')
        
        if data.ndim == 1:  # If only one column or empty
            print(f"Warning: {filepath} has no valid data")
            return None, None, bin_size
        
        # Data has two columns: energy, counts
        energies = data[:, 0]
        counts = data[:, 1]
        
        return energies, counts, bin_size
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None, None

def combine_spectra_same_bins(file_list, detector_name):
    ref_energies = None
    total_counts = None
    bin_sizes = []

    for filepath in file_list:
        energies, counts, bin_size = read_spectrum_file(filepath)

        if energies is None or counts is None or len(energies) == 0:
            continue

        if ref_energies is None:
            ref_energies = energies
            total_counts = counts.astype(float)
        else:
            # check same grid
            if not np.allclose(energies, ref_energies):
                raise ValueError(f"{filepath}: energy grid mismatch")
            total_counts += counts

        if bin_size is not None:
            bin_sizes.append(bin_size)

    if ref_energies is None:
        print(f"No valid data found for {detector_name}")
        return None, None

    if bin_sizes:
        unique_bin_sizes = set(bin_sizes)
        if len(unique_bin_sizes) > 1:
            print(f"Warning: inconsistent bin sizes: {unique_bin_sizes}")
        else:
            print(f"Consistent bin size: {bin_sizes[0]} keV")

    print(f"Total points: {len(ref_energies)}")
    print(f"Energy range: {ref_energies.min():.2f} - {ref_energies.max():.2f} keV")

    return ref_energies, total_counts

def concatenate_spectra(file_list, detector_name):
    """
    Concatenate spectra from multiple files
    Returns combined energy and counts arrays
    """
    all_energies = []
    all_counts = []
    bin_sizes = []
    
    print(f"\nProcessing {detector_name} files:")
    for filepath in file_list:
        print(f"  Reading: {os.path.basename(filepath)}")
        energies, counts, bin_size = read_spectrum_file(filepath)
        
        if energies is not None and counts is not None and len(energies) > 0:
            all_energies.extend(energies)
            all_counts.extend(counts)
            if bin_size is not None:
                bin_sizes.append(bin_size)
    
    # Convert to numpy arrays
    if all_energies:
        all_energies = np.array(all_energies)
        all_counts = np.array(all_counts)
        
        # Sort by energy
        sort_idx = np.argsort(all_energies)
        all_energies = all_energies[sort_idx]
        all_counts = all_counts[sort_idx]
        
        # Check if bin sizes are consistent
        if bin_sizes:
            unique_bin_sizes = set(bin_sizes)
            if len(unique_bin_sizes) > 1:
                print(f"  Warning: Inconsistent bin sizes found: {unique_bin_sizes}")
            else:
                print(f"  Consistent bin size: {bin_sizes[0]} keV")
        
        print(f"  Total points: {len(all_energies)}")
        print(f"  Energy range: {all_energies.min():.2f} - {all_energies.max():.2f} keV")
        
        return all_energies, all_counts
    else:
        print(f"  No valid data found for {detector_name}")
        return None, None

def plot_spectra(cdte_energies, cdte_counts, si_energies, si_counts, output_file="combined_spectra.png"):
    """
    Plot the concatenated spectra
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # CdTe plot
    if cdte_energies is not None and len(cdte_energies) > 0:
        ax1.plot(cdte_energies, cdte_counts, 'r-', linewidth=0.5, label='CdTe', alpha=0.7)
        ax1.set_ylabel('Counts')
        ax1.set_title('CdTe Detector - Combined Spectra from All Background Runs')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Add some statistics
        total_counts = np.sum(cdte_counts)
        ax1.text(0.02, 0.95, f'Total counts: {total_counts:.2e}', 
                transform=ax1.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    else:
        ax1.text(0.5, 0.5, 'No CdTe data found', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=14, color='red')
    
    # Si plot
    if si_energies is not None and len(si_energies) > 0:
        ax2.plot(si_energies, si_counts, 'b-', linewidth=0.5, label='Si', alpha=0.7)
        ax2.set_xlabel('Energy (keV)')
        ax2.set_ylabel('Counts')
        ax2.set_title('Si Detector - Combined Spectra from All Background Runs')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend()
        ax2.set_yscale('log')
        
        # Add some statistics
        total_counts = np.sum(si_counts)
        ax2.text(0.02, 0.95, f'Total counts: {total_counts:.2e}', 
                transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
    else:
        ax2.text(0.5, 0.5, 'No Si data found', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=14, color='red')
    
    plt.suptitle('Background Spectra - All Incimation Angles Combined', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved as: {output_file}")

def save_combined_spectra(cdte_energies, cdte_counts, si_energies, si_counts):
    """
    Save the combined spectra to text files with headers
    """
    if cdte_energies is not None and len(cdte_energies) > 0:
        cdte_combined = np.column_stack((cdte_energies, cdte_counts))
        header = "Combined CdTe spectra from all Background runs\n PSF radius Mirror/Leses(cm), 0.075\nObservation Time (s), 10000\nBin (keV), 0.1 "
        header += "Bin_center(keV),cnts(total)"
        np.savetxt("combined_CdTe_spectra.txt", cdte_combined, 
                   header=header, delimiter=',', fmt='%.6e,%.6e')
        print("Saved combined CdTe spectra to: combined_CdTe_spectra.txt")
    
    if si_energies is not None and len(si_energies) > 0:
        si_combined = np.column_stack((si_energies, si_counts))
        header = "Combined Si spectra from all Background runs\n PSF radius Mirror/Lenses (cm), 0.075\nObservation Time (s), 10000\nBin (keV), 0.1"
        header += "Bin_center(keV),cnts(total)"
        np.savetxt("combined_Si_spectra.txt", si_combined,
                   header=header, delimiter=',', fmt='%.6e,%.6e')
        print("Saved combined Si spectra to: combined_Si_spectra.txt")

def plot_individual_runs(file_lists, detector_name):
    """
    Optional: Plot individual runs to see variations
    """
    if not file_lists:
        return
    
    plt.figure(figsize=(12, 6))
    
    for filepath in file_lists:  
        energies, counts, _ = read_spectrum_file(filepath)
        if energies is not None and counts is not None:
            inc_angle = re.search(r'inc(\d+)', filepath)
            label = inc_angle.group(1) if inc_angle else 'unknown'
            plt.plot(energies, counts, linewidth=0.5, alpha=0.7, label=f'inc{label}')
    
    plt.xlabel('Energy (keV)')
    plt.ylabel('Counts')
    plt.title(f'{detector_name} - Individual Background Runs (first 10)')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'individual_{detector_name}_runs.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("="*60)
    print("BACKGROUND SPECTRA COMBINER")
    print("="*60)
    
    print("\nFinding spectra files...")
    cdte_files, si_files = find_spectra_files()
    
    print(f"\nFound {len(cdte_files)} CdTe files")
    print(f"Found {len(si_files)} Si files")
    
    if not cdte_files and not si_files:
        print("\nNo files found! Check the directory structure.")
        return
    
    print("\n" + "="*60)
    print("PROCESSING SPECTRA")
    print("="*60)
    
    # Concatenate spectra
    #cdte_energies, cdte_counts = concatenate_spectra(cdte_files, "CdTe")
    #si_energies, si_counts = concatenate_spectra(si_files, "Si")
     
    cdte_energies, cdte_counts = combine_spectra_same_bins(cdte_files, "CdTe")
    si_energies, si_counts = combine_spectra_same_bins(si_files, "Si")

    print("\n" + "="*60)
    print("SAVING COMBINED SPECTRA")
    print("="*60)
    save_combined_spectra(cdte_energies, cdte_counts, si_energies, si_counts)
    
    print("\n" + "="*60)
    print("PLOTTING RESULTS")
    print("="*60)
    plot_spectra(cdte_energies, cdte_counts, si_energies, si_counts)
    
    # Optional: Plot individual runs
    plot_individual = input("\nPlot individual runs? (y/n): ").lower().strip()
    if plot_individual == 'y':
        if cdte_files:
            plot_individual_runs(cdte_files, "CdTe")
        if si_files:
            plot_individual_runs(si_files, "Si")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    if cdte_energies is not None:
        print(f"CdTe - Total files: {len(cdte_files)}")
        print(f"CdTe - Total points: {len(cdte_energies)}")
        print(f"CdTe - Energy range: {cdte_energies.min():.2f} - {cdte_energies.max():.2f} keV")
        print(f"CdTe - Total counts: {np.sum(cdte_counts):.2e}")
    if si_energies is not None:
        print(f"Si - Total files: {len(si_files)}")
        print(f"Si - Total points: {len(si_energies)}")
        print(f"Si - Energy range: {si_energies.min():.2f} - {si_energies.max():.2f} keV")
        print(f"Si - Total counts: {np.sum(si_counts):.2e}")

if __name__ == "__main__":
    main()

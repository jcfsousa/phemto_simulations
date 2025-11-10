import gzip
from scipy.signal import find_peaks
import sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit




def crystal_ball(x, amp, mean, sigma, alpha, n):

    x = np.asarray(x)
    z = (x - mean) / sigma
    abs_alpha = np.abs(alpha)
    A = (n / abs_alpha)**n * np.exp(-0.5 * abs_alpha**2)
    B = n / abs_alpha - abs_alpha

    result = np.empty_like(z)
    mask = z > -abs_alpha
    result[mask] = amp * np.exp(-0.5 * z[mask]**2)
    result[~mask] = amp * A * (B - z[~mask])**(-n)

    return result

def gaussian(x, mu, sig, A): 
   return H_gauss(A, sig)*np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2)))

def H_gauss(A, sig):
   return A/(np.sqrt(2*np.pi)*sig)

def sum_of_crystal_ball_grouped(x, *params):
    """
    Sum of N Crystal Ball functions.
    Parameters are grouped by type:
    [amp1, amp2, ..., mean1, mean2, ..., sigma1, ..., alpha1, ..., n1, ...]
    Total length must be divisible by 5.
    """
    n_params = len(params)
    if n_params % 5 != 0:
        raise ValueError("Number of parameters must be a multiple of 5.")

    N = n_params // 5
    amps   = params[0:N]
    means  = params[N:2*N]
    sigmas = params[2*N:3*N]
    alphas = params[3*N:4*N]
    ns     = params[4*N:5*N]

    total = np.zeros_like(x)
    for i in range(N):
        total += crystal_ball(x, amps[i], means[i], sigmas[i], alphas[i], ns[i])
    return total
 

def search_peaks(bins, cnts, prominence = 1, width = 5):                                              
    '''This function accepts np.arrays and checks for peaks on the data. It was initially designed to search peaks on radiactive spectra.
   
   Parameters:
       bins -> x axis 
       cnts -> y axis
       peak_width -> expected peak width
   
   Output example:
        peak_bins: [ 21.5 187.5 433.5 684.5] 
        peak_indices: [ 20 186 432 683]
    '''
    cnts_filtered = np.array(cnts)
     
    peak_indices, properties = find_peaks(
        cnts_filtered, width=width
        )
  
    peak_bins = bins[peak_indices]
     
    return peak_bins, peak_indices  

def check_closest_peak(theoretical_peak, list_found_peaks, bins_found_peaks):
    '''
    Use centered bins to the the x value of the bin center.
    '''
    list_diff_energy = []
    ref_energy_list = []
    for energy_check in list_found_peaks:
        energy_diff = abs(theoretical_peak - energy_check)
        list_diff_energy.append(energy_diff)
    
    index_min_diff = np.argmin(list_diff_energy)
    peak_energy_to_use = list_found_peaks[index_min_diff]
    bin_peak_energy_to_use = bins_found_peaks[index_min_diff]

    return peak_energy_to_use, bin_peak_energy_to_use

if __name__ == '__main__':

    geofile = '/local/home/jf285468/Documents/PHEMTO/new/instruments/PHEMTO_config1.geo.setup'
    log_E=[4,8,30,50,80,100,120,150,200,250,300,400,500,600]
    

    geofile_name = geofile.split('/')[-1]
    geofile_onlyname = geofile_name.split('.')[0]

    source_basename = 'HomogeneousBeam'

    spectra_dict = {}

    for energy in log_E:
        trafile = f'./sources/simTra_files/{source_basename}{energy}keV_{geofile_onlyname}.inc1.id1.tra.gz'
        print(f'reading {trafile} ...')
        with gzip.open(trafile,'rt') as f:
            spectra_file_Si = []
            spectra_file_CdTe = []
            spectra_file_SiCdTe = []
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
                   if event_z == 0: #Si detector interaction
                       spectra_file_Si.append(event_energy)
                   elif event_z == -5: # CdTe interaction
                       spectra_file_CdTe.append(event_energy)
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
                   if ch0_event_z == 0 and ch1_event_z == 0: #Si detector interaction
                       spectra_file_Si.append(event_energy)
                   elif ch0_event_z == -5 and ch1_event_z == -5: # CdTe interaction
                       spectra_file_CdTe.append(event_energy)
                   else:
                       spectra_file_SiCdTe.append(event_energy)
                else:
                    continue
        spectra_dict[f'Si,{energy}'] = spectra_file_Si
        spectra_dict[f'CdTe,{energy}'] = spectra_file_CdTe
        spectra_dict[f'SiCdTe,{energy}'] = spectra_file_SiCdTe

    
    eff_dict = {}
    eff_si = {}
    eff_cdte = {}
    eff_sicdte = {}

    for energy in log_E:
        location_events = ['Si', 'CdTe', 'SiCdTe']
        
        total_events = 0
        for location in location_events:
            energy_list = spectra_dict[f'{location},{energy}']

            xmin = 0
            xmax = energy*2
            if location == 'Si':
                steps = round((xmax-xmin)*20,0) #0.05 keV bin
                bin = (xmax-xmin)/(steps)
            elif location == 'CdTe':
                steps = round((xmax-xmin)*2,0) # 0.5 keV bin
                bin = (xmax-xmin)/(steps)
            elif location == 'SiCdTe':
                steps = round((xmax -xmin),0) # 1 keV bin
                bin = (xmax-xmin)/steps
            bins = np.linspace(xmin, xmax, steps)
            hist, bins = np.histogram(energy_list, bins=bins)
            bins_centered = (bins[:-1] + bins[1:])/2
            hist_err = np.sqrt(hist)
            
            #plt.plot(bins_centered, hist)
            #plt.title('Spectra initial')
            #plt.show()

            initial_guess = [energy, 0.01*energy, np.sum(hist)]
            initial_guess = np.hstack(initial_guess)
            hist_err = np.sqrt(hist) 

            popt, pcov = curve_fit(
                gaussian,
                bins_centered,
                hist,
                #sigma = hist_err, 
                p0=initial_guess
            )
            mu = popt[0]
            sigma = popt[1]
            A = popt[2]*(1/bin)

            x_dummy = np.linspace(xmin, xmax, steps*10)
            x_dummy_centered = (x_dummy[:-1] + x_dummy[1:])/2
            y_fit = gaussian(x_dummy_centered,popt[0], popt[1], popt[2])
             
            dpi = 96*2
            width = 2000*1.75
            height = 1300*1.3
            fig = 1

            #plt.figure(fig, figsize=(width/dpi, height/dpi), dpi = dpi)
            #plt.errorbar(bins_centered, hist, fmt = '+', yerr=hist_err, color='blue', ecolor = 'red', label='Data')
            #plt.plot(x_dummy_centered, y_fit, label=f'Fitted Gaussian \n mu = {mu} \n sigma = {sigma} \n A = {A}', color='orange', lw=2)
            #plt.legend()
            #plt.xlabel("Energy")
            #plt.ylabel("Counts")
            #plt.grid(True)
            #plt.title(f"Fit detector {location}, {energy} keV")
            #plt.show()
            
            
            if location == 'Si':
                eff_si[f'{energy}'] = A/1e6
            elif location == 'CdTe':
                eff_cdte[f'{energy}'] = A/1e6
            elif location == 'SiCdTe':
                eff_sicdte[f'{energy}'] = A/1e6

            total_events = total_events + A

        eff_dict[f'{energy}'] = total_events/1e6

    print(eff_dict) 
    eff_list = []
    eff_si_list = []
    eff_cdte_list = []
    eff_sicdte_list = []

    for energy in log_E:
        eff_list.append(eff_dict[f'{energy}'])
        eff_si_list.append(eff_si[f'{energy}'])
        eff_cdte_list.append(eff_cdte[f'{energy}'])
        eff_sicdte_list.append(eff_sicdte[f'{energy}'])

    plt.plot(log_E, eff_list, label = 'Eff total')
    plt.plot(log_E, eff_si_list, label = 'Eff Si')
    plt.plot(log_E, eff_cdte_list, label = 'Eff CdTe')
    plt.plot(log_E, eff_sicdte_list, label = 'Eff Si+CdTe evens')
    plt.legend()

    plt.ylabel('Detection efficiency')
    plt.xlabel('Energy (keV)')

    plt.show()

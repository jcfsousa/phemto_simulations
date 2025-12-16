import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
import sys
# handle warning as exceptions
warnings.filterwarnings("error")

from file_handling import fix_spectra
from scipy.special import erfc
from scipy.optimize import curve_fit


CONSTANTS = ['a', 'b', 'c', 't']
CUT_EDGE = 5 # in pixels
BAD_PIXELS = [(216, 198)]

# -------------------- file processing --------------------

def calc_tot(energy, a, b, c, t):
    """Calculates the tot from the energy using the calibration constants."""
    return a*energy + b - c/(energy-t)

def calc_energy(tot, a, b, c, t):
    """Calculates the energy from the tot using the calibration constants."""
    p = (b-tot)/a - t
    q = (1/a) * (t*(tot-b) - c)
    det = (p**2)/4 - q
    return -p/2 + np.sqrt(det)

def get_cals(calc_dict, x, y):
    """Returns the calibration constants for a given pixel."""
    return ([calc_dict[constant][x, y] for constant in CONSTANTS])
        
def get_energy(tot, calc_dict, x, y):
    """Calculates the energy from the tot using the calibration constants for a given pixel."""
    return calc_energy(tot, *get_cals(calc_dict, x, y))

def convert_tot(df, calc_dict, func):
    """Converts the tot to energy using the calibration constants for each pixel."""
    df['energy'] = func(df['tot'], calc_dict, df['x'], df['y'])
    return df

def get_const():
    """Loads the calibration constants from the files into a dictionary."""
    cal_const = {}
    for constant in CONSTANTS:
        cal_const[constant] = np.loadtxt(f'calibration-constants/constant-{constant}.txt', unpack=True)
    return cal_const

def load_data(filepath):
    """Loads the data from the file into a numpy array."""
    return np.loadtxt(filepath, skiprows=1, unpack=True, usecols=(1, 2, 3, 4), dtype=np.int64)

def fix_file(filepath):
    filepath_fixed = filepath.replace('.t3pa', '_fixed.txt')
    if os.path.isfile(filepath_fixed):
        print('File contains non-ascii characters. Using fixed file...')
    else:
        print('File contains non-ascii characters. Fixing file...')
        fix_spectra(filepath)
    return filepath_fixed

def get_data(filepath):
    """Loads the data from the file into a pandas dataframe."""
    matrix, toa, tot, ftoa = load_data(filepath)
    data = {
        'matrix': matrix,
        'toa': toa*25 - ftoa*1.5625,
        'tot': tot,
        'x': matrix % 256,
        'y': matrix // 256
        }
    return pd.DataFrame(data)

def get_df(filepath, const):
    """Loads the data from the file into a pandas dataframe and converts the tot to energy."""
    print(f'Reading {os.path.basename(filepath)}...')
    df_data = get_data(filepath)
    print('All data loaded. Calculating energy...')
    df = convert_tot(df_data, const, get_energy)
    if len(df_data) > len(df['energy']):
        print(f"{len(df_data) - len(df['energy'])} measurements were not converted to energy.")
    else:
        print(f"Success!")
    print(f'Total number of measurements: {round(len(df)/1e6, 2)}M')
    return df

def get_diff(df):
    """Calculates the difference between the toa of each measurement, 0 for first element."""
    return np.append(np.zeros(1), np.diff(df['toa'], 1)).astype(np.int64)

def index_events(df, event_length):
    """Indexes each entry with a toa within a given event length. Returns the dataframe with the new column 'event_index'."""
    diff = get_diff(df)
    event = 1
    for i in range(len(diff)):
        if diff[i] < event_length:
            diff[i] = event
        else:
            event += 1
            diff[i] = event
    df['event_index'] = diff
    return df

def get_m_counts(df):
    """Returns the coordinates and the counts of measurements of each pixel."""
    matrix_counts = df.matrix.value_counts()
    y, x = divmod(matrix_counts.index, 256)
    m_counts = matrix_counts.values
    return x, y, m_counts

def plot_counts(df):
    """Plots the counts of measurements of the whole detector area."""
    x, y, m_counts = get_m_counts(df)
    x_max = x[m_counts == np.max(m_counts)][0]
    y_max = y[m_counts == np.max(m_counts)][0]
    print('x_max:', x_max)
    print('y_max:', y_max)
    print('max_counts:', np.max(m_counts))
    fig, ax  = plt.subplots()
    ax.set_aspect('equal')
    plt.scatter(x, y, c=m_counts, s=1, cmap='jet', vmin=1, vmax=np.nanpercentile(m_counts, 99))
    plt.xlabel('x pixel number')
    plt.ylabel('y pixel number')
    plt.plot(x_max, y_max, marker = "+", color = 'black', markersize=10, label = 'max', linestyle = 'None')
    plt.legend()
    cbar = plt.colorbar()
    cbar.set_label('counts')

def mean_transmission(energy, fraction):
    """Returns the mean transmission for each element."""
    e_trans, tranmission = np.loadtxt('transmission.txt', unpack = True, skiprows=1, delimiter=',')
    efine = np.arange(10, 1300, 1)
    transmission = np.interp(efine, e_trans, tranmission)
    energy = np.hstack(energy)
    trans = np.array([transmission[efine == int(np.round(en))] for en in energy])
    fraction = np.array(fraction, dtype = np.float64)
    return np.nansum(np.multiply(trans, fraction))/np.nansum(fraction)

def bad_pixel_mask(x, y, bad_pixel):
    """Returns a mask for bad pixels."""
    return np.logical_or((x != bad_pixel[0]), (y != bad_pixel[1]))

def bad_pixels_mask(x, y):
    """Returns a mask for bad pixels."""
    mask = np.ones(len(x), dtype = bool)
    for pixel in BAD_PIXELS:
        mask = np.logical_and(mask, bad_pixel_mask(x, y, pixel))
    return mask

def edge_mask(x, y):
    """Returns a mask for the edge pixels."""
    return (x > CUT_EDGE) & (x < (255 - CUT_EDGE)) & (y > CUT_EDGE) & (y < (255 - CUT_EDGE))

def mask_bad_pixels(df):
    mask = np.logical_and(bad_pixels_mask(df['x'], df['y']), edge_mask(df['x'], df['y']))
    return df[mask]

def mask_outliers(df):
    x, y, m_counts = get_m_counts(df)
    mask = m_counts <= np.mean(m_counts) + 3*np.std(m_counts)
    matrix = 256*y + x
    return df[np.isin(df['matrix'], matrix[mask])]

def snr(df_raw, df_masked):
    """Returns the signal to noise ratio of the measurements."""
    return round(len(df_masked)/(len(df_raw) - len(df_masked)), 2)

def mask_pixels(df):
    df_raw = df.copy()
    df = mask_outliers(mask_bad_pixels(df_raw))
    print(f"SNR: {snr(df_raw, df)} ({round(1 - len(df)/len(df_raw), 3)*100}% noise)")
    return df

def plot_beam(df):
    """Plots the beam."""
    df = mask_pixels(df)
    x_col, y_col, m_col = get_m_counts(df)
    fig, ax  = plt.subplots()
    ax.set_aspect('equal')
    im = ax.scatter(x_col, y_col, c=m_col, s=1, cmap='jet', marker = "s", vmax = np.max(m_col), vmin = np.min(m_col))
    ax.plot(x_col[np.argmax(m_col)], y_col[np.argmax(m_col)], marker = "+", color = 'red', markersize=10, label = 'max', linestyle = 'None')
    print("Max (x, y, counts):" , (x_col[np.argmax(m_col)], y_col[np.argmax(m_col)], np.max(m_col)))
    ax.set_xlabel('x pixel number')
    ax.set_ylabel('y pixel number')
    cbar = fig.colorbar(im, ax = ax)
    cbar.set_label('counts')
    ax.set_title('Colimated beam')

def get_events(df, event_length, upper_limit, mask):
    """Returns a dataframe total tot and energy of each event detected in the beam. Upper limit in energy."""
    if mask:
        df = mask_pixels(df)
    df = index_events(df, event_length)
    events = df[['event_index', 'tot', 'energy']].groupby('event_index').sum()
    mask = (events['energy'] < upper_limit)
    print(f'Number of events above limit: {np.round((1 - np.sum(mask)/len(mask))*100, 2)} %')
    events = events[mask]
    print(f'Number of remaining events: {round(len(events)/1000)}k')
    return events

def observation_time(df):
    """Returns the observation time of the measurement in seconds."""
    return (df['toa'].max() - df['toa'].min())*1E-9 # in seconds

def combine_events(files, event_length, upper_lim, mask = False):
    """Returns a dataframe with the total tot and energy of each event detected in the beam. Upper limit in energy.

    Parameters
    ----------
    files : list
        List of filepaths of the files to be combined.
    event_length : int
        Length of the event in ns.
    upper_lim : int
        Upper limit in energy.
    mask : bool
        If True, a beam mask is applied to the events. Default is False.

    Returns
    -------
    events : dataframe
        Dataframe with the total tot and energy of each event detected in the beam.
    obs_time : float
        Observation time of the measurement in seconds.        

    """

    events = []
    obs_time = 0
    
    print('Reading calibration constants...')
    const = get_const()
    

    for file in files:
        try:
            df = get_df(file, const)
        except (ValueError, UnicodeDecodeError, UserWarning):
            file_fixed = fix_file(file)
            if os.path.getsize(file_fixed) == 0:
                print(f'File {os.path.basename(file)} is empty. Continuing with next file...')
                continue
            df = get_df(file_fixed, const)
        obs_time += observation_time(df)
        events.append(get_events(df, event_length, upper_lim, mask))
    
    if len(events) > 1:
        events = pd.concat(objs = events, ignore_index=True).sort_values(by = 'tot')
    elif len(events) == 1:
        events = events[0]
    else:
        print('No events found. Exiting...')
        sys.exit()
    return events, obs_time

def plot_hist(events, title):
    """Plots the histogram of the events with bins in tot and energy."""
    events.hist(bins=500, figsize=(20,10))
    plt.suptitle(title)


# fitting --------------------------------------------------------------------

def plot_raw_spectrum(element, events, e_tot):
    plt.plot(*events[element][:2], linestyle = '-', label = element, linewidth = 1, alpha = 0.9)
    plt.grid(linestyle='dotted')
    plt.yscale('log')
    if e_tot == 'tot':
        plt.xlabel('TOT / ticks')
    else:
        plt.xlabel('energy / keV')
    plt.ylabel('counts per hour')

def plot_average(events, title, n, i):
    cmap = plt.get_cmap('tab20')
    i *= 2
    events[title] = events[title].T[events[title][0] > 5].T
    avg = np.convolve(events[title][1], np.ones(n)/n, mode='valid')
    plt.plot(events[title][0][n-1:] - n/2, avg, label = f'{title}: Average per ' + str(n) + ' keV', zorder = 10, color = cmap(i))
    plt.errorbar(*events[title][:2], fmt='.', yerr = events[title][2], label=f'{title}: Raw Data', ecolor = "orange", zorder = 1, color = cmap(i+1))
    plt.title(title)
    plt.xlabel('energy / keV')
    plt.ylabel('counts per hour')
    plt.legend()

def plot_error(events, title):
    plt.errorbar(*events[title][:2], yerr = events[title][2], fmt = ".", ecolor = "r", capsize = 2, alpha = 0.8)
    plt.title(title)
    plt.xlabel("energy / keV")
    plt.ylabel("counts per hour")

def load_single_event(dir, key, events):
    file = f"spectra_raw/{dir}/{key}.txt"
    if os.path.isfile(file):
        events[key] = np.loadtxt(file, skiprows=1, unpack=True)
        plot_raw_spectrum(key, events, 'energy')

def load_events(date, dir_list):
    events = {}
    if type(dir_list) == str:
        dir_list = [dir_list]
    for dir in dir_list:
        for key in date.keys():
            load_single_event(dir, key, events)
    return events

def myround(x, base = 0.5):
    """Rounds x to the nearest base that can be a float."""
    return base * np.round(x/base)

def print_obs_time(name, obs_time):
    """Prints the observation time in hours."""
    print(f'{name}: {obs_time//3600}h {obs_time%3600//60}m {round(obs_time%60,2)}s')

def get_counts(df, obs_time, e_tot):
    """Returns the counts of events per hour for each tot or energy."""
    tot, counts = np.unique(df[e_tot].round().values, return_counts=True)
    counts_err = np.sqrt(counts)
    counts_err = counts_err + (counts_err == 0) # account for cases where counts == 0
    return tot, counts/(obs_time/3600), counts_err/(obs_time/3600)  # tot/energy, counts per hour, counts_err per hour


def gaussian(x, mu, sig, A):
    return A/(np.sqrt(2*np.pi)*sig)*np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2)))

def erfunc(x, mu, sig, A, H):
    return H*A/(np.sqrt(2*np.pi)*sig)*erfc((x-mu)/(sig*np.sqrt(2)))

def linear(x, d):
    return x*0 + d
    

def single(x, mu, sig, A, H, d):
    return gaussian(x, mu, sig, A) + erfunc(x, mu, sig, A, H) + linear(x, d)

def double(x, mu1, sig1, A1, H1, mu2, sig2, A2, H2, d):
    return gaussian(x, mu1, sig1, A1) + erfunc(x, mu1, sig1, A1, H1) + gaussian(x, mu2, sig2, A2) + erfunc(x, mu2, sig2, A2, H2) + linear(x, d)

def fit_gaussian(tot, counts, counts_err, mask):
    tot = tot[mask]
    counts = counts[mask]
    counts_err = counts_err[mask]

    p0 = [np.mean(tot), np.std(tot), np.sum(counts), np.sum(tot)*0.001, 0]
    bounds = ([np.min(tot), 1, 0, 0, 0], [np.max(tot), 200, np.inf, np.inf, np.percentile(counts, 50).mean()])

    popt, pcov = curve_fit(single, tot, counts, sigma = counts_err, absolute_sigma=True, p0=p0, bounds = bounds, maxfev=10000)

    return popt, pcov, mask

def fit_double_gaussian(tot, counts, counts_err, mask, mask1, mask2):
    p0 = [np.mean(tot[mask1]), np.std(tot[mask1]), np.sum(counts[mask1]), np.sum(tot[mask1])*0.001, np.mean(tot[mask2]), np.std(tot[mask2]), np.sum(counts[mask2]), np.sum(tot[mask2])*0.001, 0]
    bounds = ([np.min(tot[mask1]), 1, 0, 0, np.min(tot[mask2]), 1, 0, 0, 0], [np.max(tot[mask1]), 100, np.inf, np.inf, np.max(tot[mask2]), 100, np.inf, np.inf, np.percentile(counts[np.logical_or(mask1, mask2)], 50).mean() ])
    popt, pcov = curve_fit(double, tot[mask], counts[mask], sigma=counts_err[mask], absolute_sigma=True, p0= p0, bounds=bounds, maxfev=10000)
    return popt, pcov, mask

def append_dict(cal_dict, e0, popt, pcov):
    error = np.sqrt(np.diag(pcov))
    cal_dict['e0'].append(e0)
    cal_dict['energy'].append(popt[0])
    cal_dict['energy_err'].append(error[0])
    cal_dict['sigma'].append(popt[1])
    cal_dict['sigma_err'].append(error[1])
    cal_dict['A'].append(popt[2])
    cal_dict['A_err'].append(error[2])

def do_double_gaussian_fit(peak_dict, tot, counts, counts_err, element, i):
    mask = np.logical_and(tot >= min(peak_dict[element]['lower'][i]), tot <= max(peak_dict[element]['upper'][i]))
    mask1 = np.logical_and(tot >= peak_dict[element]['lower'][i][0], tot <= peak_dict[element]['upper'][i][0])
    mask2 = np.logical_and(tot >= peak_dict[element]['lower'][i][1], tot <= peak_dict[element]['upper'][i][1])
    return fit_double_gaussian(tot, counts, counts_err, mask, mask1, mask2)

def do_single_gaussian_fit(peak_dict, tot, counts, counts_err, element, i):
    mask = np.logical_and(tot >= peak_dict[element]['lower'][i], tot <= peak_dict[element]['upper'][i])
    return fit_gaussian(tot, counts, counts_err, mask)

def fit_all(element, events, peak_dict):

    calib_dict = {
     'e0':[],
     'energy':[],
     'energy_err':[],
     'sigma':[],
     'sigma_err':[],
     'A':[],
     'A_err':[]
     }
    
    tot, counts, counts_err = events[element]
    for i, energy in enumerate(peak_dict[element]['e0']):
        if isinstance(energy, tuple):
            popt, pcov, mask = do_double_gaussian_fit(peak_dict, tot, counts, counts_err, element, i)
            append_dict(calib_dict, energy[0], popt[:3], pcov[:3,:3])
            append_dict(calib_dict, energy[1], popt[4:7], pcov[4:7, 4:7])
        else:
            popt, pcov, mask = do_single_gaussian_fit(peak_dict, tot, counts, counts_err, element, i)
            append_dict(calib_dict, energy, popt[:3], pcov[:3,:3])
    return pd.DataFrame(data = calib_dict)

def label_gaussian(e0, popt, pcov):
    error = np.sqrt(np.diag(pcov))
    return f"{e0} keV\n" + r"$\mu$" + f" = {popt[0]:.2f}" + r"$\pm$" + f" {error[0]:.1f}\n" + r"$\sigma$"+f" = {popt[1]:.1f} " + r"$\pm$" + f" {error[1]:.1f}\nA = {popt[2]:.0f} " + r"$\pm$" + f" {error[2]:.0f}"

def label_double_gaussian(e01, e02, popt, pcov):
    error = np.sqrt(np.diag(pcov))
    return f"{e01} keV\n"+r"$\mu_1$"+f" = {popt[0]:.2f} " + r"$\pm$" + f" {error[0]:.1f}\n"+r"$\sigma_1$"+f" = {popt[1]:.1f} " + r"$\pm$" + f" {error[1]:.1f}\n$A_1$ = {popt[2]:.0f} " + r"$\pm$" + f" {error[2]:.0f}\n\n{e02} keV\n"+r"$\mu_2$"+f" = {popt[4]:.2f} " + r"$\pm$" + f" {error[4]:.1f}\n"+r"$\sigma_2$"+f" = {popt[5]:.1f} " + r"$\pm$" + f" {error[5]:.1f}\n$A_2$ = {popt[6]:.0f} " + r"$\pm$" + f" {error[6]:.0f}"
    
def plot_fit(element, events, peak_dict):
    tot, counts, counts_err = events[element]
    plt.plot(tot, counts, '+')
    for i, energy in enumerate(peak_dict[element]['e0']):
        if isinstance(energy, tuple):
            popt, pcov, mask = do_double_gaussian_fit(peak_dict, tot, counts, counts_err, element, i)
            plt.plot(tot[mask], double(tot[mask], *popt), label=label_double_gaussian(energy[0], energy[1], popt, pcov) + '\n')
        else:
            popt, pcov, mask = do_single_gaussian_fit(peak_dict, tot, counts, counts_err, element, i)
            plt.plot(tot[mask], single(tot[mask], *popt), label=label_gaussian(energy, popt, pcov) + '\n')
    ticks = np.arange(0, np.max(tot)+100, 100)
    plt.ylim(1, np.max(counts)*1.3)
    plt.xlim(ticks[0], ticks[-1])
    plt.yscale('log')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.xticks(ticks)
    plt.grid(linestyle = 'dotted')
    plt.xlabel('energy / keV')
    plt.ylabel('counts per hour')
    plt.title(element + 'pre-calibration')

def chi2_red(y, yfit, sigma, dof):
    return np.sum((y - yfit)**2 / sigma**2)/dof

def plot_single_fit(element, events, peak_dict, energy_index):
    """Plots the fit of a single peak.
    Parameters
    ----------
    element : str
        Element to be fitted.
    events : dict
        Dictionary of events.
    peak_dict : dict
        Dictionary of peaks.
    energy_index : int
        Index of the energy to be fitted.
    """
    
    tot, counts, counts_err = events[element]
    energy = peak_dict[element]['e0'][energy_index]
    if isinstance(energy, tuple):
        #print('double')
        popt, pcov, mask = do_double_gaussian_fit(peak_dict, tot, counts, counts_err, element, energy_index)
        tfine = np.linspace(tot[mask][0], tot[mask][-1], 1000)
        plt.errorbar(tot[mask], counts[mask], fmt = '+', label='data', yerr =  counts_err[mask], ecolor='red', 
                     capsize=1.5, elinewidth=0.5, markeredgewidth=1.5, capthick=0.05, zorder = 1, alpha = 0.8)
        plt.plot(tfine, double(tfine, *popt), label='fit')
        plt.plot(tfine, gaussian(tfine, *popt[:3]) + gaussian(tfine, *popt[4:7]), label='gaussian', linestyle = '--')
        plt.plot(tfine, erfunc(tfine, *popt[:4]) + erfunc(tfine, *popt[4:8]), label='erfc', linestyle='--')
        chi2 = chi2_red(counts[mask], double(tot[mask], *popt), np.sqrt(counts[mask]), len(tot[mask])-len(popt))
        plt.text(0.03, 0.6, label_double_gaussian(*energy, popt, pcov) + '\n' + r"$\chi^2_{red}$ = " + 
                 f"{round(chi2, 2)}", transform=plt.gca().transAxes, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 8})
    else:
        #print('single')
        popt, pcov, mask = do_single_gaussian_fit(peak_dict, tot, counts, counts_err, element, energy_index)
        tfine = np.linspace(tot[mask][0], tot[mask][-1], 1000)
        plt.errorbar(tot[mask], counts[mask], fmt = '+', label='data', yerr =  counts_err[mask], ecolor='red',capsize=1.5, elinewidth=0.8, capthick=1, zorder = 1, alpha = 0.8)
        plt.plot(tfine, single(tfine, *popt), label='fit')
        plt.plot(tfine, gaussian(tfine, *popt[:3]), label='gaussian', linestyle = '--')
        plt.plot(tfine, erfunc(tfine, *popt[:4]), label='erfc', linestyle='--')
        chi2 = chi2_red(counts[mask], single(tot[mask], *popt), np.sqrt(counts[mask]), len(tot[mask])-len(popt))
        plt.text(0.03, 0.7, label_gaussian(energy, popt, pcov) + '\n' + r"$\chi^2_{red}$ = " + f"{round(chi2, 2)}", 
                 transform = plt.gca().transAxes, bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 8})
    plt.plot(tfine, linear(tfine, popt[-1]), label='constant background', linestyle='--')    
    #print('fit parameters = ', popt)
    #print('error', np.sqrt(np.diag(pcov)))
    plt.xlabel('energy / keV')
    plt.ylabel('counts per hour')
    plt.legend()
    plt.title(f'{element}, {energy} keV')
    plt.ylim(0, np.max(counts[mask])*1.2)
    plt.xlim(tot[mask][0], tot[mask][-1])
    
    fit_parameters = popt
    error = np.sqrt(np.diag(pcov))

    return fit_parameters, error 


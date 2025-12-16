import numpy as np
import math
from datetime import datetime as dt
import matplotlib.pyplot as plt

FWHM = 2*np.sqrt(2*np.log(2))

MC_TO_BQ = 37e3

# Activity of sources in Bq, their time of calibration, and half-life in years
ACT = {
    'Na22': (51.25e3, dt(2023, 6, 19), 2.602),
    #'Cs137': (260e3, dt(2023, 6, 19), 30.08), # wrong value!
    'Cs137': (100*MC_TO_BQ, dt(1995, 1, 1), 30.08),
    #'Ba133': (251.6e3, dt(2020, 8, 25), 10.52), # wrong value!
    'Ba133': (100*MC_TO_BQ, dt(1995, 1, 1), 10.52),
    'Eu152': (0.5*MC_TO_BQ, dt(1993, 4, 15), 13.522),
    'Eu154': (18.50e3, dt(2018, 6, 1), 8.601),
}

ACT_ERR = 0.15 # 15% uncertainty of the activity

# other constants
FWHM = 2*np.sqrt(2*np.log(2))
AREA_DETECTOR = 256**2 * (55E-3)**2 # mm^2

# ------------------- error propagation ------------------- #

def sqrt_err(x, xerr):
    """Returns the error of the square root of x."""
    return 0.5*xerr/np.sqrt(x)

def z_err(z, x, y, xerr, yerr):
    """Error propagation for z = x/y or x*y."""
    return z*np.sqrt((xerr/x)**2 + (yerr/y)**2)

# ------------------- energy calibration ------------------- #

def calibration(e_pre, a, b):
    """Returns the re-calibrated energy. Input are the calibration parameters a and b, and the pre-calibrated energy."""
    return (e_pre - b)/a

def calibration_err(e_pre, a, b, aerr, berr):
    """Calculate the error of the calibration."""
    return np.sqrt((e_pre*berr/a)**2 + ((e_pre - b)/a**2*aerr)**2)

# ------------------- resolution ------------------- #

def resolution(energy, sigma):
    """Returns the relative resolution."""
    return sigma*FWHM/energy
    
def res_func(energy, a, b, c):
    """Function to fit the resolution over energy."""
    return a**2*energy**(-2) + b**2*energy**(-1) + c**2

def res_func_err(energy, a, b, c, aerr, berr, cerr):
    """Error of the resolution function."""
    res = res_func(energy, a, b, c)
    res_err = np.sqrt((2*a*aerr*energy**(-2))**2 + (2*b*berr*energy**(-1))**2 + (2*c*cerr)**2)
    return sqrt_err(res, res_err)

def get_res(energy, popt, pcov):
    """Returns the resolution determined with the resolution function and its error in percent."""
    return np.sqrt(res_func(energy, *popt))*100, res_func_err(energy, *popt, *np.sqrt(np.diag(pcov)))*100

def res_func_abs(energy, a, b):
    """Function to fit the absolute resolution over energy."""
    return a*energy + b

def get_resolution(df, key):
    """Returns the resolution and its error for a given key."""
    df = df.xs(level=0, key = key)
    res = resolution(df['energy'], df['sigma'])
    res_err = z_err(res, df['energy'], df['sigma'], df['energy_err'], df['sigma_err'])
    return np.array([res, res_err])

def factory_res():
    """Returns the resolution of the factory values in percent."""
    factory_res = np.array([(8.3+2.9)/2, (9.9+4.5)/2])
    factory_res_err = np.array([8.3-factory_res[0], 9.9-factory_res[1]])
    energy = np.array([60, 122])

    rel = factory_res*FWHM/energy * 100
    rel_err = factory_res_err*FWHM/energy * 100

    print(f'60 keV: ({rel[0]:.1f} \u00B10 {rel_err[0]:.1f}) %')
    print(f'122 keV: ({rel[1]:.1f} \u00B10 {rel_err[1]:.1f}) %')

    return rel, rel_err

def combine_new_key(df, func_str, **kwargs):
    """Add two new columns (values, errors) to a dataframe calculated by a function called by the name of the function."""
    func = globals()[func_str]
    for i, key in enumerate(df.index.get_level_values(0).unique()):
        if i == 0:
            arr = func(df, key, **kwargs)
        else:
            arr = np.append(arr, func(df, key, **kwargs), axis = 1)
    return arr.T

# ------------------- efficiency ------------------- #

def log_eff(energy, a, b):
    """Function to fit the efficiency over energy in log scale."""
    return a + b*np.log(energy)

def convert_frac(lst):
    """Converts a list of fractions to a numpy array of floats."""
    return np.array(lst, dtype=float)/100 #convert from percent

def get_A_norm(df, key, peak_dict):
    """Returns the normalized number of counts taking the emission fraction into account and its error for a given key."""
    df = df.xs(level=0, key = key)
    A_norm = (df['A']/convert_frac(peak_dict[key]['fractions'])).values
    A_norm_err = z_err(A_norm, df['A'], convert_frac(peak_dict[key]['fractions']), df['A_err'], convert_frac(peak_dict[key]['fractions_err'])).values
    return np.array([A_norm, A_norm_err])

def area(r):
    return np.pi*r**2

def sphere(r):
    return 4*np.pi*r**2

def g_factor(det_area, d):
    """Returns the geometry-factor for a given detector area and distance. Usually g << 1."""
    return det_area/sphere(d) #d in mm, det_area in mm^2

def activity(A_0, half_life, time):
    """Returns the activity of a radioactive element at a given time."""
    return A_0*np.exp(-np.log(2)*time/half_life) # T in years, activity in 1/second

def time_diff(element, date):
    """Returns the time difference between the measurement and the last calibration of the source."""
    elm = element.split('-')[0]
    return (date[element][0] - ACT[elm][1]).days/365.25

def get_A_rel(df, key, peak_dict):
    """Returns the relative number of counts taking the emission fraction into account and its error for a given key."""
    df = df.xs(level=0, key = key)
    frac = convert_frac(peak_dict[key]['fractions'])
    total_frac = {
        'Na22': 2.8,
        'Cs137': 0.918,
        'Ba133': 2.55,
        'Eu152': 2.07,
        'Eu154': 1.8,
    }
    frac_rel = frac/total_frac[key.split('-')[0]]
    A_rel = (df['A']/frac_rel).values
    A_rel_err = z_err(A_rel, df['A'], convert_frac(peak_dict[key]['fractions']), df['A_err'], convert_frac(peak_dict[key]['fractions_err'])).values
    return np.array([A_rel, A_rel_err])

def relative_eff(A_rel, A_rel_err, element, events_dict):
    """Returns the relative efficiency and its error for a given element."""
    return A_rel/np.sum(events_dict[element][1]), A_rel_err/np.sum(events_dict[element][1])

def intrinsic_eff(A_norm, A_norm_err, element, date):
    """Returns the intrinsic efficiency and its error for a given element."""
    elm = element.split('-')[0]
    act = activity(ACT[elm][0], ACT[elm][2], time_diff(elm, date))
    abs_eff = A_norm/(act*3600)
    abs_eff_err = z_err(abs_eff, act, A_norm, act*ACT_ERR, A_norm_err)
    _g_factor = g_factor(AREA_DETECTOR, date[element][1])
    _g_factor_err = _g_factor*2*date[element][2]/date[element][1]
    return abs_eff/_g_factor, z_err(abs_eff/_g_factor, abs_eff, _g_factor, abs_eff_err, _g_factor_err) 

def get_eff(df, key, eff_func, **kwargs):
    """Returns either the relative or intrinsic efficiency and its error for a given key."""
    df = df.xs(level=0, key = key)
    eff_func = globals()[eff_func]
    if eff_func == intrinsic_eff:
        eff, eff_err =  eff_func(df['A_norm'].values, df['A_norm_err'].values, key, kwargs['date'])
    elif eff_func == relative_eff:
        eff, eff_err =  eff_func(df['A_rel'].values, df['A_rel_err'].values, key, kwargs['events_dict'])
    else:
        print('Error: eff_func not defined')
    return np.array([eff, eff_err])

def eff_energy_err(eff, energy, aerr, berr):
    """Error propagation for the efficiency over energy."""
    return eff*np.sqrt((aerr)**2 + (berr*np.log(energy))**2)

def eff_energy(energy, popt, pcov):
    """Returns the efficiency over energy and its error in percent."""
    eff = np.exp(log_eff(energy, *popt))
    eff_err = eff_energy_err(eff, energy, *np.sqrt(np.diag(pcov)))
    return eff*100, eff_err*100

# ------------------- angle dependence ------------------- #

def get_angles(df):
    """Returns the angles from the spectras."""
    return np.array([int(angle.split('-')[1]) for angle in df.index.get_level_values(0)])

def plot_angle_dependence(df, key):
    """Plots the angle dependence of a parameter of the detector."""
    angles = get_angles(df)

    fig, ax = plt.subplots()
    for energy in df.index.get_level_values(1).unique():
        mask = df.index.get_level_values(1) == energy
        ax.errorbar(angles[mask], df[key][mask]*100, yerr = df[key + "_err"][mask]*100, fmt = 'o', label=f'{energy} keV', markersize=3, capsize=2, linestyle='--')
    ax.set_xlabel('angle / degrees')
    ax.set_xticks(np.unique(angles))

    return fig, ax

# temperature dependence

def get_temperatures():
    """Returns the temperatures during the acquisitions and their errors."""
    temperature = {
        15:[[15, 15.9, 15.7], [0.5,0.2,0.6]],
        20:[[21.9, 21.7, 21.6],[0.3,0.1,0.6]],
        25:[[25.4, 26.1, 26.4],[0.5, 0.7, 0.7]],
        30:[[31.9, 32.2, 31.4],[1.3, 1.3, 0.7]]
    }

    temp = []
    temp_err = []

    for key, item in temperature.items():
        temp.append(np.mean(item[0]))
        temp_err.append(np.sqrt(np.sum(np.array(item[1])**2)))
    return temp, temp_err

def plot_temp_dependence(df, key):
    """Plots the temperature dependence of a parameter of the detector."""
    temp, temp_err = get_temperatures()

    fig, ax = plt.subplots()
    for energy in df.index.get_level_values(1).unique():
        mask = df.index.get_level_values(1) == energy
        ax.errorbar(temp, df[key][mask]*100, xerr = temp_err, yerr = df[key + "_err"][mask]*100, fmt = 'o', label=f'{energy} keV', markersize=3, capsize=2, linestyle='--')
    ax.set_xlabel('temperature / deg C')
    ax.set_xticks(np.unique(temp))

    return fig, ax

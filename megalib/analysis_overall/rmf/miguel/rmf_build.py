import os
from datetime import datetime
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path

from configurations.path_config import ResultsPaths
from .read_spectral_csv import read_csv, csv_validity_date
from .rmf_plots import plot_rmf, plot_fwhm

# ============================================================================
# Models and Fitting
# ============================================================================

def Gaussian(E, A0, A1, A2):
    return A0 * np.exp(-(E - A1) ** 2 / (2 * A2 ** 2)) / (A2 * np.sqrt(2 * np.pi))

def DeltaE(E, F, ENC):
    """Calculate energy resolution (FWHM) in eV."""
    upsilon = 3.62
    E_eV = E * 1000
    return 2.35 * np.sqrt(F * upsilon * E_eV + (ENC * upsilon) ** 2)

def fit_noise_model(energy, fwhm):
    """Fit the detector noise model using finite FWHM points only."""
    valid = np.isfinite(energy) & np.isfinite(fwhm)
    energy = np.asarray(energy, dtype=float)[valid]
    fwhm = np.asarray(fwhm, dtype=float)[valid]

    if len(energy) < 2:
        raise ValueError('Need at least two valid points to perform the fit')

    params, cov = curve_fit(DeltaE, energy, fwhm, p0=[0.137, 4.7])
    fitted = DeltaE(energy, *params)
    residuals = fwhm - fitted
    chisq = np.sum(residuals ** 2)
    dof = len(energy) - len(params)
    return params, cov, fitted, residuals, chisq, dof

def deltae_uncertainty(E, params, cov):
    """Propagate the fit covariance to the DeltaE uncertainty at energy E."""
    if cov is None:
        return np.nan

    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return np.nan

    upsilon = 3.62
    E_eV = E * 1000
    argument = upsilon * E_eV * params[0] + (upsilon * params[1]) ** 2
    if argument <= 0:
        return np.nan

    sqrt_argument = np.sqrt(argument)
    grad_f = 2.35 * upsilon * E_eV / (2 * sqrt_argument)
    grad_enc = 2.35 * (upsilon ** 2) * params[1] / sqrt_argument
    gradient = np.array([grad_f, grad_enc], dtype=float)
    variance = float(gradient @ cov @ gradient.T)

    if variance < 0:
        return np.nan

    return np.sqrt(variance)

# =============================================================================
# Al K-alpha Extrapolation
# =============================================================================

def add_extrapolated_row_to_csv(csv_file, energy_value, fwhm_value, fwhm_error):
    """Insert an extrapolated spectral row in the source CSV, sorted by energy."""
    csv_path = ResultsPaths.CAL_SPEC_RESULTS / csv_file
    df = pd.read_csv(csv_path)

    energy_series = pd.to_numeric(df['Energy_keV'], errors='coerce')
    if np.any(np.isclose(energy_series.to_numpy(dtype=float), energy_value, rtol=0.0, atol=1e-6)):
        print(f'Al K-alpha ({energy_value:.3f} keV) already exists in {csv_file}; CSV not modified.')
        return

    new_row = {column: np.nan for column in df.columns}
    new_row['Energy_keV'] = energy_value
    new_row['LinePos_keV'] = np.nan
    new_row['LinePos_Error_keV'] = np.nan
    new_row['FWHM_eV'] = fwhm_value
    new_row['FWHM_Error_eV'] = fwhm_error
    new_row['extrapolate'] = 1

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df = df.sort_values(by='Energy_keV', kind='mergesort', na_position='last').reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f'Added extrapolated Al K-alpha row to {csv_path}')

def maybe_add_al_kalpha_point(selected_data, params, cov, energy_value=1.486):
    """Optionally add an extrapolated Al K-alpha row into the selected dataset."""
    energy = np.asarray(selected_data['Nominal_Energy'], dtype=float)
    if np.any(np.isclose(energy, energy_value, rtol=0.0, atol=1e-6)):
        index = np.where(np.isclose(energy, energy_value, rtol=0.0, atol=1e-6))[0][0]
        return float(selected_data['FWHM_0'][index])

    csv_file = selected_data.get('csv_file', 'unknown')
    csv_path = ResultsPaths.CAL_SPEC_RESULTS / csv_file
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        csv_energy = pd.to_numeric(df['Energy_keV'], errors='coerce').to_numpy(dtype=float)
        if np.any(np.isclose(csv_energy, energy_value, rtol=0.0, atol=1e-6)):
            index = np.where(np.isclose(csv_energy, energy_value, rtol=0.0, atol=1e-6))[0][0]
            fwhm_series = pd.to_numeric(df['FWHM_eV'], errors='coerce').to_numpy(dtype=float)
            print(f'Al K-alpha ({energy_value:.3f} keV) already exists in {csv_file}.')
            return float(fwhm_series[index])

    add_choice = input(
        f'Al K-alpha ({energy_value:.3f} keV) is not in the selected data. Add extrapolated row? (y/n): '
    ).strip().lower()
    if add_choice not in {'y', 'yes'}:
        return float(DeltaE(energy_value, *params))

    fwhm_value = float(DeltaE(energy_value, *params))
    fwhm_error = deltae_uncertainty(energy_value, params, cov)
    insert_index = int(np.searchsorted(energy, energy_value))
    row_count = len(energy)

    selected_data['Nominal_Energy'] = np.insert(energy, insert_index, energy_value)
    selected_data['FWHM_0'] = np.insert(np.asarray(selected_data['FWHM_0'], dtype=float), insert_index, fwhm_value)

    if selected_data.get('FWHM_err') is None:
        fwhm_err = np.full(row_count, np.nan, dtype=float)
    else:
        fwhm_err = np.asarray(selected_data['FWHM_err'], dtype=float)
    selected_data['FWHM_err'] = np.insert(fwhm_err, insert_index, fwhm_error)

    if selected_data.get('Centroid_keV') is None:
        centroid = np.full(row_count, np.nan, dtype=float)
    else:
        centroid = np.asarray(selected_data['Centroid_keV'], dtype=float)
    selected_data['Centroid_keV'] = np.insert(centroid, insert_index, np.nan)

    if selected_data.get('Centroid_err_keV') is None:
        centroid_err = np.full(row_count, np.nan, dtype=float)
    else:
        centroid_err = np.asarray(selected_data['Centroid_err_keV'], dtype=float)
    selected_data['Centroid_err_keV'] = np.insert(centroid_err, insert_index, np.nan)

    if selected_data.get('extrapolate') is None:
        extrapolate = np.zeros(row_count, dtype=int)
    else:
        extrapolate = np.asarray(selected_data['extrapolate'], dtype=int)
    selected_data['extrapolate'] = np.insert(extrapolate, insert_index, 1)

    add_extrapolated_row_to_csv(csv_file, energy_value, fwhm_value, fwhm_error)

    print(
        f'\nAl K-alpha row at {energy_value:.3f} keV '
        f'(FWHM={fwhm_value:.2f} eV, error={fwhm_error:.2f} eV).'
    )
    return fwhm_value

# ============================================================================
# RMF Building
# ============================================================================

def build_rmf(params, bins, E_Si=1.75):
    """Build the RMF matrix from the fitted FWHM model."""
    E_edges = np.linspace(0.1, 10.0, bins)
    E_centers = (E_edges[:-1] + E_edges[1:]) / 2

    RMF = np.array([
        Gaussian(E_centers * 1000, 1.0, energy * 1000, DeltaE(energy, *params) / 2.36)
        for energy in E_centers
    ]).T

    tail = np.array([
        Gaussian(E_centers * 1000, -0.00127465 * energy + 0.01104828,
                 (energy - E_Si) * 1000, DeltaE(energy, *params) / 2.36)
        if energy > E_Si else np.zeros_like(E_centers)
        for energy in E_centers
    ]).T

    RMF += tail
    RMF /= np.sum(RMF, axis=0)
    return RMF

def select_data_for_rmf():
    """List all available CSV files and ask the user which one to process."""
    spectral_data = read_csv()
    print('\nAvailable spectral results files:')
    for i, data in enumerate(spectral_data):
        source = data.get('source', 'unknown')
        date = data.get('date')
        print(f"{i}: {data['csv_file']}  [{source}]  {date or 'n/a'}")

    idx = int(input('Select CSV file index to build RMF from: '))
    if idx < 0 or idx >= len(spectral_data):
        raise IndexError('Selected index is outside the available data range')
    return spectral_data[idx]

# ===========================================================================
# Save RMF Fits
# ===========================================================================

def save_fits_calibration(RMF, selected_data, save_folder=None):
    matrix = np.array(RMF, dtype=np.float64, copy=True).T
    if matrix.shape != (1024, 1024):
        raise ValueError('Calibration RMF must have shape (1024, 1024)')
    if not np.isfinite(matrix).all():
        raise ValueError('Calibration RMF contains NaN or infinite values')
    np.maximum(matrix, 0.0, out=matrix)
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError('Calibration RMF contains an empty probability row')
    matrix /= row_sums

    csv_file = selected_data.get('csv_file', 'unknown')
    oldest_date = csv_validity_date(csv_file)
    CVSD0001 = oldest_date.strftime('%Y-%m-%d')
    CVST0001 = oldest_date.strftime('%H:%M:%S')
    date_name = oldest_date.strftime('%Y%m%dT%H%M%S')

    date_prod = datetime.utcnow()
    strDate = date_prod.strftime('%Y-%m-%dT%H:%M:%S')
    strDateName = date_prod.strftime('%Y%m%dT%H%M%S')

    calname = 'MXT-GND-MATRIX-ALL'
    filename = f'{calname}_{date_name}_{strDateName}.fits'
    if save_folder is None:
        folder = ResultsPaths.RMF
    else:
        folder = Path(save_folder)
    filepath = folder / filename

    hdr = fits.Header()
    hdr['EXTNAME'] = 'MATRIX'
    hdr['PRDM_VER'] = (1, 'Product Data Model Version')
    hdr['LONGSTRN'] = ('OGIP 1.0', 'The HEASARC Long String Convention may be used.')
    hdr['TELESCOP'] = ('SVOM', 'Telescope or mission name')
    hdr['INSTRUME'] = ('MXT', 'Instrument name')
    hdr['ORIGIN'] = ('MIC', 'Origin of FITS file')
    hdr['CREATOR'] = (os.path.basename(__file__), 'Program that created this FITS file')
    hdr['CONFIGUR'] = ('1.0.0', 'Software configuration')
    hdr['DATE'] = (strDate, 'FITS file creation date')
    hdr['DATATYPE'] = ('Calib', 'Data channel media')
    hdr['PROC_VER'] = (1, 'pre-processing version')
    hdr['MJDREFI'] = (57754, 'int. portion in MJD for reference 01/01/2017')
    hdr['MJDREFF'] = (0.0, 'fractional portion in MJD for reference time')
    hdr['TIMESYS'] = ('UTC', 'Time frame system')
    hdr['TIMEUNIT'] = ('s', 'Time unit')
    hdr['TIMEREF'] = ('LOCAL', 'Time reference frame')
    hdr['TSTART'] = (0.0, 'start UTC of processed data to create file')
    hdr['TSTOP'] = (0.0, 'last UTC of processed data to create file')
    hdr['CCLS0001'] = ('BCF', 'HEASARC class, dataset is Basic Calibration')
    hdr['CDTP0001'] = ('DATA', 'HEASARC code, dataset is real data')
    hdr['CCNM0001'] = ('MATRIX-ALL', 'HEASARC codename, dataset name')
    hdr['CBD10001'] = ('CHAN(1-1024)', 'HEASARC parameter limitation')
    hdr['CBD20001'] = ('ENER(0.1-10.0)keV', 'HEASARC parameter limitation')
    hdr['CVSD0001'] = (CVSD0001, 'HEASARC start date of validity yyyymmdd')
    hdr['CVST0001'] = (CVST0001, 'HEASARC start time of validity hhmmss')
    hdr['CDES0001'] = (f'RMF created with data from CSV file {csv_file}', 'HEASARC brief desc')
    hdr['FILTER'] = ('', '')
    hdr['CHANTYPE'] = ('PI', 'detector channels are corrected')
    hdr['DETCHANS'] = (1024, 'total number of raw detector channels')
    hdr['HDUCLASS'] = ('OGIP', '')
    hdr['HDUCLAS1'] = ('RESPONSE', 'extension contains response data')
    hdr['HDUCLAS2'] = ('RSP_MATRIX', 'extension contains a response matrix')
    hdr['HDUCLAS3'] = ('REDIST', 'optional, photon redistribution process only')
    hdr['HDUVERS'] = ('1.3.0', 'obsolete, for old software')
    hdr['RMFVERSN'] = ('1992a', 'obsolete, for old software')
    hdr['TLMIN4'] = (1, 'first channel, minimum of F_CHAN column')
    hdr['LO_THRES'] = (0, 'optional minimum proba threshold, below is 0')
    hdr['EFFAREA'] = 1
    hdr['NUMGRP'] = (1024, 'optional, the sum of the N_GRP column')
    hdr['NUMELT'] = (matrix.size, 'optional, the sum of the N_CHAN column')

    E_RMF = np.linspace(0.1, 10.0, RMF.shape[0] + 1)
    energ_lo = E_RMF[:-1]
    energ_hi = E_RMF[1:]
    n_grp = np.ones(RMF.shape[0], dtype=int)
    f_chan = np.ones(RMF.shape[0], dtype=int)
    n_chan = np.full(matrix.shape[0], matrix.shape[1], dtype=int)

    cols_rmf = [
        fits.Column(name='ENERG_LO', array=energ_lo, format='1E', unit='keV'),
        fits.Column(name='ENERG_HI', array=energ_hi, format='1E', unit='keV'),
        fits.Column(name='N_GRP', array=n_grp, format='1I'),
        fits.Column(name='F_CHAN', array=f_chan, format='1I'),
        fits.Column(name='N_CHAN', array=n_chan, format='1I'),
        fits.Column(name='MATRIX', array=matrix, format=f'{matrix.shape[1]}E'),
    ]
    RMF_hdu = fits.BinTableHDU.from_columns(cols_rmf, header=hdr)

    hdr_Eb = fits.Header()
    hdr_Eb['EXTNAME'] = 'EBOUNDS'
    hdr_Eb['TELESCOP'] = 'SVOM'
    hdr_Eb['INSTRUME'] = 'MXT'
    hdr_Eb['ORIGIN'] = 'MIC'
    hdr_Eb['Date'] = (strDate, 'FITS file creation date')
    hdr_Eb['DATATYPE'] = ('Calib', 'Data channel media')
    hdr_Eb['FILTER'] = ''
    hdr_Eb['CHANTYPE'] = ('PI', 'detector channels are corrected')
    hdr_Eb['DETCHANS'] = (1024, 'total number of raw detector channels')
    hdr_Eb['HDUCLASS'] = 'OGIP'
    hdr_Eb['HDUCLAS1'] = ('RESPONSE', 'extension contains response data')
    hdr_Eb['HDUCLAS2'] = ('EBOUNDS', 'extension contains energy boundaries')
    hdr_Eb['HDUVERS'] = '1.3.0'
    hdr_Eb['TLMIN1'] = 1
    hdr_Eb['TLMAX1'] = 1024

    cols_ebounds = [
        fits.Column(name='CHANNEL', format='I', array=np.arange(1, 1025), unit=''),
        fits.Column(name='E_MIN', format='E', array=E_RMF[:-1], unit='keV'),
        fits.Column(name='E_MAX', format='E', array=E_RMF[1:], unit='keV'),
    ]
    Ebounds_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols_ebounds), header=hdr_Eb)

    hdr_prim = fits.Header()
    hdr_prim['COMMENT'] = "FITS (Flexible Image Transport System) format is defined in 'Astronomy"
    hdr_prim['COMMENT'] = "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"
    primaryHDU = fits.PrimaryHDU(header=hdr_prim)

    hdu = fits.HDUList([primaryHDU, RMF_hdu, Ebounds_hdu])
    hdu.writeto(filepath, overwrite=True, checksum=True)
    print(f'RMF saved to {filepath}')
    return filepath

def save_fits_CS(RMF, selected_data, save_folder=None):
    if save_folder is None:
        folder = ResultsPaths.RMF
    else:
        folder = Path(save_folder)
    csv_file = selected_data.get('csv_file', 'unknown')
    oldest_date = csv_validity_date(csv_file)
    data_date = oldest_date.strftime('%Y-%m-%d %H:%M:%S')
    oldest_date_str = oldest_date.strftime('%Y%m%dT%H%M%S')

    date_prod = datetime.utcnow()
    strDate = date_prod.strftime('%Y-%m-%dT%H:%M:%S')
    strDateName = date_prod.strftime('%Y%m%dT%H%M%S')

    filename = f'MXT-GND-MATRIX-CS_{oldest_date_str}_{strDateName}.fits'
    filepath = folder / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    E_RMF = np.linspace(0.1, 10.0, RMF.shape[0] + 1)
    energ_lo = E_RMF[:-1]
    energ_hi = E_RMF[1:]

    hdr_Eb = fits.Header()
    hdr_Eb['EXTNAME'] = 'EBOUNDS'
    hdr_Eb['TELESCOP'] = 'SVOM'
    hdr_Eb['INSTRUME'] = 'MXT'
    hdr_Eb['ORIGIN'] = 'MIC'
    hdr_Eb['Date'] = (strDate, 'FITS file creation date')
    hdr_Eb['DATATYPE'] = ('Calib', 'Data channel media')
    hdr_Eb['FILTER'] = ''
    hdr_Eb['CHANTYPE'] = ('PI', 'detector channels are corrected')
    hdr_Eb['DETCHANS'] = (1024, 'total number of raw detector channels')
    hdr_Eb['HDUCLASS'] = 'OGIP'
    hdr_Eb['HDUCLAS1'] = ('RESPONSE', 'extension contains response data')
    hdr_Eb['HDUCLAS2'] = ('EBOUNDS', 'extension contains energy boundaries')
    hdr_Eb['HDUVERS'] = '1.3.0'
    hdr_Eb['TLMIN1'] = 1
    hdr_Eb['TLMAX1'] = 1024

    cols_ebounds = [
        fits.Column(name='CHANNEL', format='I', array=np.arange(1, 1025), unit=''),
        fits.Column(name='E_MIN', format='E', array=energ_lo, unit='keV'),
        fits.Column(name='E_MAX', format='E', array=energ_hi, unit='keV'),
    ]
    Ebounds_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols_ebounds), header=hdr_Eb)

    hdr_prim = fits.Header()
    hdr_prim['DATE'] = strDate
    hdr_prim['CSV_FILE'] = csv_file
    hdr_prim['DATA_DATE'] = data_date
    hdr_prim['SHAPE'] = str(RMF.shape)
    hdr_prim['COMMENT'] = 'RMF for CS use.'
    primaryHDU = fits.PrimaryHDU(data=RMF, header=hdr_prim)

    hdu = fits.HDUList([primaryHDU, Ebounds_hdu])
    hdu.writeto(filepath, overwrite=True)
    print(f'CS RMF saved to {filepath}')

# ===========================================================================
# RMF Processing
# ===========================================================================

def rmf_process():
    """Process a selected spectral CSV file, fit the FWHM model, build an RMF, and save outputs."""
    print('\n' + '=' * 70)
    print('RMF Processing Tool')
    print('=' * 70)

    selected_data = select_data_for_rmf()

    # check if there is any row with the extrapolate value as 1 in the selected data, if there is remove from the analyses and print a warning
    extrapolate = selected_data.get('extrapolate')
    if extrapolate is not None and np.any(extrapolate == 1):
        print('Warning: Rows with extrapolate=1 found and will be removed from the analysis.')
        mask = extrapolate != 1
        for key in ['Nominal_Energy', 'FWHM_0', 'FWHM_err', 'Centroid_keV', 'Centroid_err_keV', 'extrapolate']:
            if key in selected_data and selected_data[key] is not None:
                selected_data[key] = selected_data[key][mask]

    # create a folder for saving plots with the name of the file
    csv_file = selected_data.get('csv_file', 'unknown')
    folder_name = Path(csv_file).stem
    save_folder = ResultsPaths.RMF / folder_name
    save_folder.mkdir(parents=True, exist_ok=True)

    energy = np.asarray(selected_data['Nominal_Energy'], dtype=float)
    fwhm = np.asarray(selected_data['FWHM_0'], dtype=float)
    params, cov, fit_data, residuals, chisq, dof = fit_noise_model(energy, fwhm)
    print(f"Fit → F = {params[0]:.4f}, ENC = {params[1]:.2f} e-, χ²/dof = {chisq:.1f}/{dof}")


    fwhm_al = maybe_add_al_kalpha_point(selected_data, params, cov, energy_value=1.486)

    plot_fwhm([selected_data], fit=True, save_folder=save_folder)

    print(f'FWHM at Al K-alpha (1.486 keV): {fwhm_al:.2f} eV')

    RMF = build_rmf(params, 2 ** 10 + 1)
    plot_rmf(RMF, [selected_data], save_folder=save_folder)

    save_choice = input('Do you want to save the RMF to FITS? (y/n): ')
    if save_choice.lower() == 'y':
        save_fits_calibration(RMF, selected_data, save_folder=save_folder)
    else:
        print('Skipping RMF FITS save.')

    save_choice = input('Do you want to save the CS RMF to FITS? (y/n): ')
    if save_choice.lower() == 'y':
        RMF_CS = build_rmf(params, 2 ** 11 + 1)
        save_fits_CS(RMF_CS, selected_data, save_folder=save_folder)
    else:
        print('Skipping CS RMF FITS save.')

    return selected_data, params, RMF

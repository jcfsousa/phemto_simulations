#!/usr/bin/env python3
"""Build detector redistribution matrices (RMF) from the LaueRing monochromatic
spectra produced by the parser.

Input   : analysis_overall/rmf/data_rmf_spectra/
          LaueRing{E}keV_config..._spec_hist-{EVENT_TYPE}_Det-{Si|CdTe}.txt
          Each file is CSV: Bin_center(keV), counts  on a common 0.1 keV grid.

For each detector (Si, CdTe, and Combined = Si + CdTe) we build
    M[i_trueE, j_measE] = counts
and normalize every TRUE-energy row to unity:
    R(E_meas | E_true) = M[i, :] / sum_j M[i, :]
That normalized matrix is the RMF (redistribution matrix): the probability
that a photon of true energy E_true is recorded in measured bin E_meas.

The row normalization discards the detection efficiency, which belongs in the
ARF.  We therefore ALSO save, per energy, the raw total counts and
efficiency = total_counts / N_SIM, so the ARF can be rebuilt later.

Outputs (-> rmf/rmf_output/):
    rmf_matrices.npz   all grids + raw/normalized matrices + efficiencies
    RMF_<det>.png      2D image of each normalized RMF
    RMF_<name>.fits    OGIP RMF (redistribution only) for XSPEC
    ARF_<name>.fits    OGIP ARF (effective area = efficiency x beam area) for XSPEC
    efficiency.png     detected fraction vs true energy per detector
    arf.png            effective area [cm^2] vs true energy per detector

NOTE on "Combined": this is the real shared/total-energy response read from the
Det-Both spectra (a single event's deposits summed across both detectors).  It is
NOT the sum of the Si and CdTe histograms: that naive sum double-counts photons
that deposit in both detectors.

In addition to the PNG plots, each response (Si/LED, CdTe/HED, Combined) is written
as an OGIP-compliant RMF FITS file (RMF_<name>.fits) that loads directly in
XSPEC/PyXspec.  The RMF stores only the redistribution (rows normalized to 1); the
detection efficiency and collecting area are carried by the matching ARF
(ARF_<name>.fits), so that for a photon model s(E) [ph/cm2/s/keV] XSPEC forms
    counts(E) = ARF(E) * RMF(.|E) * integral(s) * exposure.

ARF (effective area) convention
-------------------------------
PHEMTO is a FOCUSING instrument: two optics concentrate photons onto a small
focal spot on the detector -- a grazing-incidence X-ray MIRROR (~1.6-80 keV) and
a Laue LENS (Si(111) crystals, ~50-1000 keV).  The instrument effective area is
the combined optics area times the detector's detection efficiency:
    A_eff(E) = A_optics(E) * eff(E),   eff(E) = tot(E) / N_SIM        [cm^2]
    A_optics(E) = A_mirror(E) + A_lens(E)
The two optics are SUMMED where their bands overlap (~50-80 keV); each is zero
outside its own tabulated band, so A_optics is mirror-only below 50 keV and
lens-only above 80 keV.  A_mirror comes from MIRROR_AEFF_FILE, A_lens from
LENS_AEFF_FILE (Laue lens, from F. Frontera).  eff(E) is the fraction of
focal-spot photons the MC detects.  The LaueRing MC beam (a 0.075 cm / 1.5 mm
PSF disk) is only the focal spot the photons were fired onto; its area is a
fluence placeholder that cancels in eff = tot/N_SIM and is NOT a collecting area.
"""
import os
import re
import glob
from datetime import datetime, timezone

# If HEASoft/XSPEC is sourced in this shell it sets TCL_LIBRARY/TK_LIBRARY to
# empty (or its own incomplete tcltk), which breaks the system Tk that
# matplotlib's interactive backend uses (TclError: can't find tk.tcl). This
# script needs no HEASoft, so point those vars back at a working system tcl/tk.
def _fix_tcltk():
    for var, sub, probe in (('TCL_LIBRARY', 'tcl', 'init.tcl'),
                            ('TK_LIBRARY', 'tk', 'tk.tcl')):
        cur = os.environ.get(var)
        if cur and os.path.exists(os.path.join(cur, probe)):
            continue                      # already valid, leave it
        for base in ('/usr/share/tcltk', '/usr/lib', '/usr/share', '/usr/lib64'):
            for d in sorted(glob.glob(os.path.join(base, sub + '8.*')), reverse=True):
                if os.path.exists(os.path.join(d, probe)):
                    os.environ[var] = d
                    break
            if os.environ.get(var) and os.path.exists(os.path.join(os.environ[var], probe)):
                break
_fix_tcltk()

import numpy as np
from astropy.io import fits
import matplotlib
# Non-interactive backend: no GUI toolkit (no Tk), works headless, just writes
# the PNGs. To get pop-up windows instead, comment this line out and set
# SHOW=True below (the _fix_tcltk() above makes the Tk backend usable).
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# --- configuration ------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(HERE, 'data_rmf_spectra_cal')
OUT_DIR = os.path.join(HERE, 'rmf_output')
EVENT_TYPE = 'masked'          # which event class to use (masked = final good)
N_SIM = 1_000_000              # photons simulated per energy (for efficiency)
# Focusing-optics effective area A_optics(E) [cm^2] -- the real collecting area.
# Two optics focus onto the detector: a grazing-incidence X-ray MIRROR (~1.6-80
# keV) and a Laue LENS (~50-1000 keV). They are SUMMED where they overlap
# (50-80 keV), each contributing zero outside its own band. The ARF is
# A_optics(E) * eff(E); see the ARF convention in the docstring.
LENS_AEFF_FILE = os.path.normpath(os.path.join(
    HERE, '..', '..', 'optics', 'laue_lenses', 'Aeff_phemto_0.5E_Si.txt'))
MIRROR_AEFF_FILE = os.path.normpath(os.path.join(
    HERE, '..', '..', 'optics', 'mirror', 'area_mirror.csv'))
E_MEAS_MAX = 1000             # keV, x-axis upper limit for the plots
SHOW = True                   # True -> also plt.show() the figures

DETECTORS = ['Si', 'CdTe', 'Both']
# FITS/XSPEC instrument name used for each internal result key.
FITS_NAMES = {'Si': 'LED', 'CdTe': 'HED', 'Combined': 'Combined'}

_PAT = re.compile(r'LaueRing(\d+(?:\.\d+)?)keV.*spec_hist-' + EVENT_TYPE + r'_Det-(Si|CdTe|Both)\.txt$')


def discover_files():
    """Return {det: {E_keV: path}} for all matching spectra."""
    found = {d: {} for d in DETECTORS}
    for f in glob.glob(os.path.join(SPEC_DIR, f'*spec_hist-{EVENT_TYPE}_Det-*.txt')):
        m = _PAT.search(os.path.basename(f))
        if m:
            found[m.group(2)][float(m.group(1))] = f
    return found


def load_spectrum(path):
    """Return (bin_centers[keV], counts)."""
    e, c = np.loadtxt(path, delimiter=',', comments='#', unpack=True) # the spec saved from specMaker is bin_centers
    return e, c


def build_matrix(file_map, meas_centers):
    """file_map: {E_keV: path}. Returns (true_E sorted, raw counts matrix)."""
    energies = sorted(file_map)
    M = np.zeros((len(energies), meas_centers.size))
    for i, E in enumerate(energies):
        e, c = load_spectrum(file_map[E])
        if not np.allclose(e, meas_centers):
            raise ValueError(f'grid mismatch in {file_map[E]}')
        M[i] = c
    return np.array(energies, float), M


def normalize_rows(M):
    """Normalize each row to sum 1 (RMF). Returns (RMF, row_totals)."""
    tot = M.sum(axis=1)
    R = np.divide(M, tot[:, None], out=np.zeros_like(M), where=tot[:, None] > 0)
    return R, tot


def edges_from_centers(c):
    """Bin edges from (possibly non-uniform) centers."""
    c = np.asarray(c, float)
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate(([c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]))


def plot_rmf(name, true_E, meas_centers, R, out_png):
    x_edges = edges_from_centers(meas_centers)
    y_edges = edges_from_centers(true_E)
    fig, ax = plt.subplots(figsize=(10, 6))
    vmin = R[R > 0].min() if np.any(R > 0) else 1e-6
    im = ax.pcolormesh(x_edges, y_edges, R, norm=LogNorm(vmin=vmin, vmax=R.max()),
                       cmap='turbo', shading='flat')
    ax.plot([0, E_MEAS_MAX], [0, E_MEAS_MAX], 'w--', lw=0.8, alpha=0.7,
            label=r'$E_{\rm meas}=E_{\rm true}$')
    ax.set(xlim=(0, E_MEAS_MAX), xlabel='Measured energy [keV]',
           ylabel='True (incident) energy [keV]',
           title=f'RMF - {name} detector (rows normalized to 1)')
    ax.legend(loc='upper left')
    fig.colorbar(im, ax=ax, label=r'$R(E_{\rm meas}\,|\,E_{\rm true})$')
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f'  wrote {out_png}')


def save_rmf_fits(name, true_E, meas_centers, R, out_path):
    """Write the normalized RMF R as an OGIP-compliant FITS file for XSPEC.

    Layout follows the SVOM/MXT product in miguel/rmf_build.py (PrimaryHDU +
    MATRIX BinTable + EBOUNDS BinTable) but is generalized to a rectangular,
    MC-derived matrix:
        R[i, j] = R(E_meas_j | E_true_i)   -- rows (true E) already sum to 1,
    so, unlike Miguel's writer, we do NOT transpose: R is already in the OGIP
    orientation (one table row per incident-energy bin, vector over channels).
    """
    matrix = np.ascontiguousarray(R, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError(f'RMF {name} contains NaN or infinite values')
    np.maximum(matrix, 0.0, out=matrix)
    # Renormalize non-empty rows; leave all-zero rows (empty sim bins) at zero.
    row_sums = matrix.sum(axis=1, keepdims=True)
    np.divide(matrix, row_sums, out=matrix, where=row_sums > 0)

    n_energ, n_chan = matrix.shape

    # Incident (true) energy bins -> table rows.
    e_edges = edges_from_centers(true_E)
    energ_lo = e_edges[:-1]
    energ_hi = e_edges[1:]

    # Measured channel grid -> EBOUNDS.
    ch_edges = edges_from_centers(meas_centers)
    e_min = ch_edges[:-1]
    e_max = ch_edges[1:]
    channel = np.arange(1, n_chan + 1)

    # Full-row (uncompressed) storage: one group spanning all channels per row.
    n_grp = np.ones(n_energ, dtype=np.int16)
    f_chan = np.ones(n_energ, dtype=np.int16)      # 1-based first channel
    n_chan_col = np.full(n_energ, n_chan, dtype=np.int16)

    now = datetime.now(timezone.utc)
    str_date = now.strftime('%Y-%m-%dT%H:%M:%S')

    hdr = fits.Header()
    hdr['EXTNAME'] = 'MATRIX'
    hdr['LONGSTRN'] = ('OGIP 1.0', 'The HEASARC Long String Convention may be used.')
    hdr['TELESCOP'] = ('PHEMTO', 'Telescope or mission name')
    hdr['INSTRUME'] = (name, 'Detector / response name (LED, HED, Combined)')
    hdr['ORIGIN'] = ('CEA', 'Origin of FITS file')
    hdr['CREATOR'] = (os.path.basename(__file__), 'Program that created this FITS file')
    hdr['DATE'] = (str_date, 'FITS file creation date')
    hdr['CCLS0001'] = ('BCF', 'HEASARC class, dataset is Basic Calibration')
    hdr['CDTP0001'] = ('DATA', 'HEASARC code, dataset is real data')
    hdr['CCNM0001'] = ('MATRIX', 'HEASARC codename, dataset name')
    hdr['CBD10001'] = (f'CHAN(1-{n_chan})', 'HEASARC parameter limitation')
    hdr['CBD20001'] = (f'ENER({true_E.min():g}-{true_E.max():g})keV', 'HEASARC parameter limitation')
    hdr['CDES0001'] = (f'MC RMF {name} from LaueRing sims, event={EVENT_TYPE}', 'HEASARC brief desc')
    hdr['FILTER'] = ('', '')
    hdr['CHANTYPE'] = ('PI', 'detector channels are corrected')
    hdr['DETCHANS'] = (n_chan, 'total number of raw detector channels')
    hdr['HDUCLASS'] = ('OGIP', '')
    hdr['HDUCLAS1'] = ('RESPONSE', 'extension contains response data')
    hdr['HDUCLAS2'] = ('RSP_MATRIX', 'extension contains a response matrix')
    hdr['HDUCLAS3'] = ('REDIST', 'photon redistribution process only')
    hdr['HDUVERS'] = ('1.3.0', 'obsolete, for old software')
    hdr['RMFVERSN'] = ('1992a', 'obsolete, for old software')
    hdr['TLMIN4'] = (1, 'first channel, minimum of F_CHAN column')
    hdr['LO_THRES'] = (0, 'optional minimum proba threshold, below is 0')
    hdr['EFFAREA'] = (1, 'RMF only; detection efficiency belongs in the ARF')
    hdr['NUMGRP'] = (n_energ, 'optional, the sum of the N_GRP column')
    hdr['NUMELT'] = (int(matrix.size), 'optional, the sum of the N_CHAN column')

    cols_rmf = [
        fits.Column(name='ENERG_LO', array=energ_lo, format='1E', unit='keV'),
        fits.Column(name='ENERG_HI', array=energ_hi, format='1E', unit='keV'),
        fits.Column(name='N_GRP', array=n_grp, format='1I'),
        fits.Column(name='F_CHAN', array=f_chan, format='1I'),
        fits.Column(name='N_CHAN', array=n_chan_col, format='1I'),
        fits.Column(name='MATRIX', array=matrix, format=f'{n_chan}E'),
    ]
    rmf_hdu = fits.BinTableHDU.from_columns(cols_rmf, header=hdr)

    hdr_eb = fits.Header()
    hdr_eb['EXTNAME'] = 'EBOUNDS'
    hdr_eb['TELESCOP'] = 'PHEMTO'
    hdr_eb['INSTRUME'] = name
    hdr_eb['ORIGIN'] = 'CEA'
    hdr_eb['DATE'] = (str_date, 'FITS file creation date')
    hdr_eb['FILTER'] = ''
    hdr_eb['CHANTYPE'] = ('PI', 'detector channels are corrected')
    hdr_eb['DETCHANS'] = (n_chan, 'total number of raw detector channels')
    hdr_eb['HDUCLASS'] = 'OGIP'
    hdr_eb['HDUCLAS1'] = ('RESPONSE', 'extension contains response data')
    hdr_eb['HDUCLAS2'] = ('EBOUNDS', 'extension contains energy boundaries')
    hdr_eb['HDUVERS'] = '1.3.0'
    hdr_eb['TLMIN1'] = 1
    hdr_eb['TLMAX1'] = n_chan

    cols_ebounds = [
        fits.Column(name='CHANNEL', format='J', array=channel, unit=''),
        fits.Column(name='E_MIN', format='E', array=e_min, unit='keV'),
        fits.Column(name='E_MAX', format='E', array=e_max, unit='keV'),
    ]
    ebounds_hdu = fits.BinTableHDU.from_columns(fits.ColDefs(cols_ebounds), header=hdr_eb)

    hdr_prim = fits.Header()
    hdr_prim['COMMENT'] = "FITS (Flexible Image Transport System) format is defined in 'Astronomy"
    hdr_prim['COMMENT'] = "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"
    primary_hdu = fits.PrimaryHDU(header=hdr_prim)

    hdul = fits.HDUList([primary_hdu, rmf_hdu, ebounds_hdu])
    hdul.writeto(out_path, overwrite=True, checksum=True)
    print(f'  wrote {out_path}  (MATRIX {n_energ} x {n_chan})')


def load_lens_aeff(path=LENS_AEFF_FILE):
    """Laue-lens effective area table -> (E[keV], Aeff[cm^2]), E ascending."""
    E, A = np.loadtxt(path, comments='#', unpack=True)
    order = np.argsort(E)
    return E[order], A[order]


def load_mirror_aeff(path=MIRROR_AEFF_FILE):
    """X-ray mirror effective area CSV -> (E[keV], Aeff[cm^2]), E ascending."""
    E, A = np.loadtxt(path, delimiter=',', skiprows=1, unpack=True)  # header: Energy_keV,Area_cm2
    order = np.argsort(E)
    return E[order], A[order]


def optics_components(true_E):
    """Optics effective area on true_E: (combined, mirror, lens) in cm^2.

    Each optic is interpolated onto true_E and set to zero outside its own
    tabulated band; the combined area is their sum (they overlap ~50-80 keV).
    """
    El, Al = load_lens_aeff()
    Em, Am = load_mirror_aeff()
    lens = np.interp(true_E, El, Al, left=0.0, right=0.0)
    mirror = np.interp(true_E, Em, Am, left=0.0, right=0.0)
    return mirror + lens, mirror, lens


def optics_on_grid(true_E):
    """Combined focusing-optics effective area on true_E (zero outside bands)."""
    return optics_components(true_E)[0]


def save_arf_fits(name, true_E, tot, aeff_opt, out_path):
    """Write the OGIP ARF (effective area) that pairs with RMF_<name>.fits.

    SPECRESP(E) = A_optics(E) * eff(E),   eff(E) = tot(E) / N_SIM   [cm^2]
    i.e. the focusing Laue-lens effective area times the detector's detection
    efficiency (see the ARF convention in the module docstring).  `aeff_opt` is
    A_optics already interpolated onto `true_E` (via optics_on_grid).
    The ENERG grid is built from `true_E` with edges_from_centers() -- the SAME
    grid as the RMF MATRIX, which XSPEC requires for an ARF+RMF pair.
    """
    tot = np.asarray(tot, float)
    eff = tot / N_SIM                              # detector detection efficiency
    specresp = eff * np.asarray(aeff_opt, float)   # effective area [cm^2]
    if not np.isfinite(specresp).all():
        raise ValueError(f'ARF {name} contains NaN or infinite values')
    np.maximum(specresp, 0.0, out=specresp)

    e_edges = edges_from_centers(true_E)
    energ_lo = e_edges[:-1]
    energ_hi = e_edges[1:]

    now = datetime.now(timezone.utc)
    str_date = now.strftime('%Y-%m-%dT%H:%M:%S')

    hdr = fits.Header()
    hdr['EXTNAME'] = 'SPECRESP'
    hdr['TELESCOP'] = ('PHEMTO', 'Telescope or mission name')
    hdr['INSTRUME'] = (name, 'Detector / response name (LED, HED, Combined)')
    hdr['FILTER'] = ('', 'no filter')
    hdr['ORIGIN'] = ('CEA', 'Origin of FITS file')
    hdr['CREATOR'] = (os.path.basename(__file__), 'Program that created this FITS file')
    hdr['DATE'] = (str_date, 'FITS file creation date')
    hdr['HDUCLASS'] = ('OGIP', 'format conforms to OGIP standard')
    hdr['HDUCLAS1'] = ('RESPONSE', 'extension contains response data')
    hdr['HDUCLAS2'] = ('SPECRESP', 'extension contains an ARF (effective area)')
    hdr['HDUVERS'] = ('1.1.0', 'version of the file format')
    hdr['ARFVERSN'] = ('1992a', 'obsolete, for old software')
    hdr['CCLS0001'] = ('BCF', 'HEASARC class, dataset is Basic Calibration')
    hdr['CDTP0001'] = ('DATA', 'HEASARC code, dataset is real data')
    hdr['CCNM0001'] = ('SPECRESP', 'HEASARC codename, dataset name')
    hdr['CBD10001'] = (f'ENER({true_E.min():g}-{true_E.max():g})keV', 'HEASARC parameter limitation')
    hdr['CDES0001'] = (f'MC ARF {name}: optics Aeff x detector eff', 'HEASARC brief desc')
    hdr['COMMENT'] = ('A_eff = (A_mirror + A_lens)(E) * eff(E); optics = '
                      f'{os.path.basename(MIRROR_AEFF_FILE)} + {os.path.basename(LENS_AEFF_FILE)}')

    cols = [
        fits.Column(name='ENERG_LO', array=energ_lo, format='1E', unit='keV'),
        fits.Column(name='ENERG_HI', array=energ_hi, format='1E', unit='keV'),
        fits.Column(name='SPECRESP', array=specresp, format='1E', unit='cm**2'),
    ]
    arf_hdu = fits.BinTableHDU.from_columns(cols, header=hdr)

    hdr_prim = fits.Header()
    hdr_prim['COMMENT'] = "FITS (Flexible Image Transport System) format is defined in 'Astronomy"
    hdr_prim['COMMENT'] = "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"
    primary_hdu = fits.PrimaryHDU(header=hdr_prim)

    hdul = fits.HDUList([primary_hdu, arf_hdu])
    hdul.writeto(out_path, overwrite=True, checksum=True)
    print(f'  wrote {out_path}  (SPECRESP {specresp.size} bins, '
          f'peak {specresp.max():.4g} cm^2)')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = discover_files()

    meas_centers, _ = load_spectrum(next(iter(files['Si'].values())))

    results = {}      # name -> (true_E, raw_M, RMF, totals)
    tE_si, M_si = build_matrix(files['Si'], meas_centers)
    tE_cd, M_cd = build_matrix(files['CdTe'], meas_centers)
    tE_both, M_both = build_matrix(files['Both'], meas_centers)
    true_E = tE_si

    results['Si'] = (tE_si, M_si, *normalize_rows(M_si))
    results['CdTe'] = (tE_cd, M_cd, *normalize_rows(M_cd))
    results['Combined'] = (tE_both, M_both, *normalize_rows(M_both))

    # Combined optics (mirror + Laue lens) on the shared true-energy grid.
    aeff_opt, aeff_mirror, aeff_lens = optics_components(true_E)
    print(f'Optics (mirror + lens): peak {aeff_opt.max():.1f} cm^2; '
          f'nonzero in {true_E[aeff_opt > 0].min():.1f}-{true_E[aeff_opt > 0].max():.0f} keV '
          f'(mirror {true_E[aeff_mirror > 0].min():.1f}-{true_E[aeff_mirror > 0].max():.0f}, '
          f'lens {true_E[aeff_lens > 0].min():.0f}-{true_E[aeff_lens > 0].max():.0f})')

    # --- report + plots ---
    print(f'True energies [keV]: {true_E.tolist()}')
    print(f"{'det':>9} {'E[keV]':>7} {'total cnts':>12} {'efficiency':>11}")
    for name, (tE, M, R, tot) in results.items():
        for E, t in zip(tE, tot):
            print(f'{name:>9} {E:>7.0f} {t:>12.0f} {t / N_SIM:>11.4f}')
        plot_rmf(name, tE, meas_centers, R, os.path.join(OUT_DIR, f'RMF_{name}.png'))
        fits_name = FITS_NAMES[name]
        save_rmf_fits(fits_name, tE, meas_centers, R,
                      os.path.join(OUT_DIR, f'RMF_{fits_name}.fits'))
        save_arf_fits(fits_name, tE, tot, aeff_opt,
                      os.path.join(OUT_DIR, f'ARF_{fits_name}.fits'))

    # --- efficiency curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in results:
        tE, M, R, tot = results[name]
        ax.plot(tE, tot / N_SIM, '-', label=name)
    ax.set(xlabel='True energy [keV]', ylabel=f'Detected fraction (/{N_SIM:g})',
           title='Detection efficiency (seed for the ARF)')
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_xscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'efficiency.png'), dpi=150)
    print(f"  wrote {os.path.join(OUT_DIR, 'efficiency.png')}")

    # --- effective area (ARF) curve: A_optics(E) * detector efficiency ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(true_E, aeff_mirror, ':', color='gray', lw=1.0, label='mirror (optics)')
    ax.plot(true_E, aeff_lens, '--', color='gray', lw=1.0, label='Laue lens (optics)')
    ax.plot(true_E, aeff_opt, 'k-', lw=1.2, alpha=0.7, label='optics total (mirror+lens)')
    for name in results:
        tE, M, R, tot = results[name]
        ax.plot(tE, (tot / N_SIM) * aeff_opt, '-', label=f'{name} (optics x eff)')
    ax.set(xlabel='True energy [keV]', ylabel='Effective area [cm$^2$]',
           title='ARF: combined optics effective area x detector efficiency')
    ax.grid(alpha=0.3, which='both')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'arf.png'), dpi=150)
    print(f"  wrote {os.path.join(OUT_DIR, 'arf.png')}")

    # --- save everything ---
    #np.savez(os.path.join(OUT_DIR, 'rmf_matrices.npz'),
    #         meas_energy=meas_centers, true_energy=true_E, n_sim=N_SIM,
    #         Si_raw=M_si, Si_rmf=results['Si'][2], Si_total=results['Si'][3],
    #         CdTe_raw=M_cd, CdTe_rmf=results['CdTe'][2], CdTe_total=results['CdTe'][3],
    #         Combined_raw=M_both, Combined_rmf=results['Combined'][2],
    #         Combined_total=results['Combined'][3])
    #print(f"  wrote {os.path.join(OUT_DIR, 'rmf_matrices.npz')}")

    if SHOW:
        plt.show()


if __name__ == '__main__':
    main()

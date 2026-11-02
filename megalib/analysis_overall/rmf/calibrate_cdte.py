#!/usr/bin/env python3
"""CdTe energy recalibration from Crystal Ball photopeak fits.

The LaueRing monochromatic CdTe spectra in ``data_rmf_spectra/`` are
UNCALIBRATED: charge-collection losses push every photopeak slightly below the
true incident energy, and the offset grows with energy (e.g. the 900 keV
photopeak sits near 895 keV).

This script
  1. loads the CdTe photopeak spectra for a chosen set of calibration energies,
  2. fits a Crystal Ball (Gaussian core + low-energy power-law tail, on a
     linear background) to each photopeak and reports the reduced chi-square,
  3. builds a LINEAR calibration   E_true = a * E_meas + b   from the fitted
     peak centres,
  4. plt.show()s the individual fits, the calibration line + residuals, and a
     before/after comparison of example spectra on the raw vs calibrated axis,
  5. asks whether to apply that linear calibration to EVERY spectrum file in
     data_rmf_spectra/.  Counts are re-binned onto the ORIGINAL 0.1 keV grid
     (total counts conserved) and calibrated copies written to
     ``data_rmf_spectra_cal/``.

Run it with the project venv:
    env-phemto/bin/python3 rmf/calibrate_cdte.py
"""
import os
import sys
import re
import glob

# If HEASoft/XSPEC is sourced in this shell it sets TCL_LIBRARY/TK_LIBRARY to
# empty (or its own incomplete tcltk), breaking the system Tk that matplotlib's
# interactive backend needs (TclError: can't find tk.tcl). This script shows
# plots interactively, so repoint those vars at a working system tcl/tk.
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
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rebin_spectra as rb   # scheme chooser + count-conserving rebin

# --- configuration ------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_DIR = os.path.join(HERE, 'data_rmf_spectra')
OUT_DIR = os.path.join(HERE, 'data_rmf_spectra_cal')
EVENT_TYPE = 'masked'
CONFIG_TAG = 'config3x3_1cm'

# Energies (keV) used to anchor the calibration.
CALIB_ENERGIES = [10, 20, 30, 50, 100, 150, 200, 250,
                  300, 350, 400, 500, 600, 700, 800, 900]
# Energies whose raw vs calibrated spectra are overlaid in the before/after plot.
SHOWCASE_ENERGIES = [30, 100, 300, 900]


# --- I/O ----------------------------------------------------------------
def cdte_path(E):
    return os.path.join(SPEC_DIR,
                        f'LaueRing{float(E)}keV_{CONFIG_TAG}_spec_hist-{EVENT_TYPE}_Det-CdTe.txt')


def load_spectrum(path):
    """Return (bin_centers[keV], counts)."""
    e, c = np.loadtxt(path, delimiter=',', comments='#', unpack=True)
    return e, c


def edges_from_centers(c):
    """Bin edges from (uniform) centers."""
    c = np.asarray(c, float)
    mid = 0.5 * (c[:-1] + c[1:])
    return np.concatenate(([c[0] - (mid[0] - c[0])], mid,
                           [c[-1] + (c[-1] - mid[-1])]))


# --- Crystal Ball -------------------------------------------------------
def crystal_ball(x, mu, sigma, alpha, n, amp, b0, b1):
    """Crystal Ball (Gaussian core, LOW-energy power-law tail) + linear bkg.

    Gaussian for (x-mu)/sigma > -alpha, power-law tail below.  alpha>0 is the
    tail transition in units of sigma, n>1 the tail power.  b0 + b1*(x-mu) is a
    local linear background (Compton continuum under the photopeak).
    """
    t = (x - mu) / sigma
    a = abs(alpha)
    A = (n / a) ** n * np.exp(-0.5 * a * a)
    B = n / a - a
    core = np.exp(-0.5 * t * t)
    with np.errstate(over='ignore', invalid='ignore'):
        # tail is only used where t <= -a; the np.where discards the rest
        tail = A * np.power(np.clip(B - t, 1e-12, None), -n)
    peak = np.where(t > -a, core, tail)
    return amp * peak + b0 + b1 * (x - mu)


def grid_spectrum(e, c, edges):
    """Put a native spectrum onto the chosen grid.

    edges is None -> keep the native uniform grid (width = NATIVE_WIDTH).
    Otherwise -> count-conserving rebin.  Returns (centers, counts, widths).
    """
    if edges is None:
        return e, c, np.full_like(e, rb.NATIVE_WIDTH)
    return rb.rebin(e, c, edges)


def fit_photopeak(e, c, w, E):
    """Fit the Crystal Ball to the photopeak of a spectrum of true energy E.

    Fits the DENSITY (counts / bin width) so that variable-width bins are not
    distorted by their width; on a uniform grid this is just a constant rescale
    and gives the same mu / chi2 as fitting raw counts.

    Returns dict with mu, mu_err, chi2_red, the fit window and best-fit params.
    """
    dens = c / w                          # counts per keV
    # locate the photopeak: highest density bin above 0.6*E
    hi = e > 0.6 * E
    idx = np.where(hi)[0]
    ip = idx[np.argmax(dens[idx])]
    mu0 = e[ip]
    peak_d = dens[ip]

    sig0 = max(0.3, 0.004 * E)            # ~0.4% energy resolution
    # window: wide low-energy side for the tail, tighter above
    lo = mu0 - max(8.0, 0.045 * E)
    up = mu0 + max(3.0, 0.020 * E)
    win = (e >= lo) & (e <= up)
    xw, yw = e[win], dens[win]
    err = np.sqrt(np.maximum(c[win], 1.0)) / w[win]   # Poisson on counts -> density

    b0_0 = float(np.median(yw[:max(3, len(yw) // 10)]))  # bkg from low-E edge
    p0 = [mu0, sig0, 1.2, 3.0, peak_d - b0_0, b0_0, 0.0]
    bounds = ([lo, 0.05, 0.2, 1.01, 0.0, 0.0, -np.inf],
              [up, 0.1 * E + 1, 12.0, 60.0, 5 * peak_d + 10, 5 * peak_d + 10, np.inf])

    popt, pcov = curve_fit(crystal_ball, xw, yw, p0=p0, sigma=err,
                           absolute_sigma=True, bounds=bounds, maxfev=100000)
    perr = np.sqrt(np.diag(pcov))
    resid = (yw - crystal_ball(xw, *popt)) / err
    dof = max(1, len(xw) - len(popt))
    chi2_red = float(np.sum(resid ** 2) / dof)

    return dict(E=E, mu=popt[0], mu_err=perr[0], sigma=popt[1],
                chi2_red=chi2_red, popt=popt, window=(lo, up),
                xw=xw, yw=yw)


# --- calibration --------------------------------------------------------
def linear_calibration(fits):
    """Weighted linear fit  E_true = a * mu_meas + b  over the peak fits."""
    mu = np.array([f['mu'] for f in fits])
    mu_err = np.array([f['mu_err'] for f in fits])
    Etrue = np.array([f['E'] for f in fits], float)
    w = 1.0 / np.clip(mu_err, 1e-3, None) ** 2
    (a, b), cov = np.polyfit(mu, Etrue, 1, w=np.sqrt(w), cov=True)
    pred = a * mu + b
    resid = Etrue - pred
    ss_res = np.sum((Etrue - pred) ** 2)
    ss_tot = np.sum((Etrue - Etrue.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return dict(a=float(a), b=float(b), a_err=float(np.sqrt(cov[0, 0])),
                b_err=float(np.sqrt(cov[1, 1])), r2=float(r2),
                mu=mu, Etrue=Etrue, resid=resid)


def apply_calibration(e, c, a, b, edges):
    """Re-bin counts onto the ORIGINAL grid after E_cal = a*e + b.

    Each original bin is an INTERVAL of width ~0.1 keV; under the calibration it
    maps to [a*e_lo+b, a*e_hi+b].  Its counts are distributed over the
    destination bins in proportion to the overlap length, i.e. the counts are
    treated as uniformly spread across the (calibrated) bin width.

    Doing this with np.histogram on the bin *centres* instead (dropping each
    bin's whole count at a single calibrated point) aliases against the
    destination grid: because a~1, the calibrated centres are spaced a*0.1 keV
    while the target bins are 0.1 keV, so periodically a target bin catches zero
    points -> a comb of empty / under-populated bins.  The overlap/cumulative
    method below avoids that while conserving total counts exactly.
    """
    src_edges = a * edges_from_centers(e) + b        # calibrated source edges
    # cumulative counts vs energy, piecewise-linear across each source bin
    cum = np.concatenate(([0.0], np.cumsum(c)))      # len == len(src_edges)
    cum_at = np.interp(edges, src_edges, cum)        # clamps outside -> 0 beyond
    return np.diff(cum_at)


# --- plots --------------------------------------------------------------
def plot_fits(fits):
    n = len(fits)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, fits):
        ax.step(f['xw'], f['yw'], where='mid', color='0.4', lw=0.8, label='data')
        xx = np.linspace(f['xw'][0], f['xw'][-1], 600)
        ax.plot(xx, crystal_ball(xx, *f['popt']), 'r-', lw=1.4, label='Crystal Ball')
        ax.axvline(f['mu'], color='b', ls=':', lw=1.0)
        ax.axvline(f['E'], color='g', ls='--', lw=0.9)
        ax.set_title(f"E={f['E']:g} keV   "
                     r"$\mu$=" f"{f['mu']:.2f}   "
                     r"$\chi^2_\nu$=" f"{f['chi2_red']:.2f}",
                     fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.axis('off')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle('CdTe photopeak Crystal Ball fits  '
                 '(blue = fitted peak, green = true energy)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))


def plot_calibration(cal):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                   gridspec_kw=dict(height_ratios=[3, 1]))
    mu, Et = cal['mu'], cal['Etrue']
    ax1.plot(mu, Et, 'ko', label='fitted peaks')
    xs = np.linspace(0, mu.max() * 1.05, 200)
    ax1.plot(xs, cal['a'] * xs + cal['b'], 'r-',
             label=f"E_true = {cal['a']:.5f}·E_meas + {cal['b']:.4f}")
    ax1.plot(xs, xs, 'k--', lw=0.8, alpha=0.6, label='identity (E_meas=E_true)')
    ax1.set(ylabel='True energy [keV]',
            title=f"Linear calibration   R²={cal['r2']:.6f}")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    ax2.axhline(0, color='r', lw=1)
    ax2.plot(mu, cal['resid'], 'ko-', ms=4)
    ax2.set(xlabel='Measured (raw) peak energy [keV]',
            ylabel='True − fit [keV]',
            title='Calibration residuals')
    ax2.grid(alpha=0.3)
    fig.tight_layout()


def plot_before_after(cal, edges):
    fig, axes = plt.subplots(1, len(SHOWCASE_ENERGIES),
                             figsize=(5 * len(SHOWCASE_ENERGIES), 4.2))
    axes = np.atleast_1d(axes)
    a, b = cal['a'], cal['b']
    for ax, E in zip(axes, SHOWCASE_ENERGIES):
        e, c = load_spectrum(cdte_path(E))
        c_cal = apply_calibration(e, c, a, b, edges)
        lo, hi = E - max(8, 0.05 * E), E + max(5, 0.03 * E)
        m = (e >= lo) & (e <= hi)
        ax.step(e[m], c[m], where='mid', color='0.6', lw=1.0, label='raw')
        ax.step(e[m], c_cal[m], where='mid', color='C3', lw=1.3, label='calibrated')
        ax.axvline(E, color='g', ls='--', lw=1.0, label='true energy')
        ax.set(title=f'E = {E:g} keV', xlabel='Energy [keV]', ylabel='counts')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('Before / after linear calibration (CdTe photopeaks)', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))


# --- apply to all files -------------------------------------------------
def apply_to_all(cal):
    a, b = cal['a'], cal['b']
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SPEC_DIR, '*.txt')))
    n_ok = 0
    for path in files:
        with open(path) as fh:
            header = [ln for ln in fh if ln.startswith('#')]
        e, c = load_spectrum(path)
        edges = edges_from_centers(e)
        c_cal = apply_calibration(e, c, a, b, edges)
        out = os.path.join(OUT_DIR, os.path.basename(path))
        with open(out, 'w') as fh:
            fh.writelines(header)
            fh.write(f'# linear calibration applied (from CdTe photopeaks): '
                     f'E_cal = {a:.6f}*E_raw + {b:.6f} keV\n')
            for ei, ci in zip(e, c_cal):
                fh.write(f'{ei:.18e},{ci:.18e}\n')
        n_ok += 1
    print(f'  wrote {n_ok} calibrated spectra to {OUT_DIR}')


# --- main ---------------------------------------------------------------
def main():
    # ask which grid the fits should run on
    label, edges = rb.choose_scheme()
    if edges is not None:
        ww = np.diff(edges)
        print(f'\nFitting on {label} grid: {len(edges)-1} bins, '
              f'width {ww.min():.3f} -> {ww.max():.3f} keV')
    else:
        print(f'\nFitting on {label} grid (native 0.1 keV)')

    print('\nFitting Crystal Ball to CdTe photopeaks ...')
    fits = []
    for E in CALIB_ENERGIES:
        path = cdte_path(E)
        if not os.path.exists(path):
            print(f'  [skip] {E} keV : file not found')
            continue
        e, c = load_spectrum(path)
        ce, cc, cw = grid_spectrum(e, c, edges)
        try:
            f = fit_photopeak(ce, cc, cw, E)
            fits.append(f)
            flag = '  (poor fit)' if f['chi2_red'] > 5 else ''
            print(f"  E={E:>4g} keV  mu={f['mu']:8.3f} +/- {f['mu_err']:.3f}  "
                  f"sigma={f['sigma']:.3f}  chi2_red={f['chi2_red']:7.2f}{flag}")
        except Exception as exc:
            print(f'  [fail] {E} keV : {exc}')

    if len(fits) < 2:
        print('Not enough successful fits for a calibration.')
        return

    cal = linear_calibration(fits)
    print('\nLinear calibration  E_true = a*E_meas + b')
    print(f"  a = {cal['a']:.6f} +/- {cal['a_err']:.6f}")
    print(f"  b = {cal['b']:.6f} +/- {cal['b_err']:.6f} keV")
    print(f"  R^2 = {cal['r2']:.6f}   "
          f"max |residual| = {np.max(np.abs(cal['resid'])):.3f} keV")

    # reference grid (edges) for the before/after and for application
    e_ref, _ = load_spectrum(cdte_path(CALIB_ENERGIES[0]))
    edges = edges_from_centers(e_ref)

    plot_fits(fits)
    plot_calibration(cal)
    plot_before_after(cal, edges)
    print('\nShowing fits + calibration + before/after. Close the windows to continue.')
    plt.show()

    ans = input(f'\nApply this linear calibration to ALL *.txt spectra in '
                f'{os.path.relpath(SPEC_DIR, HERE)}/ ?\n'
                f'  (calibrated copies -> {os.path.relpath(OUT_DIR, HERE)}/)  [y/N] ').strip().lower()
    if ans in ('y', 'yes'):
        apply_to_all(cal)
        print('Done.')
    else:
        print('Not applied. Calibration coefficients above can be reused later.')


if __name__ == '__main__':
    main()

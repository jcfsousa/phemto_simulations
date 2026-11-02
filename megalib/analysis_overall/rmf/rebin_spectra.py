#!/usr/bin/env python3
"""Variable-width rebinning for the LaueRing spectra.

The raw spectra are histogrammed on a FIXED 0.1 keV grid.  That is finer than
the detector resolution at low energy (good) but wasteful and noisy at high
energy, where the photopeak is ~2 keV wide and the statistics are thin.  This
module rebins a spectrum onto a NON-UNIFORM grid: narrow bins at low energy,
wide bins at high energy.

Two edge-building schemes are provided:

  * resolution-matched  (recommended) -- bin width = FWHM(E) / BINS_PER_FWHM,
    where FWHM comes from the detector's own EnergyResolution model.  This puts
    a constant number of bins across every peak, so no energy is over- or
    under-sampled relative to what the detector can actually resolve.

  * logarithmic -- constant relative width dE/E (geometric edges).  Simpler,
    detector-agnostic, also "fine low / coarse high".

Rebinning is COUNT-CONSERVING: each native 0.1 keV bin's counts are summed into
whatever new bin contains its centre.  You can never recover detail finer than
the native 0.1 keV grid, so widths are clamped to >= the native width (you only
ever MERGE native bins, never split them).

Run the self-demo (no files written):
    env-phemto/bin/python3 rmf/rebin_spectra.py
"""
import os
import re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DET_FILE = os.path.join(HERE, '..', '..', 'detectors', 'CdTe3x3detectorBig.det')
SPEC_DIR = os.path.join(HERE, 'data_rmf_spectra')
OUT_DIR = os.path.join(HERE, 'data_rmf_spectra_rebinned')

# --- tunables -----------------------------------------------------------
E_MIN = 1.0          # keV, low edge of the rebinned grid
E_MAX = 1000.0       # keV, high edge
BINS_PER_FWHM = 4.0  # resolution-matched: bins across one FWHM (higher = finer)
N_LOG_BINS = 400     # logarithmic scheme: total number of bins
NATIVE_WIDTH = 0.1   # keV, the simulation grid step (floor on any new bin)


# --- detector resolution model -----------------------------------------
def load_resolution(det_file=DET_FILE):
    """Parse 'EnergyResolution Gaus <E> <mean> <sigma>' -> (E[keV], sigma[keV])."""
    E, S = [], []
    with open(det_file) as fh:
        for ln in fh:
            m = re.search(r'EnergyResolution\s+Gaus\s+([\d.]+)\s+[\d.]+\s+([\d.]+)', ln)
            if m:
                E.append(float(m.group(1)))
                S.append(float(m.group(2)))
    return np.array(E), np.array(S)


def sigma_at(E, res_E, res_sigma):
    """Interpolate (and linearly extrapolate) the resolution sigma at energy E."""
    return np.interp(E, res_E, res_sigma,
                     left=res_sigma[0], right=res_sigma[-1])


# --- edge builders ------------------------------------------------------
def resolution_edges(emin=E_MIN, emax=E_MAX, bins_per_fwhm=BINS_PER_FWHM,
                     det_file=DET_FILE, native=NATIVE_WIDTH):
    """Variable edges with width = FWHM(E)/bins_per_fwhm, floored at `native`."""
    res_E, res_sigma = load_resolution(det_file)
    edges = [emin]
    e = emin
    while e < emax:
        fwhm = 2.3548 * sigma_at(e, res_E, res_sigma)
        width = max(native, fwhm / bins_per_fwhm)
        e = e + width
        edges.append(e)
    edges[-1] = emax
    return np.array(edges)


def log_edges(emin=E_MIN, emax=E_MAX, nbins=N_LOG_BINS, native=NATIVE_WIDTH):
    """Geometric (constant dE/E) edges, floored so no bin is finer than native."""
    edges = np.geomspace(emin, emax, nbins + 1)
    # enforce the native-width floor by walking and merging too-narrow bins
    out = [edges[0]]
    for e in edges[1:]:
        if e - out[-1] >= native:
            out.append(e)
    if out[-1] != emax:
        out.append(emax)
    return np.array(out)


# --- the rebin ----------------------------------------------------------
def rebin(centers, counts, new_edges):
    """Count-conserving rebin of a histogram onto new_edges.

    Returns (new_centers, new_counts, new_widths).  new_centers are the bin
    midpoints; divide new_counts by new_widths to plot a density (counts/keV),
    which is what you compare across unequal bins.
    """
    new_counts, _ = np.histogram(centers, bins=new_edges, weights=counts)
    new_centers = 0.5 * (new_edges[:-1] + new_edges[1:])
    new_widths = np.diff(new_edges)
    return new_centers, new_counts, new_widths


# --- scheme chooser -----------------------------------------------------
def choose_scheme():
    """Ask which rebinning scheme to use and build its edges.

    Returns (label, edges) or (label, None) for the native 'no rebin' case.
    """
    print('\nHow do you want to rebin? (fine bins at low E, coarse at high E)')
    print('  [1] resolution-matched  '
          '(bin width = FWHM(E)/N, from the detector resolution)   [recommended]')
    print('  [2] logarithmic         (constant relative width dE/E)')
    print('  [3] native              (keep the 0.1 keV grid, no rebin)')
    choice = (input('Choose [1/2/3] (default 1): ').strip() or '1')

    if choice == '3':
        return 'native 0.1 keV', None

    if choice == '2':
        raw = input(f'  number of log bins over {E_MIN:g}-{E_MAX:g} keV '
                    f'(default {N_LOG_BINS}): ').strip()
        nbins = int(raw) if raw else N_LOG_BINS
        edges = log_edges(nbins=nbins)
        return f'logarithmic ({nbins} bins)', edges

    # default: resolution-matched
    raw = input(f'  bins per FWHM (default {BINS_PER_FWHM:g}; '
                f'higher = finer): ').strip()
    bpf = float(raw) if raw else BINS_PER_FWHM
    edges = resolution_edges(bins_per_fwhm=bpf)
    return f'resolution-matched ({bpf:g} bins/FWHM)', edges


# --- demo plot ----------------------------------------------------------
def demo_plot(label, edges, demo_E=(30, 300, 900)):
    import matplotlib.pyplot as plt

    def load(path):
        return np.loadtxt(path, delimiter=',', comments='#', unpack=True)

    def cdte(E):
        return os.path.join(SPEC_DIR,
                            f'LaueRing{float(E)}keV_config3x3_1cm_spec_hist-masked_Det-CdTe.txt')

    fig, axes = plt.subplots(2, len(demo_E), figsize=(5 * len(demo_E), 7))
    for j, E in enumerate(demo_E):
        e, c = load(cdte(E))
        nc, cc, w = rebin(e, c, edges)
        lo, hi = E - max(10, 0.08 * E), E + max(6, 0.04 * E)
        m = (e >= lo) & (e <= hi)
        ax = axes[0, j]
        ax.step(e[m], c[m] / NATIVE_WIDTH, where='mid', color='0.6', lw=0.9,
                label='native 0.1 keV')
        mr = (nc >= lo) & (nc <= hi)
        ax.step(nc[mr], cc[mr] / w[mr], where='mid', color='C3', lw=1.4,
                label='rebinned')
        ax.axvline(E, color='g', ls='--', lw=0.9)
        ax.set(title=f'E = {E:g} keV  (density)', xlabel='Energy [keV]',
               ylabel='counts / keV')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax2 = axes[1, j]
        ax2.step(nc[mr], w[mr], where='mid', color='C0', lw=1.3, label='bin width')
        ax2.axhline(NATIVE_WIDTH, color='0.6', ls=':', label='native 0.1 keV')
        ax2.set(xlabel='Energy [keV]', ylabel='bin width [keV]',
                title=f'bin width near {E:g} keV')
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    fig.suptitle(f'Rebinning: {label}', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()


# --- apply to all files -------------------------------------------------
def apply_to_all(label, edges):
    import glob
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SPEC_DIR, '*.txt')))
    for path in files:
        with open(path) as fh:
            header = [ln for ln in fh if ln.startswith('#')]
        e, c = np.loadtxt(path, delimiter=',', comments='#', unpack=True)
        nc, cc, w = rebin(e, c, edges)
        out = os.path.join(OUT_DIR, os.path.basename(path))
        with open(out, 'w') as fh:
            for ln in header:
                if 'Bin size' in ln or 'Bin_center' in ln:
                    continue
                fh.write(ln)
            fh.write(f'# rebinned: {label} (variable width)\n')
            fh.write('# Bin_center(keV),cnts(total),bin_width(keV)\n')
            for ci_e, ci_c, ci_w in zip(nc, cc, w):
                fh.write(f'{ci_e:.18e},{ci_c:.18e},{ci_w:.18e}\n')
    print(f'  wrote {len(files)} rebinned spectra to '
          f'{os.path.relpath(OUT_DIR, HERE)}/')


# --- main ---------------------------------------------------------------
def main():
    label, edges = choose_scheme()
    if edges is None:
        print('Native grid kept - nothing to do.')
        return
    w = np.diff(edges)
    print(f'\n{label}: {len(edges)-1} bins, '
          f'width {w.min():.3f} -> {w.max():.3f} keV')

    demo_plot(label, edges)

    ans = input(f'\nApply this rebinning to ALL *.txt spectra in '
                f'{os.path.relpath(SPEC_DIR, HERE)}/ ?\n'
                f'  (rebinned copies -> {os.path.relpath(OUT_DIR, HERE)}/)  [y/N] '
                ).strip().lower()
    if ans in ('y', 'yes'):
        apply_to_all(label, edges)
        print('Done.')
    else:
        print('Not applied.')


if __name__ == '__main__':
    main()

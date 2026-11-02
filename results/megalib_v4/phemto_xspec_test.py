#!/usr/bin/env python3
"""Fold a source model through the detector RMF and look at the response.

Two source types:
  * delta lines (default) -- one or more monochromatic lines
  * crab                  -- the Crab power law F(E)=K(E/1keV)^-alpha
                             (arXiv:astro-ph/0406058: K=14.44, alpha=2.169)

The ARF (effective area) is applied by default: the ARF_*.fits sitting next to
the RMF is picked up automatically, so the folded spectrum is in physical
counts/s/keV. Use --arf to point elsewhere, or --no-arf for the RMF-only
(redistribution shape, arbitrary normalization).

Run:
    source activate_phemto_xspec.sh
    python simple_response.py                       # default lines (+auto ARF)
    python simple_response.py 50 100 200            # custom line energies [keV]
    python simple_response.py --source crab         # Crab power law
    python simple_response.py --source crab --crab-index 2.1 --crab-norm 10
    python simple_response.py 50 100 --rmf rmf_output/RMF_LED.fits  # ARF_LED auto
    python simple_response.py 70 --rmf rmf_output/RMF_HED.fits --arf rmf_output/ARF_HED.fits
    python simple_response.py 100 --no-arf          # RMF only, shape
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import xspec

HERE = os.path.dirname(os.path.abspath(__file__))

ap = argparse.ArgumentParser()
ap.add_argument('lines', type=float, nargs='*', default=[50.0, 100.0, 200.0, 300.0, 400.0],
                help='line energies [keV] (used when --source lines)')
ap.add_argument('--source', choices=['lines', 'crab'], default='lines',
                help='what to fold through the RMF')
ap.add_argument('--rmf', default=os.path.join(HERE, 'rmf_output', 'RMF_Combined.fits'))
ap.add_argument('--arf', default=None,
                help='ARF (effective area) FITS. Default: auto (the ARF_*.fits '
                     'next to the RMF). Pass a path to override.')
ap.add_argument('--no-arf', action='store_true',
                help='fold through the RMF only (redistribution, no effective area)')
ap.add_argument('--norm', type=float, default=1.0, help='norm (ph/cm2/s) per line')
# Crab power law F(E) = K (E/1keV)^(-alpha) ph/cm2/s/keV
# (arXiv:astro-ph/0406058v1): K = 14.44, alpha = 2.169
ap.add_argument('--crab-index', type=float, default=2.169, help='Crab photon index (alpha)')
ap.add_argument('--crab-norm', type=float, default=14.44,
                help='Crab powerlaw norm K (ph/cm2/s/keV at 1 keV)')
ap.add_argument('--rebin', type=int, default=1,
                help='combine N adjacent 0.1 keV channels for display (default 1)')
ap.add_argument('--out', default=None, help='output PNG')
args = ap.parse_args()

RMF = os.path.abspath(args.rmf)

# Resolve the ARF: explicit --arf, else auto (the ARF_*.fits beside the RMF),
# unless --no-arf. With an ARF the folded spectrum is in physical counts/s/keV
# (effective area applied); without one XSPEC assumes a flat 1 cm^2 and the
# normalization is arbitrary (shape only).
if args.no_arf:
    ARF = None
elif args.arf is not None:
    ARF = os.path.abspath(args.arf)
else:
    base = os.path.basename(RMF)
    cand = os.path.join(os.path.dirname(RMF), base.replace('RMF_', 'ARF_', 1))
    ARF = cand if (base.startswith('RMF_') and os.path.exists(cand)) else None
if ARF is not None and not os.path.exists(ARF):
    raise SystemExit(f'ARF not found: {ARF}')
print(f'RMF: {RMF}')
print(f'ARF: {ARF if ARF else "(none -- RMF only, flat 1 cm^2, shape only)"}')

# --- 1. build the source model --------------------------------------------
xspec.AllData.clear()
xspec.AllModels.clear()

if args.source == 'crab':
    model = xspec.Model('powerlaw')
    model.powerlaw.PhoIndex = args.crab_index
    model.powerlaw.norm = args.crab_norm
    title = f'Crab power law (G={args.crab_index:g}, norm={args.crab_norm:g})'
    out = args.out or os.path.join(HERE, 'rmf_output', 'response_crab.png')
else:
    LINES = sorted(args.lines)
    model = xspec.Model(' + '.join(['gaussian'] * len(LINES)))
    for i, E in enumerate(LINES):
        base = 3 * i
        model(base + 1).values = E          # LineE  [keV]
        model(base + 2).values = 0.0        # Sigma = 0 -> monochromatic (delta)
        model(base + 3).values = args.norm  # norm
    title = 'Detector response to delta lines: ' + ', '.join(f'{E:g}' for E in LINES) + ' keV'
    out = args.out or os.path.join(HERE, 'rmf_output', 'response_lines.png')

# --- 2. fold through the RMF (no noise -> ideal response) -------------------
fak = os.path.join(HERE, 'rmf_output', 'resp_source.fak')
if os.path.exists(fak):
    os.remove(fak)
fs = xspec.FakeitSettings(response=RMF, arf=(ARF or ''), exposure=1.0, fileName=fak)
xspec.AllData.fakeit(1, [fs], applyStats=False)   # applyStats=False = no Poisson noise

# --- 3. pull out the folded spectrum ---------------------------------------
xspec.Plot.device = '/null'
xspec.Plot.xAxis = 'keV'
xspec.Plot('data')
x = np.array(xspec.Plot.x())     # channel energy [keV]
y = np.array(xspec.Plot.y())     # counts / s / keV

# optional display rebinning: average N adjacent bins (energy density preserved
# since the native bins are uniform 0.1 keV)
if args.rebin > 1:
    k = args.rebin
    nb = (len(x) // k) * k
    x = x[:nb].reshape(-1, k).mean(axis=1)
    y = y[:nb].reshape(-1, k).mean(axis=1)

# --- 4. plot: folded response + overlaid input source ----------------------
#title += '  [+ARF: physical counts/s]' if ARF else '  [RMF only: shape, arb. norm]'
fig, ax = plt.subplots(figsize=(5,4))
ax.step(x, y, where='mid', lw=0.9, color='C0', label='detector response')
ax.set_yscale('log')
binw = float(np.median(np.diff(x))) if x.size > 1 else 0.1   # keV (0.1 native x rebin)
ax.set(xlabel='Measured energy [keV]',
       ylabel=f'counts s$^{{-1}}$ keV$^{{-1}}$ ({binw:g} keV bins)', title=title)
# minor ticks on the (left) log y-axis: 2..9 x 10^n between the decade majors
ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=100))
ax.tick_params(axis='y', which='major', length=5, left=True)
ax.tick_params(axis='y', which='minor', length=2.5, left=True)

if args.source == 'crab':
    # overlay the INPUT photon spectrum (power law) on a twin axis
    ax.set_xscale('log')
    ax.set_xlim(2, 1.1 * x.max())
    ax.set_ylim(10e-7, 10e4)
    ax2 = ax.twinx()
    ein = np.logspace(np.log10(2), np.log10(x.max()), 400)
    photons = args.crab_norm * ein ** (-args.crab_index)   # ph/cm2/s/keV
    ax2.plot(ein, photons, 'r--', lw=1.4, label='input source (Crab power law)')
    ax2.set_yscale('log')
    ax2.set_ylabel('input photons cm$^{-2}$ s$^{-1}$ keV$^{-1}$', color='r')
    ax2.tick_params(axis='y', colors='r')
    # single combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
else:
    # overlay the INPUT source: a delta spike at each line energy
    peak = y.max()
    for i, E in enumerate(LINES):
        ax.plot([E, E], [1e-5, peak], color='r', lw=1.2,
                label='input source (delta lines)' if i == 0 else None)
        ax.plot(E, peak, 'rv', ms=7)
    ax.set_xlim(0, 1.1 * max(LINES))
    ax.legend()
    #ax.set_ylim(10e-4,1)
    #ax.set_xlim(115,172)

fig.tight_layout()
fig.savefig(out, dpi=150)
print(f'source: {args.source}')
print(f'wrote {out}')

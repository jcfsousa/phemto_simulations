#!/usr/bin/env python3
"""Plot the PHEMTO Laue-lens effective area.

Data from F. Frontera (11-05-2026):
  Aeff_phemto_0.5E_Si.txt     fine grid, DeltaE = E/2, columns: E[keV] Aeff[cm2]
  Aeff_phemto_logbinE_Si.txt  8 log-spaced bins, columns: E_low E_high Aeff[cm2]
"""
import os
import numpy as np
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
fine_path = os.path.join(here, 'Aeff_phemto_0.5E_Si.txt')
logbin_path = os.path.join(here, 'Aeff_phemto_logbinE_Si.txt')

# --- load ---------------------------------------------------------------
E, Aeff = np.loadtxt(fine_path, comments='#', unpack=True)
e_lo, e_hi, aeff_bin = np.loadtxt(logbin_path, comments='#', unpack=True)

# --- report -------------------------------------------------------------
i_max = int(np.argmax(Aeff))
print(f'Fine grid : {E.size} points, {E.min():.0f}-{E.max():.0f} keV')
print(f'Peak Aeff : {Aeff[i_max]:.1f} cm^2 at {E[i_max]:.0f} keV')
print(f'Log bins  : {e_lo.size} intervals, {e_lo.min():.0f}-{e_hi.max():.0f} keV')

# --- plot ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(E, Aeff, color='tab:blue', lw=1.6,
        label=r'$\Delta E = E/2$ (fine grid)')

# log-binned values drawn as a step/histogram across each interval
edges = np.append(e_lo, e_hi[-1])
ax.stairs(aeff_bin, edges, color='tab:orange', lw=1.8,
          label='log-binned (8 intervals)')

ax.set_xscale('log')
ax.set_xlabel('Energy [keV]')
ax.set_ylabel(r'Effective area [cm$^2$]')
ax.set_title('PHEMTO Laue lens effective area (Si(111))')
ax.grid(True, which='both', alpha=0.3)
ax.legend()
fig.tight_layout()

out = os.path.join(here, 'Aeff_phemto_plot.png')
fig.savefig(out, dpi=150)
print(f'Saved -> {out}')
plt.show()

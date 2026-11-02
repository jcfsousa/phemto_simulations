#!/usr/bin/env python3
"""Propagate a source model through the PHEMTO Laue-lens effective area.

Input  : source photon spectrum  S(E)      [ph / keV / cm^2 / s]
Lens   : effective area           Aeff(E)   [cm^2]   (Frontera, DeltaE=E/2)
Output : collected photon flux    F(E) = S(E) * Aeff(E)   [ph / keV / s]

F(E) is the flux focused by the lens *as a whole*; the next step will
spread it uniformly over the crystals built in make_ring_source.py.

Test model: Crab Nebula power law.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

# --- effective area -----------------------------------------------------
AEFF_FILE = ('/local/home/jf285468/documents/phd/phemto/phemto_simulations/'
             'megalib/optics/laue_lenses/Aeff_phemto_0.5E_Si.txt')
E_aeff, Aeff_vals = np.loadtxt(AEFF_FILE, comments='#', unpack=True)


def effective_area(E):
    """Lens effective area [cm^2] at energy E [keV], linearly interpolated.
    Returns 0 outside the tabulated range (no reflection there)."""
    return np.interp(E, E_aeff, Aeff_vals, left=0.0, right=0.0)


# --- source model -------------------------------------------------------
def crab_powerlaw(E):
    """Crab Nebula photon spectrum [ph/keV/cm^2/s] (Toor & Seward 1974).
    dN/dE = K * E^-gamma,  K = 9.7,  gamma = 2.1,  E in keV."""
    K, gamma = 9.7, 2.1
    return K * np.power(E, -gamma)


# --- propagate ----------------------------------------------------------
# work on the native Aeff energy grid (50-1000 keV, 1 keV steps)
E = E_aeff
S = crab_powerlaw(E)              # ph/keV/cm^2/s
F = S * effective_area(E)         # ph/keV/s  -> flux after the lens

# total collected rate in the 50-400 keV PHEMTO band
band = (E >= 50) & (E <= 400)
rate = np.trapz(F[band], E[band])
print(f'Collected Crab rate 50-400 keV: {rate:.2f} ph/s')
print(f'Peak collected flux: {F.max():.3f} ph/keV/s at {E[np.argmax(F)]:.0f} keV')

# --- plot ---------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(E, S, color='tab:green')
axes[0].set(xscale='log', yscale='log', xlabel='Energy [keV]',
            ylabel=r'$S(E)$ [ph keV$^{-1}$ cm$^{-2}$ s$^{-1}$]',
            title='Input: Crab power law')

axes[1].plot(E, effective_area(E), color='tab:blue')
axes[1].set(xscale='log', xlabel='Energy [keV]',
            ylabel=r'$A_{\rm eff}$ [cm$^2$]', title='Laue lens effective area')

axes[2].plot(E, F, color='tab:red')
axes[2].set(xscale='log', yscale='log', xlabel='Energy [keV]',
            ylabel=r'$F(E)=S\,A_{\rm eff}$ [ph keV$^{-1}$ s$^{-1}$]',
            title='Output: flux after the lens')

for ax in axes:
    ax.grid(True, which='both', alpha=0.3)
fig.suptitle('Crab Nebula through PHEMTO Laue lens', fontsize=13)
fig.tight_layout()
plt.show()

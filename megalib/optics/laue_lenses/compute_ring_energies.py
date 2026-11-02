#!/usr/bin/env python3
"""Compute the ring energies available between 50 and 400 keV for which the
Si(111) crystal rings do NOT intersect radially.

Geometry (same constants as make_ring_source.py / generate_laue_geo.py):
    theta_b = arcsin(hc / (2 d_hkl E))        Bragg angle
    r(E)    = focal * tan(2 theta_b)          ring radius [cm]

Each ring is an annulus of radial width l_f (the focusing crystal size).
Two rings do not intersect if their radii differ by at least l_f (+ margin).
r(E) decreases as E increases, so we start at r(E_min) (largest radius) and
step inward by (l_f + margin), inverting each radius back to an energy:
    E(r) = hc / (2 d_hkl * sin( 0.5 * arctan(r / focal) ))
"""
import numpy as np

# --- constants (keep in sync with make_ring_source.py) ------------------
d_hkl = 3.135      # A, Si(111)
hc = 12.4          # A*keV
focal = 2000.0     # cm
l_f = 3.0          # cm, radial crystal size (annulus width)
l_nf = 1.0         # cm, tangential crystal size

E_min, E_max = 50.0, 450.0
margin = 0.0       # cm, extra radial clearance between rings (0 = tiles touch)


def radius(E):
    theta_b = np.arcsin(hc / (2 * d_hkl * E))
    return focal * np.tan(2 * theta_b)


def energy(r):
    theta_b = 0.5 * np.arctan(r / focal)
    return hc / (2 * d_hkl * np.sin(theta_b))


def n_crystals(r):
    return int(2 * np.pi * r / l_nf)


def available_energies(step=None):
    """Greedy packing from E_min (largest r) inward to E_max."""
    step = (l_f + margin) if step is None else step
    r_lo, r_hi = radius(E_max), radius(E_min)   # r_lo < r_hi
    energies = []
    r = r_hi
    while r >= r_lo - 1e-9:
        energies.append(energy(r))
        r -= step
    # make sure the high-energy edge (E_max) is represented without overlap:
    # add it only if it clears the last ring by at least one step
    if energies and (radius(energies[-1]) - r_lo) >= step - 1e-9:
        energies.append(E_max)
    return np.array(energies)


def main():
    print(f'Band {E_min:.0f}-{E_max:.0f} keV, annulus width l_f={l_f} cm, '
          f'margin={margin} cm')
    print(f'r({E_min:.0f} keV) = {radius(E_min):.2f} cm   '
          f'r({E_max:.0f} keV) = {radius(E_max):.2f} cm   '
          f'radial span = {radius(E_min)-radius(E_max):.2f} cm')

    E = available_energies()
    print(f'\n{len(E)} non-overlapping rings available:\n')
    print(f"{'#':>3} {'E [keV]':>9} {'r [cm]':>9} {'dr_prev [cm]':>12} {'N_xtal':>7}")
    r_prev = None
    for k, e in enumerate(E):
        r = radius(e)
        dr = '' if r_prev is None else f'{r_prev - r:11.3f}'
        print(f'{k:>3} {e:9.2f} {r:9.3f} {dr:>12} {n_crystals(r):>7}')
        r_prev = r

    print(f'\nTotal crystals over all rings: '
          f'{sum(n_crystals(radius(e)) for e in E)}')
    print('\nEnergy list (keV):')
    print('[' + ', '.join(f'{e:.1f}' for e in E) + ']')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Build the ARF (ancillary response / effective area) to accompany the RMFs.

The RMF (build_rmf.py) carries only the normalized redistribution shape; the
detection efficiency was deliberately factored out. The ARF puts it back as an
energy-dependent EFFECTIVE AREA in cm^2:

    ARF_det(E) = Aeff_lens(E) * efficiency_det(E)
               = Aeff_lens(E) * N_detected_det(E) / N_sim

  * Aeff_lens(E)  : PHEMTO Laue-lens effective area (Frontera table), the area
                    over which the source flux is collected and focused.
  * efficiency_det: fraction of the focused photons recorded by that detector,
                    taken from the RMF run (total masked counts / N_sim).

Folded together, predicted counts obey
    C(E_meas) = exposure * sum_E flux(E) * ARF(E) * RMF(E_meas | E)
which is the standard OGIP forward model, so ARF (cm^2) x RMF (unitless) must be
built from the SAME run -- hence we read build_rmf.py's rmf_matrices.npz.

Inputs  : rmf/rmf_output/rmf_matrices.npz   (true_E, *_total, n_sim)
          optics/laue_lenses/Aeff_phemto_0.5E_Si.txt   (lens Aeff, DeltaE=E/2)
Outputs : rmf/rmf_output/arf_curves.npz, ARF.png, ARF_<det>.txt
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'rmf_output')
NPZ = os.path.join(OUT_DIR, 'rmf_matrices.npz')
AEFF_LENS = os.path.join(HERE, '..', '..', 'optics', 'laue_lenses',
                         'Aeff_phemto_0.5E_Si.txt')
DETECTORS = ['Si', 'CdTe', 'Combined']
SHOW = False


def lens_effective_area(E):
    """Lens Aeff [cm^2] at energy E [keV] (linear interp, 0 outside table)."""
    e_tab, a_tab = np.loadtxt(AEFF_LENS, comments='#', unpack=True)
    return np.interp(E, e_tab, a_tab, left=0.0, right=0.0)


def main():
    if not os.path.exists(NPZ):
        raise SystemExit('run build_rmf.py first (rmf_matrices.npz missing)')
    d = np.load(NPZ)
    true_E = d['true_energy']
    n_sim = float(d['n_sim'])

    Aeff_lens = lens_effective_area(true_E)

    arf = {}
    print(f'N_sim = {n_sim:g}')
    print(f'Lens Aeff [cm^2] at true E: '
          f'{dict(zip(true_E.astype(int), np.round(Aeff_lens, 0)))}')
    print(f"\n{'det':>9} {'E[keV]':>7} {'eff':>9} {'Aeff_lens':>11} {'ARF[cm^2]':>11}")
    for det in DETECTORS:
        eff = d[f'{det}_total'] / n_sim          # detector detection efficiency
        a = Aeff_lens * eff                      # full-instrument effective area
        arf[det] = a
        for E, ef, av in zip(true_E, eff, a):
            print(f'{det:>9} {E:>7.0f} {ef:>9.4f} '
                  f'{Aeff_lens[np.where(true_E == E)][0]:>11.0f} {av:>11.2f}')
        # per-detector OGIP-style text ARF (energy, effective area)
        np.savetxt(os.path.join(OUT_DIR, f'ARF_{det}.txt'),
                   np.column_stack([true_E, a]),
                   header='E_true[keV]  ARF[cm^2]  (Aeff_lens * detector efficiency)')

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(true_E, Aeff_lens, 'k--', lw=1, label='Lens Aeff (no detector)')
    for det in DETECTORS:
        ax.plot(true_E, arf[det], 'o-', label=f'ARF {det}')
    ax.set(xlabel='True energy [keV]', ylabel=r'Effective area [cm$^2$]',
           title='ARF = lens Aeff x detector efficiency')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'ARF.png'), dpi=150)
    print(f"\n  wrote {os.path.join(OUT_DIR, 'ARF.png')}")

    np.savez(os.path.join(OUT_DIR, 'arf_curves.npz'),
             true_energy=true_E, Aeff_lens=Aeff_lens,
             **{f'ARF_{det}': arf[det] for det in DETECTORS})
    print(f"  wrote {os.path.join(OUT_DIR, 'arf_curves.npz')}")
    for det in DETECTORS:
        print(f"  wrote {os.path.join(OUT_DIR, f'ARF_{det}.txt')}")

    if SHOW:
        plt.show()


if __name__ == '__main__':
    main()

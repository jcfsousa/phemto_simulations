#!/usr/bin/env python3
"""Convert a CadTool CIE map (parquet) into a MEGAlib EnergyLossMap (LossMap.dat).

Produces a DEPTH-ONLY charge-collection map: for the pixel-cell corner column
(X == X.min, Y == Y.min) it takes the CIE(z) profile and writes it as a MEGAlib
MFunction3D "DP" map that is flat in x,y. Point a Voxel3D detector at it with
    <Detector>.EnergyLossMap  <this_file>.dat
and keep the existing EnergyResolution lines (the CCE layers under them).

The map is applied in cosima at the TRUE interaction position before
voxelization, so the depth dependence is real and revan then reconstructs from
the degraded energies.

Usage
-----
    cie_to_lossmap.py <cie_parquet> [-o OUTDIR] [-d] [--xy-halfsize CM] [--flip-z]

    <cie_parquet>        path to the CadTool CIE parquet (columns Qf, Qinit, X, Y, Z in um)
    -o / --output OUTDIR output directory (default: this script's folder)
    -d / --debug         show a plot of the generated LossMap
    --xy-halfsize CM     detector half-size in x,y [cm] for the flat grid (default 3.6)
    --flip-z             flip the z axis (anode orientation) if the CCE tail is on the wrong side
"""
import os
import argparse
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
#  CIE parquet -> depth-only LossMap.dat
# --------------------------------------------------------------------------
def build_lossmap(cie_path, out_dir, xy_halfsize=3.6, z_halfsize=0.1, flip_z=False,
                  test=False, hecht=False, mutau_e=None, mutau_h=None, bias=None):
    """Build a MEGAlib EnergyLossMap .dat and return its path.

    Normal mode: the depth profile CCE(z) comes from the CadTool CIE parquet
    (corner column X==X.min, Y==Y.min). Test mode (`test=True`): CCE(z) comes
    from the analytic formula in example_depth_dependantE/
    EnergyResolutionTester.MapCreator.cxx instead of the parquet.

    The map is written in the sensitive-volume LOCAL frame (cm), centered at the
    detector center: x,y in [-xy_halfsize,+xy_halfsize], z in
    [-z_halfsize,+z_halfsize] with z=0 the mid-depth plane. The z profile is
    padded (flat hold) out to +-z_halfsize so the map spans exactly the detector
    thickness -- otherwise MEGAlib (MDVoxel3D::Validate) would linearly STRETCH
    the map to the detector bounds and misplace the depths."""
    zh = float(z_halfsize)
    if test:
        # MapCreator.cxx:  Value = -1/(50*(z+1.01)) + 0.95  (clamped to >= 0.01),
        # defined there over z in [-1,1]. Evaluate it on z normalized to the
        # detector thickness (z_norm = z/zh) so the same gradient spans the real
        # detector. No parquet needed.
        n = 10000
        z_cm = np.linspace(-zh, zh, n)
        z_norm = z_cm / zh
        cce = -1.0 / (50 * (z_norm + 1)) + 1.01
        cce = np.where(cce < 0.0, 0.01, cce)
        stem = 'test_depth_formula'
        src_msg = 'MapCreator.cxx formula (test)'
    elif hecht:
        # Hecht equation (planar electrodes, both carriers), total induced charge.
        # Electrons drift to the ANODE, holes to the CATHODE. Convention (matches
        # the CadTool CIE map): ANODE at z=-zh, CATHODE at z=+zh, so
        #   d_e = z + zh  (electron drift distance to the anode)
        #   d_h = zh - z  (hole drift distance to the cathode)
        #   CCE(z) = (le/d)[1 - exp(-d_e/le)] + (lh/d)[1 - exp(-d_h/lh)]
        # le = mutau_e*|E|, lh = mutau_h*|E|, |E| = |bias|/d, d = 2*zh.
        # The bias is NEGATIVE in these detectors; only its magnitude enters.
        # Use --flip-z if the anode is on the +z face in your geometry.
        if mutau_e is None or mutau_h is None or bias is None or bias == 0:
            raise SystemExit('hecht mode needs --mutau-e, --mutau-h (cm^2/V) and --bias (V, nonzero)')
        d = 2 * zh                                           # cm (thickness)
        Emag = abs(bias) / d                                 # V/cm (field magnitude)
        le = mutau_e * Emag                                  # cm (electron drift length)
        lh = mutau_h * Emag                                  # cm (hole drift length)
        n = 100
        z_cm = np.linspace(-zh, zh, n)
        d_e = z_cm + zh                                      # electron -> anode (z=-zh)
        d_h = zh - z_cm                                      # hole -> cathode (z=+zh)
        cce = (le / d) * (1 - np.exp(-d_e / le)) + (lh / d) * (1 - np.exp(-d_h / lh))
        cce = np.clip(cce, 0.0, 1.0)
        stem = 'hecht_depth'
        src_msg = (f'Hecht (mutau_e={mutau_e:g}, mutau_h={mutau_h:g} cm^2/V, '
                   f'|bias|={abs(bias):g} V, d={d:g} cm, le={le:.4g} cm, lh={lh:.4g} cm; '
                   f'anode z<0, electrons->anode)')
    else:
        df = pd.read_parquet(cie_path)
        df['CIE'] = df['Qf'] / df['Qinit']                   # -> ~[0, 1]
        # Depth distribution at the minimum X and minimum Y (pixel-cell corner).
        sub = df[(df['X'] == df['X'].min()) & (df['Y'] == df['Y'].min())]
        prof = sub.groupby('Z')['CIE'].mean().sort_index()   # mean collapses duplicate Z rows
        z_cm = prof.index.to_numpy() * 1e-4                   # um -> cm
        cce = prof.to_numpy()
        stem = os.path.splitext(os.path.basename(cie_path))[0]
        src_msg = f'CIE corner (X={df["X"].min():.0f}, Y={df["Y"].min():.0f}) um'

    if flip_z:
        # Flip so the anode side matches MEGAlib's local z for your CdTe.
        z_cm, cce = -z_cm[::-1], cce[::-1]

    # Pad the z axis out to the detector faces (+-z_halfsize) with flat-hold, so
    # the map exactly matches the sensitive thickness and is NOT rescaled.
    if z_cm[0] > -zh:
        z_cm = np.concatenate(([-zh], z_cm)); cce = np.concatenate(([cce[0]], cce))
    if z_cm[-1] < zh:
        z_cm = np.concatenate((z_cm, [zh])); cce = np.concatenate((cce, [cce[-1]]))

    xy = float(xy_halfsize)
    XA = YA = [-xy, xy]                                       # flat in x,y (depth-only)

    out_path = os.path.join(out_dir, f'{stem}.LossMap.dat')
    with open(out_path, 'w') as f:
        f.write('XA ' + ' '.join(f'{v:g}' for v in XA) + '\n')
        f.write('YA ' + ' '.join(f'{v:g}' for v in YA) + '\n')
        f.write('ZA ' + ' '.join(f'{z:.6g}' for z in z_cm) + '\n')
        for x in XA:
            for y in YA:
                for z, c in zip(z_cm, cce):
                    f.write(f'DP {x:g} {y:g} {z:.6g} {c:.6g}\n')
        f.write('EN\n')

    print(f'Wrote {out_path}')
    print(f'  source: {src_msg}, {z_cm.size} z-nodes, '
          f'CCE range [{cce.min():.4f}, {cce.max():.4f}]'
          + (' (z flipped)' if flip_z else ''))
    return out_path


# --------------------------------------------------------------------------
#  Debug plotting (from ~/tmp/mrga_tmp.py) -- only used with -d
# --------------------------------------------------------------------------
def _load_lossmap(path):
    xa = ya = za = None
    dp = []
    with open(path) as f:
        for ln in f:
            t = ln.split()
            if not t:
                continue
            if t[0] == 'XA':
                xa = np.array(t[1:], float)
            elif t[0] == 'YA':
                ya = np.array(t[1:], float)
            elif t[0] == 'ZA':
                za = np.array(t[1:], float)
            elif t[0] == 'DP':
                dp.append([float(v) for v in t[1:5]])
            elif t[0] == 'EN':
                break
    dp = np.array(dp)                                         # columns: x, y, z, value
    V = np.full((xa.size, ya.size, za.size), np.nan)
    ix = {v: i for i, v in enumerate(xa)}
    iy = {v: i for i, v in enumerate(ya)}
    iz = {v: i for i, v in enumerate(za)}
    for x, y, z, val in dp:
        V[ix[min(ix, key=lambda a: abs(a - x))],
          iy[min(iy, key=lambda a: abs(a - y))],
          iz[min(iz, key=lambda a: abs(a - z))]] = val
    return xa, ya, za, V


def plot_lossmap(path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    xa, ya, za, V = _load_lossmap(path)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(za, np.nanmean(V, axis=(0, 1)), 'o-')
    ax[0].set(xlabel='z (depth) [cm]', ylabel='CCE (mean over x,y)',
              title='Depth profile')
    ax[0].grid(alpha=.3)
    iy = ya.size // 2
    slice_ = V[:, iy, :]
    im = ax[1].pcolormesh(za, xa, slice_, shading='auto', cmap='viridis',
                          norm=LogNorm(vmin=0.8, vmax=1.0))
    ax[1].set(xlabel='z [cm]', ylabel='x [cm]', title=f'CCE slice at y={ya[iy]:.3f} cm')

    ax[0].set_yscale('log')
    ax[0].set_ylim(0.8, 1.02)
    fig.colorbar(im, ax=ax[1], label='CCE (log)')
    fig.suptitle(os.path.basename(path))
    fig.tight_layout()
    plt.show()


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='CadTool CIE parquet -> MEGAlib EnergyLossMap .dat')
    ap.add_argument('cie_path', nargs='?', default=None,
                    help='path to the CadTool CIE parquet (not needed with -t)')
    ap.add_argument('-o', '--output', default=HERE,
                    help='output directory (default: this script folder)')
    ap.add_argument('-d', '--debug', action='store_true', help='show a plot of the map')
    ap.add_argument('-t', '--test', action='store_true',
                    help='test map: CCE(z) from the MapCreator.cxx formula instead of the CIE parquet')
    ap.add_argument('-ht', '--hecht', action='store_true',
                    help='Hecht map: CCE(z) from the Hecht equation (needs --mutau-e, --mutau-h, --bias)')
    ap.add_argument('--mutau-e', type=float, default=None,
                    help='electron mobility-lifetime product mu*tau_e [cm^2/V] (Hecht mode)')
    ap.add_argument('--mutau-h', type=float, default=None,
                    help='hole mobility-lifetime product mu*tau_h [cm^2/V] (Hecht mode)')
    ap.add_argument('--bias', type=float, default=None,
                    help='detector bias voltage [V] (Hecht mode); negative is fine, magnitude is used')
    ap.add_argument('--xy-halfsize', type=float, default=3.6,
                    help='detector x,y sensitive half-size [cm] (default 3.6 = CdTe3x3Big)')
    ap.add_argument('--z-halfsize', type=float, default=0.1,
                    help='detector z sensitive half-size [cm] (default 0.1 = 2 mm thick)')
    ap.add_argument('--flip-z', action='store_true', help='flip the z (anode) orientation')
    args = ap.parse_args()

    if not args.test and not args.hecht and args.cie_path is None:
        ap.error('cie_path is required unless -t/--test or -ht/--hecht is given')

    os.makedirs(args.output, exist_ok=True)
    out_path = build_lossmap(args.cie_path, args.output, xy_halfsize=args.xy_halfsize,
                             z_halfsize=args.z_halfsize, flip_z=args.flip_z, test=args.test,
                             hecht=args.hecht, mutau_e=args.mutau_e, mutau_h=args.mutau_h,
                             bias=args.bias)

    if args.debug:
        plot_lossmap(out_path)


if __name__ == '__main__':
    main()

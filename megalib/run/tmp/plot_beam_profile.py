#!/usr/bin/env python3
"""Visualize a MEGAlib/cosima 2D-Function beam profile (MapProfileBeam input).

File format (Cosima.pdf sec. 6.2):
  IP <NONE|LIN>            interpolation
  XA x0 x1 ...  | XB lo hi n   x-axis bin CENTERS in cm (equidistant)
  YA y0 y1 ...  | YB lo hi n   y-axis bin CENTERS in cm
  AP ix iy val            value of the function at grid node (ix,iy); IDs start at 0
  EN                      end of data
"""
import os
import numpy as np
import matplotlib.pyplot as plt

DAT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'BeamProfile2D.dat')


def _axis(tokens):
    """Return bin-center array from an XA/YA (centers) or XB/YB (lo hi n) line."""
    key = tokens[0]
    vals = [float(v) for v in tokens[1:]]
    if key in ('XB', 'YB'):
        lo, hi, n = vals[0], vals[1], int(vals[2])
        return np.linspace(lo, hi, n)
    return np.array(vals)


def load_profile(path):
    x = y = None
    interp = 'LIN'
    aps = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tok = line.split()
            key = tok[0]
            if key == 'IP':
                interp = tok[1]
            elif key in ('XA', 'XB'):
                x = _axis(tok)
            elif key in ('YA', 'YB'):
                y = _axis(tok)
            elif key == 'AP':
                aps.append((int(tok[1]), int(tok[2]), float(tok[3])))
            elif key == 'EN':
                break
    # grid node values Z[iy, ix], zeros where not listed
    Z = np.zeros((y.size, x.size))
    for ix, iy, val in aps:
        Z[iy, ix] = val
    return x, y, Z, interp


def main():
    x, y, Z, interp = load_profile(DAT)
    dx = x[1] - x[0] if x.size > 1 else 1.0
    dy = y[1] - y[0] if y.size > 1 else 1.0
    extent = [x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2]

    print(f'Interpolation : {interp}')
    print(f'x grid : {x.size} bins, {x.min():.3f}..{x.max():.3f} cm (step {dx:.3f})')
    print(f'y grid : {y.size} bins, {y.min():.3f}..{y.max():.3f} cm (step {dy:.3f})')
    print(f'profile : nonzero in x=[{x[np.any(Z>0,axis=0)].min():.2f},'
          f'{x[np.any(Z>0,axis=0)].max():.2f}] cm, peak value {Z.max():g}')

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(Z, origin='lower', extent=extent, aspect='auto',
                   cmap='viridis', interpolation='nearest')
    # mark the grid nodes
    gx, gy = np.meshgrid(x, y)
    ax.scatter(gx, gy, s=8, c='white', alpha=0.4, linewidths=0)
    fig.colorbar(im, ax=ax, label='beam intensity (relative)')
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')
    ax.set_title('BeamProfile2D.dat  (MapProfileBeam start-position map)')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()

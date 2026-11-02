
import os, sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print('Script usage:')
    print('make_ring_source_SOURCE.py geometry_file_path.geo.setup')
    sys.exit()

geo_path = sys.argv[1]   # Geometry file path (.geo.setup)

# =====================================================================
#  Realistic source generator for the PHEMTO Laue lens.
#  Unlike the _RMF version (flat, per-energy, for the response matrix),
#  this folds in the lens EFFECTIVE AREA and a SOURCE MODEL:
#      per-crystal flux = [ integral over DeltaE=E/2 of S(E)*Aeff(E) ] / n
#  All rings are written into ONE .source file so cosima distributes the
#  total photon budget across crystals in proportion to their real flux,
#  i.e. a realistic focused observation of the source through the lens.
#
#  Geometry / crystal-count logic (ring_geometry) is identical to the
#  _RMF script and to ../../optics/laue_lenses/generate_laue_geo.py.
# =====================================================================

# Total photons for the whole observation, distributed across all crystals
# in proportion to their flux. NOTE: "10e6" is ambiguous; this is 1e6.
N_PHOTONS = int(1e6)

# Energies = non-overlapping ring energies (matches generate_laue_geo.py)
E_list = [50.0, 51.0, 52.0, 53.0, 54.1, 55.2, 56.4, 57.6, 58.9, 60.2, 61.6,
          63.1, 64.6, 66.2, 67.9, 69.7, 71.6, 73.6, 75.7, 77.9, 80.3, 82.8,
          85.5, 88.4, 91.4, 94.7, 98.2, 102.0, 106.1, 110.6, 115.4, 120.7,
          126.4, 132.8, 139.8, 147.7, 156.4, 166.3, 177.5, 190.3, 205.0,
          222.3, 242.8, 267.4, 297.6, 335.4, 384.3, 449.8]

# Sillicon lattice length (111)
d_hkl = 3.135    # A
hc = 12.4        # A*keV
focal = 2000.0   # cm
l_f = 3          # cm, focussing size of the crystal (radial)
l_nf = 1         # cm, non focussing size of the crystal (tangential)
order = [1]      # diffraction orders (not implemented)

SHOW_PLOT = False   # set True to see the geometry + flux diagnostics


def ring_geometry(E):
    """Ring geometry for energy E [keV] -- SAME logic as generate_laue_geo.py.
    Returns (theta_b [rad], r [cm], n_crystals)."""
    theta_b = np.arcsin(np.clip(hc / (2 * d_hkl * E), -1.0, 1.0))
    r = focal * np.tan(2 * theta_b)
    n = int(2 * np.pi * r / l_nf)
    return theta_b, r, n


################## LENSES EFF AREA ######################
# From Filippo Frontera 11-05-2026. Aeff_phemto_0.5E_Si.txt gives the lens
# effective area assuming an energy band DeltaE = E/2 (Si(111), 1st order plus
# 3rd/4th order reflectivity extending above 400 keV).
fine_path = os.path.join('../../optics/laue_lenses/', 'Aeff_phemto_0.5E_Si.txt')
E_tab, Aeff_tab = np.loadtxt(fine_path, comments='#', unpack=True)
print(f'Aeff table: {E_tab.size} pts, {E_tab.min():.0f}-{E_tab.max():.0f} keV, '
      f'peak {Aeff_tab.max():.0f} cm^2 at {E_tab[np.argmax(Aeff_tab)]:.0f} keV')


def effective_area(E):
    """Lens effective area [cm^2] at energy E [keV], linearly interpolated.
    Zero outside the tabulated range (no reflection there)."""
    return np.interp(E, E_tab, Aeff_tab, left=0.0, right=0.0)


# ---------------------------------------------------------------------
#  Source models [ph/keV/cm^2/s]
# ---------------------------------------------------------------------
def crab_powerlaw(E):
    """Crab Nebula power law [ph/keV/cm^2/s] (Toor & Seward 1974).
    dN/dE = K * E^-gamma,  K = 9.7, gamma = 2.1, E in keV."""
    K, gamma = 9.7, 2.1
    return K * np.power(E, -gamma)


def flat_flux(E):
    """Flat source spectrum: constant 1 ph/keV/cm^2/s at every energy."""
    return np.ones_like(np.asarray(E, dtype=float))


SOURCE_MODELS = {'crab': crab_powerlaw, 'flat': flat_flux}


def choose_source_model():
    """Ask the user which source model to use and return (name, function)."""
    names = list(SOURCE_MODELS)
    print('\nAvailable source models:')
    for k, name in enumerate(names):
        print(f'  [{k}] {name}')
    while True:
        ans = input('Select source model (name or index) [crab]: ').strip() or 'crab'
        if ans in SOURCE_MODELS:
            return ans, SOURCE_MODELS[ans]
        if ans.isdigit() and int(ans) < len(names):
            return names[int(ans)], SOURCE_MODELS[names[int(ans)]]
        print(f'  -> unknown model "{ans}", try again.')


model_name, source_model = choose_source_model()
print(f'Using source model: {model_name}')


def ring_flux(E0):
    """Photon rate [ph/s] the whole lens focuses for a ring at energy E0,
    integrating S(E)*Aeff(E) over the ring band DeltaE = E0/2 (0.75-1.25 E0)."""
    e_lo, e_hi = 0.75 * E0, 1.25 * E0
    e = np.linspace(e_lo, e_hi, 400)
    return np.trapz(source_model(e) * effective_area(e), e)


# ---------------------------------------------------------------------
#  Build ring geometry + per-crystal flux (Aeff x source model)
# ---------------------------------------------------------------------
rings = []
for i, E in enumerate(E_list):
    theta_b, r, n_of_crystals = ring_geometry(E)

    phi = np.linspace(0, 2 * np.pi, n_of_crystals, endpoint=False)
    source_x, source_y = r * np.cos(phi), r * np.sin(phi)
    source_z = np.full(n_of_crystals, focal)

    if E < 50:
        # Below 50 keV no diffraction: perpendicular incidence (theta = 0).
        dir_x = np.zeros(n_of_crystals)
        dir_y = np.zeros(n_of_crystals)
        dir_z = -source_z.astype(float)
    else:
        dir_x = 0.0 - source_x
        dir_y = 0.0 - source_y
        dir_z = 0.0 - source_z

    f_ring = ring_flux(E)                 # ph/s collected by this ring
    f_xtal = f_ring / n_of_crystals       # ph/s per crystal (uniform in ring)

    rings.append(dict(E=E, r=r, n=n_of_crystals,
                      sx=source_x, sy=source_y, sz=source_z,
                      dx=dir_x, dy=dir_y, dz=dir_z,
                      flux_ring=f_ring, flux_xtal=f_xtal))

total_flux = sum(rg['flux_ring'] for rg in rings)   # ph/s over the whole lens

# ---------------------------------------------------------------------
#  Report
# ---------------------------------------------------------------------
print(f'\nSource model: {model_name}   Total lens flux: {total_flux:.4g} ph/s')
print(f"{'E [keV]':>9} {'r [cm]':>9} {'N_xtal':>7} "
      f"{'ring flux [ph/s]':>17} {'per-xtal [ph/s]':>16} {'~events':>9}")
for rg in rings:
    exp_ev = N_PHOTONS * rg['flux_ring'] / total_flux if total_flux > 0 else 0
    print(f"{rg['E']:>9g} {rg['r']:>9.2f} {rg['n']:>7d} "
          f"{rg['flux_ring']:>17.4g} {rg['flux_xtal']:>16.4g} {exp_ev:>9.0f}")

# ---------------------------------------------------------------------
#  Diagnostic plots (optional)
# ---------------------------------------------------------------------
if SHOW_PLOT:
    E_grid = E_tab
    S_grid = source_model(E_grid)
    F_grid = S_grid * effective_area(E_grid)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(E_grid, S_grid, 'g'); axes[0].set(xscale='log', yscale='log',
        xlabel='Energy [keV]', ylabel=r'$S(E)$ [ph keV$^{-1}$cm$^{-2}$s$^{-1}$]',
        title=f'Input: {model_name}')
    axes[1].plot(E_grid, effective_area(E_grid), 'b'); axes[1].set(xscale='log',
        yscale='log', xlabel='Energy [keV]', ylabel=r'$A_{\rm eff}$ [cm$^2$]',
        title='Effective area')
    axes[2].plot(E_grid, F_grid, 'r'); axes[2].set(xscale='log', yscale='log',
        xlabel='Energy [keV]', ylabel=r'$F=S\,A_{\rm eff}$ [ph keV$^{-1}$s$^{-1}$]',
        title='After the lens')
    for a in axes:
        a.grid(True, which='both', alpha=0.3)
    fig.suptitle(f'{model_name} through PHEMTO Laue lens')
    fig.tight_layout()
    plt.show()

# ---------------------------------------------------------------------
#  Write ONE combined source file: all rings, each crystal weighted by its
#  flux. cosima splits N_PHOTONS across crystals in proportion to Flux.
# ---------------------------------------------------------------------
out_name = f'laue_{model_name}.source'
source_file = open(out_name, 'w')
source_file.write('Geometry         {0}\n'.format(geo_path))
source_file.write('PhysicsListEM    LivermorePol\n\n')
source_file.write('StoreSimulationInfo       all\n\n')
source_file.write('Run LaueSource\n')
source_file.write('LaueSource.FileName              LaueSource_{0}\n'.format(model_name))
source_file.write('LaueSource.NEvents               {0}\n\n'.format(N_PHOTONS))

xid = 0
for rg in rings:
    for i in range(rg['n']):
        xid += 1
        source_file.write('\n')
        source_file.write('LaueSource.Source Xtal{0}\n'.format(xid))
        source_file.write('Xtal{0}.ParticleType	1\n'.format(xid))
        source_file.write('Xtal{0}.Beam	MapProfileBeam	{1} {2} {3} 90 {4} {5} {6} BeamProfile2D.dat\n'.format(
            xid, rg['sx'][i], rg['sy'][i], rg['sz'][i], rg['dx'][i], rg['dy'][i], rg['dz'][i]))
        source_file.write('Xtal{0}.Spectrum	Mono {1}\n'.format(xid, rg['E']))
        source_file.write('Xtal{0}.Flux	{1}\n'.format(xid, rg['flux_xtal']))
        source_file.write('Xtal{0}.Polarization	Random\n'.format(xid))

source_file.close()
print(f'\nWrote {out_name}: {xid} crystals, NEvents={N_PHOTONS}, '
      f'model={model_name}')

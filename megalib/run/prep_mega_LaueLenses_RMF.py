from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.patches as mpatches
import os, sys
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
#  Response-matrix source generator for the PHEMTO Laue lens.
#  For EACH energy in E_list we write one cosima .source file that emits
#  N_PHOTONS photons for that (monochromatic) ring, distributed EQUALLY
#  over the ring's crystals. Running all of them builds the per-energy
#  detector response needed for the response matrix.
#
#  The number of crystals per ring uses the SAME logic as
#  ../../optics/laue_lenses/generate_laue_geo.py (ring_geometry), so each
#  source crystal maps 1:1 to a geometry crystal (same n, endpoint=False).
# =====================================================================

CdTe_matrix_size = [3]
dists = [1]
instruments_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/instruments'
output_path_simTra = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources/simTra_files'
sources_path = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/sources'
revan_config = 'comptons_klein-abs-comptEne.cfg'

config_lst = []
for matrix_size in CdTe_matrix_size:
    for dist in dists:
        config = f"config{matrix_size}x{matrix_size}_{dist}cm"
        config_lst.append(config)

# Photons generated per energy (per ring), split equally over its crystals.
N_PHOTONS = int(1e6)

E_list = np.append(np.arange(1,50,0.5), np.arange(50,1000,1)) 
#E_list = np.arange(286,1000,1.0)

# Sillicon lattice length (111)
# ref: Structure determination of the CoSi2.Si(111) interface by x-ray
#      standing-wave analysis, Frischer et al.
# d = a / np.sqrt(h**2 + k**2 + l**2)
# a_si = 5.431 A
# h,l,k are the miller indices- 1,1,1 
d_hkl = 3.135    # A
# d_hkl = 3.2665 # A, d for Germanium 111
hc = 12.4        # A*keV
focal = 2001   # cm, To focus on CdTe, Si and CdTe seperated by 1cm
l_f = 3          # cm, focussing size of the crystal (radial)
l_nf = 1         # cm, non focussing size of the crystal (tangential)
order = [1]      # How many orders of diffraction to use. Not implemented for now

SHOW_PLOT = False # set True to see the 3D layout (blocks on plt.show)


################################################################################
################################################################################
################################################################################

def ring_geometry(E):
    """Ring geometry for energy E [keV] -- SAME logic as generate_laue_geo.py.
    Returns (theta_b [rad], r [cm], n_crystals)."""
    # clip to the arcsin domain: below ~2 keV hc/(2 d E) > 1
    if E < 50:
        return 0, 0, 1
    theta_b = np.arcsin(np.clip(hc / (2 * d_hkl * E), -1.0, 1.0))
    r = focal * np.tan(2 * theta_b)
    n = int(2 * np.pi * r / l_nf)
    return theta_b, r, n


# ---------------------------------------------------------------------
#  Build the ring geometry (positions + photon directions) per energy
# ---------------------------------------------------------------------
rings = []

if SHOW_PLOT:
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    cmap = plt.cm.jet

for i, E in enumerate(E_list):
    theta_b, r, n_of_crystals = ring_geometry(E)
    source_z = np.full(n_of_crystals, focal)
    phi = np.linspace(0, 2 * np.pi, n_of_crystals, endpoint=False)
    source_x, source_y = r * np.cos(phi), r * np.sin(phi)

    if E < 50:
        # Below 50 keV the lens does not diffract: photons arrive perpendicular - at this stage I dont model X-ray mirror
        # to the detector (theta = 0, straight down the -z axis), NOT focused.
        dir_x = np.zeros(n_of_crystals)
        dir_y = np.zeros(n_of_crystals)
        dir_z = -source_z.astype(float)

    # Direction of the photons leaving each crystal
    else:
        # (x, y, z) of the crystals on the ring. endpoint=False -> n distinct
        # crystals, matching generate_laue_geo.py (no duplicate tile at phi=0).
        # Reference system: (0,0,0) = center of the detector top surface, z up.
        phi = np.linspace(0, 2 * np.pi, n_of_crystals, endpoint=False)
        source_x, source_y = r * np.cos(phi), r * np.sin(phi)

        # Focal spot coordinate
        focal_spot_x, focal_spot_y, focal_spot_z = 0.0, 0.0, -1.5 # FOCUS AT CDTE DETECTOR SURFACE
        dir_x = focal_spot_x - source_x
        dir_y = focal_spot_y - source_y
        dir_z = focal_spot_z - source_z

    rings.append(dict(E=E, r=r, n=n_of_crystals,
                      sx=source_x, sy=source_y, sz=source_z,
                      dx=dir_x, dy=dir_y, dz=dir_z))


            #ax.plot([source_x[j],dir_x[j]], [source_y[j],dir_y[j]], [source_z[j],dir_z[j]], color = (0,j/len(source_x)/2,j/len(source_x)))
if SHOW_PLOT:
    n_random = 10
    chosen = np.random.choice(len(rings), size=n_random, replace=False)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    cmap = plt.cm.jet

    for k, idx in enumerate(sorted(chosen)):
        rg = rings[idx]
        c = cmap(k / max(n_random - 1, 1))
        ax.scatter(rg['sx'], rg['sy'], rg['sz'], marker='.', color=c,
                   label=f"{int(np.round(rg['E'],0))} keV")
        # draw each photon's actual path: crystal -> crystal + direction
        step = max(1, rg['n'] // 100 )
        for j in range(0, rg['n'], step):
            ax.plot([rg['sx'][j], rg['sx'][j] + rg['dx'][j]],
                    [rg['sy'][j], rg['sy'][j] + rg['dy'][j]],
                    [rg['sz'][j], rg['sz'][j] + rg['dz'][j]],
                    color=c, alpha=0.3)
            ax.plot([rg['sx'][j], rg['sx'][j] + 0],
                    [rg['sy'][j], rg['sy'][j] + 0],
                    [rg['sz'][j], rg['sz'][j] - rg['dz'][j]/2],
                    color=c, alpha=0.3)

    R_lens = ring_geometry(50)[1]
    ang = np.linspace(0, 2*np.pi, 200)
    disc_verts = [list(zip(R_lens*np.cos(ang),
                           R_lens*np.sin(ang),
                           np.full_like(ang, focal)))]
    disc = Poly3DCollection(disc_verts, alpha=0.4,
                            facecolor='gray', edgecolor='dimgray')
    ax.add_collection3d(disc)
    lens_proxy = mpatches.Patch(facecolor='gray', alpha=0.15, edgecolor='dimgray',
                            label='Lens plane (50 keV aperture)')
    ax.scatter(0, 0, -1.5, color='black', marker='*', s=120, label='Focal point')
    ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); ax.set_zlabel('z [cm]')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [lens_proxy])
    plt.show()
    sys.exit(1)


################################################################################
################################################################################
################################################################################

# ---------------------------------------------------------------------
#  Write one source file per energy: N_PHOTONS split equally over the
#  ring's crystals (equal Flux -> cosima distributes NEvents equally).
# ---------------------------------------------------------------------

SourceName = 'LaueRing'
print(f'N_PHOTONS per energy: {N_PHOTONS}')
print(f"{'E [keV]':>9} {'r [cm]':>9} {'N_xtal':>7} {'photons/xtal':>13}")


for config in config_lst:
    geofile=f'{instruments_path}/PHEMTO_{config}.geo.setup'
    with open(f'./runCosima{config}.sh', mode='w') as f:
        total_crystals = 0
        for n, ring in enumerate(rings):
            E = ring['E']
            if E < 50 or E > 400: #X-ray mirror and second order difraction on LaueLenses.... Approx............
                string1= f"""# An example run for Cosima \n 
                 \nVersion          1  
                 \nGeometry         {geofile} 
                 \nPhysicsListEM    LivermorePol   // Allow polarization \n
                 \n# Output formats\n
                 StoreSimulationInfo       all\n
                 StoreCalibrated   true\n
                 StoreSimulationInfoIonization true\n
                 DiscretizeHits    true\n
                 \nRun {SourceName} \n  //(laue, xray focus), mono for Q100
                 {SourceName}.FileName              {output_path_simTra}/{SourceName}{E}keV_{config} \n
                 {SourceName}.NEvents               {N_PHOTONS}\n\n\n
                 {SourceName}.Source One \n
                 One.ParticleType        1 \n
                 One.Beam                HomogeneousBeam  0 0 10 0 0 -1 0.075  // psf diameter = 1.5mm\n
                 One.Spectrum            Mono  {E}\n
                 One.Flux                1\n 
                 One.Polarization Random"""

                # Polarized Source prepared
                source_file_name=f'{sources_path}/{SourceName}{E}keV_{config}.source'
                source_file=open(source_file_name,'w')
                source_file.write(string1)

            else: # Laue Lenses
                n_of_crystals = ring['n']
                source_x, source_y, source_z = ring['sx'], ring['sy'], ring['sz']
                dir_x, dir_y, dir_z = ring['dx'], ring['dy'], ring['dz']
                total_crystals += n_of_crystals

                source_file = open(f'{sources_path}/{SourceName}{E}keV_{config}.source', 'w')
                source_file.write('Geometry         {0}\n'.format(geofile))
                source_file.write('PhysicsListEM    LivermorePol\n')
                source_file.write('\n')
                source_file.write('StoreSimulationInfo       all\n')
                source_file.write('StoreCalibrated   true\n')
                source_file.write('StoreSimulationInfoIonization true\n')
                source_file.write('DiscretizeHits    true\n')
                source_file.write('\n')
                source_file.write('Run {0}\n'.format(SourceName))
                source_file.write('{0}.FileName              {1}/{2}{3}keV_{4}\n'.format(SourceName, output_path_simTra, SourceName, E, config))
                source_file.write('{0}.NEvents               {1}\n'.format(SourceName, N_PHOTONS))
                source_file.write('\n')
                    
                TEST = False

                if TEST:
                    phi_deg = np.degrees(np.arctan2(source_y, source_x)) % 360
                    targets = [0, 45, 90]
                    debug_idx = sorted({
                        int(np.argmin(np.abs((phi_deg - t + 180) % 360 - 180)))
                        for t in targets
                    })
                    for i in debug_idx:
                        source_file.write('\n')
                        source_file.write('LaueRing.Source Xtal{0}\n'.format(i + 1))
                        source_file.write('Xtal{0}.ParticleType	1\n'.format(i + 1))
                        source_file.write('Xtal{0}.Beam	MapProfileBeam	{1} {2} {3} 0 {4} {5} {6} {7}/BeamProfile2D.dat\n'.format(i + 1, source_x[i], source_y[i], source_z[i], dir_x[i], dir_y[i], dir_z[i], sources_path))
                        source_file.write('Xtal{0}.Spectrum	Mono {1}\n'.format(i + 1, E))
                        # Equal flux on every crystal -> the N_PHOTONS events are split
                        # equally over the ring (uniform distribution).
                        source_file.write('Xtal{0}.Flux	1\n'.format(i + 1))
                        source_file.write('Xtal{0}.Polarization	Random\n'.format(i + 1))
                else:
                    for i in range(n_of_crystals):
                        source_file.write('\n')
                        source_file.write('LaueRing.Source Xtal{0}\n'.format(i + 1))
                        source_file.write('Xtal{0}.ParticleType	1\n'.format(i + 1))
                        source_file.write('Xtal{0}.Beam	MapProfileBeam	{1} {2} {3} 0 {4} {5} {6} {7}/BeamProfile2D.dat\n'.format(i + 1, source_x[i], source_y[i], source_z[i], dir_x[i], dir_y[i], dir_z[i], sources_path))
                        source_file.write('Xtal{0}.Spectrum	Mono {1}\n'.format(i + 1, E))
                        # Equal flux on every crystal -> the N_PHOTONS events are split
                        # equally over the ring (uniform distribution).
                        source_file.write('Xtal{0}.Flux	1\n'.format(i + 1))
                        source_file.write('Xtal{0}.Polarization	Random\n'.format(i + 1))

            source_file.close()

            runCode1=f'{sources_path}/{SourceName}{E}keV_{config}.source'
            # For Cosima
            f.write(f"mdelay cosima 22; cosima -z -v 0 {runCode1} >> /dev/null & sleep 1;")
        f.write('\necho "Waiting for Cosima to run...."')
        f.write('\nwait')

    print(f'\nWrote {len(rings)} source files. Total crystals: {total_crystals}')


# Prep Revan
for config in config_lst:
    geofile=f'{instruments_path}/PHEMTO_{config}.geo.setup'
    with open(f"./runRevan{config}.sh", mode='w') as f:
        for E in E_list:
            source_file1=f'{output_path_simTra}/{SourceName}{E}keV_{config}'
            f.write(f"revan -c {revan_config} -a -n -f {source_file1}.inc1.id1.sim.gz -g {geofile} \n")


# Prep Run
with open(f"./runAll.sh", mode='w') as f:
    for config in config_lst:
        f.write(f'bash ./runCosima{config}.sh && echo "Cosima ended... Starting Revan..." && bash ./runRevan{config}.sh \n')
    f.close()

print('Run ./runAll.sh to run Cosima and Revan')



# Prep Parse for event effs, pre_process and specMaker
output_parser = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/rmf/data'

base_path_analysis_overall = '/local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall'



with open(f"./runParser.sh", mode='w') as f:
    f.write(f'source {base_path_analysis_overall}/env-phemto/bin/activate ;') # Activate python3 virtual environment
    for config in config_lst:
            ###### Parsing .tra -> .t3pa ######
            for E in E_list:
                tra_gz_filePath=f'{output_path_simTra}/{SourceName}{E}keV_{config}.inc1.id1.tra.gz'
                run_name = f"{SourceName}{E}keV_{config}"
                output_path_config = f"{output_parser}/{run_name}"


                #f.write(f'python3 {base_path_analysis_overall}/parse.py -f {tra_gz_filePath} -o {output_path_config}; ') # whole detector

                #f.write(f'python3 {base_path_analysis_overall}/polarimetry/pre_process.py -i {output_path_config};') # Pre_process, single, double, mult, 

                f.write(f'python3 {base_path_analysis_overall}/polarimetry/specMaker.py -i {output_path_config} -o {output_path_config} -l -r 0-{E*1.2};')

                f.write(f'cp {output_path_config}/parquet/masked/spectra/*.txt /local/home/jf285468/documents/phd/phemto/phemto_simulations/megalib/analysis_overall/rmf/data_rmf_spectra/;')

print('Run ./runParser.sh to run .tra.gz -> .t3pa parser, pre_process.py and specMaker.py')

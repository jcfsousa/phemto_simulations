# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 15:39:58 2022

@author: Lisa

Modified: 08 Jan 2026
Contributor: Jose Sousa
"""

import os, sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) != 4:
    print('Script usage:')
    print('make_ring_source.py geometry_file_path ring_energy psf_size')
    sys.exit()

geo_path = sys.argv[1] #Geometry file path
E = float(sys.argv[2])#keV #Tested energy 
psf_size = float(sys.argv[3]) #cm 

d_hkl = 3.2665 #A d for Germanium 111
hc = 12.4 #A*keV
focal = 2000.0 #cm
l_f = 3 #cm, focussing size of the crystal
l_nf = 1 #cm, non focussing size of the crystal
order = [1] #How many order of diffraction to use. Not implemented for now

#Evaluate Bragg angle
theta_b = np.arcsin(hc/(2*d_hkl*E)) 
#Evaluate ring radius
r = focal*np.tan(2*theta_b) 
#Evaluate number of crystals
n_of_crystals = int(2*np.pi*r/l_nf) 

#Evaluate (x, y, z) position of the source on the ring.
#Reference system: (0,0,0) is placed on the center of the top surface of the detector. Z positive going up
source_x, source_y, source_z = r*np.cos(np.linspace(0,2*np.pi,n_of_crystals)), r*np.sin(np.linspace(0,2*np.pi,n_of_crystals)), np.array([2000]*n_of_crystals)
#Focal spot coordinate
focal_spot_x, focal_spot_y, focal_spot_z = 0,0,0
#Evaluate the direction of the photons coming from the source
dir_x, dir_y, dir_z = focal_spot_x - source_x, focal_spot_y - source_y, focal_spot_z - source_z

#Plot the sources and the photon directions. #################################
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(source_x, source_y, source_z, label = 'Crystals', marker = '.')
ax.scatter(0,0,0, label = 'Focal Point')

for i in range(len(source_x)):
    ax.plot([source_x[i],dir_x[i]], [source_y[i],dir_y[i]], [source_z[i],dir_z[i]], color = (0,i/len(source_x)/2,i/len(source_x)))
ax.legend()
#####################################################################################

#Create the source file for megalib
source_file = open('{0}keV_ring.source'.format(E), 'w')
source_file.write('Geometry         {0}\n'.format(geo_path))
source_file.write('PhysicsListEM    LivermorePol\n')
source_file.write('\n')
source_file.write('StoreSimulationInfo       all\n')
source_file.write('\n')
source_file.write('Run LaueRing\n')
source_file.write('LaueRing.FileName              LaueRing_{0}keV\n'.format(E))
source_file.write('LaueRing.NEvents               1000000\n')
source_file.write('\n')

for i in range(n_of_crystals):
    source_file.write('\n')
    source_file.write('LaueRing.Source Xtal{0}\n'.format(i+1))
    source_file.write('Xtal{0}.ParticleType	1\n'.format(i+1))
    source_file.write('Xtal{0}.Beam	MapProfileBeam	{1} {2} {3} 90 {4} {5} {6} BeamProfile2D.dat\n'.format(i+1, source_x[i], source_y[i], source_z[i], dir_x[i], dir_y[i], dir_z[i]))
    source_file.write('Xtal{0}.Spectrum	Mono {1}\n'.format(i+1, E))
    source_file.write('Xtal{0}.Flux	1\n'.format(i+1))
    source_file.write('Xtal{0}.Polarization	Random\n'.format(i+1))

source_file.close()


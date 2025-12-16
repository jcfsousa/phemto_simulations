import numpy as np
import matplotlib.pyplot as plt

def compton_angle(E0, E1):
    me = 511 # electron mass in keV/c²
    c = 299792458 # speed of light in m/s
    cos_theta = 1 - (me)*(1/E1 - 1/E0)
    theta = np.arccos(cos_theta)
    degrees = np.degrees(theta)
    print("----------------------")
    print("E0: ", E0)
    print("E1: ", E1)
    print("degrees: ", degrees)
    print("----------------------")
    print("\n")
    return degrees
    

def compton_photon(E0, theta):
    me = 511 # electron mass in keV/c²
    c = 299792458 # speed of light in m/s
    theta = np.radians(theta)
    cos_theta = np.cos(theta)
    E1 = E0/(1 + (E0/me)*(1 - cos_theta))
    return E1

def angle_electron(E0,theta):
    me = 511 # electron mass in keV/c²
    c = 299792458 # speed of light in m/s
    theta = np.radians(theta)
    tan_alpha = (1/(1+(E0/me))*(1/np.tan(theta/2)))
    alpha = np.arctan(tan_alpha)
    degrees = np.degrees(alpha)
    return degrees

def get_energy_electron(E0, theta):
    me = 511 # electron mass in keV/c²
    c = 299792458 # speed of light in m/s
    theta = np.radians(theta)
    elec_energy = E0*(((E0/me)*(1-np.cos(theta))))/(1+(E0/me)*(1-np.cos(theta)))
    return elec_energy

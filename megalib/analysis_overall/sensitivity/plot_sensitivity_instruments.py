import csv
import numpy as np
import matplotlib.pyplot as plt

phemto_sensitivity_mirror_file = "PHEMTO_sensitivity_mirror.csv" #erg/cm2/s
phemto_sensitivity_lenses_file = "PHEMTO_sensitivity_lenses.csv" #erg/cm2/s
nustar_sensitivity_file = "NUSTAR_sensitivity.csv" #erg/cm2/s
isgri_sensitivity_file = "ISGRI_sensitivity.txt" #ph/cm2/s/keV, need to multiply by E**2/6.242e8


phemto_mirror_energies = []
phemto_mirror_sensitivity = []

with open(phemto_sensitivity_mirror_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        phemto_mirror_energies.append(float(row[0]))
        phemto_mirror_sensitivity.append(float(row[1]))

phemto_lenses_energies = []
phemto_lenses_sensitivity =[]

with open(phemto_sensitivity_lenses_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        phemto_lenses_energies.append(float(row[0]))
        phemto_lenses_sensitivity.append(float(row[1]))


nustar_energies = []
nustar_sensitivity = []
with open(nustar_sensitivity_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        nustar_energies.append(float(row[0]))
        nustar_sensitivity.append(float(row[1]))


isgri_energies = []
isgri_sensitivity = []

with open(isgri_sensitivity_file, "r", encoding='latin-1') as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        if line.startswith(":") or line.lower().startswith("data"):
            continue

        parts = line.split()

        for i in range(0, len(parts), 2):
            E = float(parts[i])
            S = float(parts[i + 1]) * ((E**2)/(6.242e8))
            isgri_energies.append(E)
            isgri_sensitivity.append(S)



print(phemto_lenses_energies)
print(phemto_mirror_energies)
for i, e in enumerate(phemto_mirror_energies):
    if round(e,0) == 50:
        s = phemto_mirror_sensitivity[i]

phemto_lenses_energies.insert(0,50)
phemto_lenses_sensitivity.insert(0,s)

with open("../../../results/megalib_v2/phemto_lenses_sensitivity_v2.csv", 'w') as f:
    f.write('Note:, File containing the PHEMTO sensitivity for the Laue Lenses. The energy is in [keV] and sensitivity in [erg cm^-2 s^-1]\n')
    f.write('version,2\n')
    f.write('energy,sensitivity\n')
    for e, s in zip(phemto_lenses_energies ,phemto_lenses_sensitivity):
        f.write(f'{e},{s}\n')

with open("../../../results/megalib_v2/phemto_mirror_sensitivity_v2.csv", 'w') as f:
    f.write('Note:, File containing the PHEMTO sensitivity for the x-ray Mirror. The energy is in [keV] and sensitivity in [erg cm^-2 s^-1]\n')
    f.write('version,2\n')
    f.write('energy,sensitivity\n')
    for e, s in zip(phemto_mirror_energies ,phemto_mirror_sensitivity):
        f.write(f'{e},{s}\n')



# CrabFlux
# arXiv:astro-ph/0406058v1 2 Jun 2004
# F(E) = k (E/1kev)^(-alpha) ph/cm2/s/keV

K = 14.44  # ph/cm²/s/keV
alpha = 2.169
e = np.arange(1.6,1.4e3,1)
f_crab = (K * (e)**(-alpha)) * ((e**2)/6.242e8) #erg/cm2/s

crab = [1e-4, 1e-5, 1e-6, 1e-7]
crab_legend = [r'100 $\mu$Crab', r'10 $\mu$Crab', r'1 $\mu$Crab', r'0.1 $\mu$Crab']

plt.figure(figsize=(6,5))
plt.plot(phemto_mirror_energies, phemto_mirror_sensitivity, c='red')
plt.plot(phemto_lenses_energies, phemto_lenses_sensitivity, c='green')
plt.plot(nustar_energies, nustar_sensitivity, c='blue', linestyle='--')
plt.plot(isgri_energies, isgri_sensitivity, c='k', linestyle='--')

for j,c in enumerate(crab):
    i = len(e)-950
    plt.plot(e, c*f_crab, c='k', linestyle = '--', alpha = 0.4)
    plt.text(e[i],c*f_crab[i]*1.2, crab_legend[j], c='k', alpha=0.5)

plt.text(0.22,0.1, 'PHEMTO mirror', color='red', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.66,0.22, 'PHEMTO lenses', color='green', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.3,0.36, 'NuSTAR', color='blue', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.3,0.8, 'INTEGRAL IBIS/ISGRI', color='k', fontsize=12, transform=plt.gca().transAxes)
plt.ylabel(r'Sensitivity [erg cm$^{-2}$s$^{-1}$]', fontsize=12)
plt.xlabel(r'Energy [keV]', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.4)

plt.tick_params(axis='both', which='major', labelsize=11)
plt.tick_params(axis='both', which='minor', labelsize=9)
plt.xlim(1.4,1.4e3)
plt.tight_layout()
plt.savefig('../../../results/megalib_v2/PHEMTO_sensitivity_v2.png', dpi=600)

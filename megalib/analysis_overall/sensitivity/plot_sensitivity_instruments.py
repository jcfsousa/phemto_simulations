import csv
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


plt.figure(figsize=(6,5))
plt.plot(phemto_mirror_energies, phemto_mirror_sensitivity, c='red', label='PHEMTO mirror')
plt.plot(phemto_lenses_energies, phemto_lenses_sensitivity, c='green', label='PHEMTO lens')
plt.plot(nustar_energies, nustar_sensitivity, c='blue', label='NuSTAR', linestyle='--')
plt.plot(isgri_energies, isgri_sensitivity, c='k', label='INTEGRAL: ISGRI', linestyle='--')

plt.text(0.22,0.1, 'PHEMTO mirror', color='red', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.7,0.25, 'PHEMTO lenses', color='green', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.27,0.35, 'NuSTAR', color='blue', fontsize=12, transform=plt.gca().transAxes)
plt.text(0.3,0.8, 'INTEGRAL IBIS/ISGRI', color='k', fontsize=12, transform=plt.gca().transAxes)
plt.ylabel(r'Sensitivity [erg cm$^{-2}$s$^{-1}$]', fontsize=12)
plt.xlabel(r'Energy [keV]', fontsize=12)
plt.xscale('log')
plt.yscale('log')
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.4)

plt.tick_params(axis='both', which='major', labelsize=11)
plt.tick_params(axis='both', which='minor', labelsize=9)
plt.tight_layout()
plt.show()

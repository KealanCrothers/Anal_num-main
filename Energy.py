import numpy as np
import matplotlib.pyplot as plt

# Le fichier doit contenir trois colonnes : t, E_pot, E_cin
data = np.loadtxt('energy_fork_1.0.txt')
t = data[:,0] * 1000  # Convertir les temps en milliseconde
Epot = data[:,1] * 1000 # Convertir l'énergie potentielle en millijoule
Ekin = data[:,2] * 1000 # Convertir l'énergie cinétique en millijoule
Etotal = Epot + Ekin

plt.figure()
plt.plot(t, Epot, label='Énergie potentielle')
plt.plot(t, Ekin, label='Énergie cinétique')
plt.plot(t, Etotal, label='Énergie totale')
plt.xlabel('Temps [ms]')
plt.ylabel('Énergie [mJ]')
plt.title("Énergies au cours du temps")
plt.legend()
plt.grid(True)
plt.show()
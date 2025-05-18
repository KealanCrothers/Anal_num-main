# modal_analysis.py
import numpy as np
from scipy.io import mmread
from scipy.sparse.linalg import eigsh
import argparse

def main():
    p = argparse.ArgumentParser(description="Analyse modale")
    p.add_argument("--K",       default="K.mtx",        help="Matrice de raideur MatrixMarket")
    p.add_argument("--M",       default="M.mtx",        help="Matrice de masse MatrixMarket")
    p.add_argument("--u0",      default="initial_fork_1.0.txt",  help="Cond. init. disp. (ux uy vx vy par ligne)")
    p.add_argument("--nev",     type=int, default=300,     help="Nombre de modes")

    Lref = 1.0  # m
    E = 2.1e11  # Pa
    rho = 7850  # kg/m3
    p.add_argument("--Lref",   type=float, default=Lref,  help="Longueur de référence")
    p.add_argument("--E",      type=float, default=E,     help="Module d'Young")
    p.add_argument("--rho",    type=float, default=rho,   help="Masse volumique")
    args = p.parse_args()

    # 1) charger K et M
    K = mmread(args.K).tocsr()
    M = mmread(args.M).tocsr()

    # 2) résoudre problème généralisé
    nev = args.nev
    eigvals, eigvecs = eigsh(K, k=nev, M=M, sigma=0.0, which='LM')
    omega_star = np.sqrt(eigvals)   # nondim.

    # 3) calcul de tau et conversion en Hz
    tau = args.Lref * np.sqrt(args.rho/args.E)
    omega = omega_star / tau        # rad/s réel
    freq = omega / (2*np.pi)        # Hz

    # 4) charger conditions initiales
    data = np.loadtxt(args.u0)      # Nx4 : ux uy vx vy
    N = data.shape[0]
    u0 = data[:,0:2].reshape(2*N)
    v0 = data[:,2:4].reshape(2*N)

    # 5) normalisation M-orthogonale
    for i in range(nev):
        phi = eigvecs[:,i]
        norm = np.sqrt(phi.dot(M.dot(phi)))
        eigvecs[:,i] /= norm

    # 6) coefficients modaux
    a = np.array([phi.dot(M.dot(u0)) for phi in eigvecs.T])
    b = np.array([phi.dot(M.dot(v0)) for phi in eigvecs.T])

    # 7) sauvegarde
    np.savetxt("frequencies.csv",
               np.vstack((np.arange(1,nev+1), freq)).T,
               header="mode,frequency_Hz", fmt="%d,%.6e")
    np.save("modes.npy", eigvecs)    # shape (2N, nev)
    np.savetxt("coeffs.csv",
               np.vstack((np.arange(1,nev+1), a, b)).T,
               header="mode,a_m,b_m", fmt="%d,%.6e,%.6e")

    print(" fichiers générés : frequencies.csv, modes.npy, coeffs.csv")


def fft():
    import numpy as np
    import matplotlib.pyplot as plt
    # 1) Charger time.txt : colonnes t*, ux*, uy*, vx*, vy*
    data = np.loadtxt('time_fork_1.0.txt')  # Nx4 : ux uy vx vy
    t = data[:,0]
    ux = data[:,1]  # par exemple ux du nœud choisi
    uy = data[:,2]

    Lref = 1.0  # m
    E = 2.1e11  # Pa
    rho = 7850  # kg/m3
    # 2) FFT
    N = len(t)
    dt = t[1]-t[0]
    Ufx = np.fft.rfft(ux)
    Ufy = np.fft.rfft(uy)
    freqs = np.fft.rfftfreq(N, dt)

    # 3) Tracé
    plt.figure()
    plt.semilogx(freqs, np.abs(Ufx), label='FFT ux')
    plt.semilogy(freqs, np.abs(Ufy), label='FFT uy')
    # superpose les fréquences propres non-dim (omega_star/(2π))
    omega_star = np.loadtxt('frequencies.csv', delimiter=',', skiprows=1)[:,1] * (Lref*np.sqrt(rho/E))
    f_modes = omega_star
    for f in f_modes[:10]:
        plt.axvline(f, color='r', linestyle='--')
    plt.xlabel('Fréquence adim. $f^*$')
    plt.ylabel('|FFT|')
    plt.title('Spectre du déplacement vs modes propres')
    plt.show()



def energy_modale():
    import numpy as np
    import matplotlib.pyplot as plt

    # constantes physiques
    Lref = 1.0      # m
    E    = 2.1e11   # Pa
    rho  = 7850     # kg/m3

    # 1) on charge les coefficients modaux
    # coeffs.csv : colonnes [mode, a_m, b_m]
    coeffs = np.loadtxt('coeffs.csv', delimiter=',', skiprows=1)
    a = coeffs[:,1]
    b = coeffs[:,2]

    # 2) on charge les fréquences propres non-dim (rad/s*)
    # frequencies.csv : colonnes [mode, f_Hz]
    f_Hz = np.loadtxt('frequencies.csv', delimiter=',', skiprows=1)[:,1]
    # on reconstruit omega* sachant que f_Hz = (omega*/tau)/(2*pi)
    tau = Lref * np.sqrt(rho/E)
    omega_star = 2*np.pi * f_Hz * tau   # rad/s*

    # 3) énergie modale adimensionnelle
    E_star = 0.5*(omega_star**2 * a**2 + b**2)

    # 4) passage en Joules : facteur E·Lref^3
    E_real = E_star * E * Lref**3 * 1000 # mJ

    # affichage
    for i, Em in enumerate(E_real, start=1):
        print(f"Mode {i:2d}: E = {Em:.3e} [mJ]")

    # tracé
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,len(E_real)+1), E_real, 'o-')
    plt.xlabel('Mode')
    plt.ylabel('Énergie modale [J]')
    plt.title('Énergie modale réelle vs mode')
    plt.grid(alpha=0.3)
    plt.show()
    
# reconstitution.py
import numpy as np
import matplotlib.pyplot as plt

def reconstitution(
    animation_dir="plot/animation",
    coeffs_file="coeffs.csv",
    freqs_file="frequencies.csv",
    total_frames=4000,
    Lref=1.0,    # m
    E=2.1e11,    # Pa
    rho=7850     # kg/m³
):
    """
    Compare la solution Newmark (fichiers animation_i.txt) et sa reconstitution modale
    à partir des 10 premiers modes, en traçant l’erreur ||U_num - U_mod|| non normalisée
    en fonction du temps adimensionnel t* = i * dt*.
    """
    # 1) charger a_m et b_m
    data_cb = np.loadtxt(coeffs_file, delimiter=",", skiprows=1)
    a = data_cb[:,1]
    b = data_cb[:,2]

    # charger les modes propres
    modes = np.load("modes.npy")  # shape (2N, nev)
    print(modes.shape)


    # 2) charger les fréquences propres adimensionnelles f*
    freqs_Hz = np.loadtxt(freqs_file, delimiter=",", skiprows=1)[:,1]
    tau       = Lref * np.sqrt(rho/E)
    omega_star = 2*np.pi * freqs_Hz * tau   # pulsations non-dim

    # 3) paramétrage Newmark (mêmes dt*, T* que dans ton C)
    #    si dt_C et T_C sont en adim., alors ici dt* = dt_C, T* = total_frames*dt_C
    #    Pour conserver ta logique, on prend dt* = 0.03 (comme dans ton exemple)
    dt_star = 0.03
    times   = np.arange(total_frames) * dt_star

    errs = np.zeros(total_frames)

    # 4) boucle sur chaque frame i
    for i in range(total_frames):
        print(f"Frame {i+1}/{total_frames}", end="\r")
        fname = f"{animation_dir}/animation_{i}.txt"
        # Lecture du déplacement Newmark (skip header "Size ...")
        data = np.loadtxt(fname, skiprows=1, delimiter=",")
        U_num = data[:,:2].reshape(-1)  # vecteur (2N,)

        # Reconstruction modale avec les 10 premiers modes
        U_mod = np.zeros_like(U_num)
        for m in range(4):
            U_mod += (a[m]*np.cos(omega_star[m]*i*dt_star) + (b[m]/omega_star[m])*np.sin(omega_star[m]*i*dt_star)) * modes[:,m]

        # erreur Euclidienne globale (non normalisée)
        errs[i] = np.linalg.norm(U_num - U_mod)

    # 5) tracé
    plt.figure(figsize=(6,3))
    plt.plot(times, errs, "-", lw=1)
    plt.xlabel(r"Temps adim. $t^*$")
    plt.ylabel(r"Erreur $\|U_{num}-U_{mod}\|$")
    plt.title("Erreur de reconstitution modale (10 modes)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()



if __name__=="__main__":
    main()
    fft()
    energy_modale()
    reconstitution()

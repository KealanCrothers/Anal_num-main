#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from numpy.typing import NDArray
import os

# ───── USER PARAMETERS ────────────────────────────────────────────────────────
mesh_file     = "plot/fork.txt"
modes_file    = "modes.npy"
coeffs_file   = "coeffs.csv"
freqs_file    = "frequencies.csv"
anim_dir      = "plot/animation/"  # folder with animation_0.txt etc.

speedup       = 15             # speedup factor for true solution
mode_indices  = [0, 1, 2, 3]
tau           = 1.0 * np.sqrt(7.85e+03 / 2.10e+11)  # time constant [s]
T             = 120 * tau       # total time [s]
dt            = 0.03 * tau * speedup    # time step between frames [s]
deform_fac    = 3e2
gif_out       = "plot/modes_vs_true.gif"
progress_every = 50
# ───── END USER PARAMETERS ────────────────────────────────────────────────────

# ───── MESH ───────────────────────────────────────────────────────────────────
class Mesh:
    def __init__(self, fname):
        with open(fname, "r") as f:
            self.nnodes = int(f.readline().split()[3])
            self.nodes = np.zeros((self.nnodes, 2))
            for i in range(self.nnodes):
                _, coords = f.readline().split(":")
                self.nodes[i] = [float(x) for x in coords.split()]
            line = f.readline()
            while "triangles" not in line and "quads" not in line:
                line = f.readline()
            self.nlocal = 3 if "triangles" in line else 4
            self.nelem = int(line.split()[3])
            self.elem = np.zeros((self.nelem, self.nlocal), int)
            for i in range(self.nelem):
                _, rest = f.readline().split(":")
                self.elem[i] = [int(x) for x in rest.split()]

    def plot(self, displace=None, ax=None, **kwargs):
        if ax is None: ax = plt.gca()
        pts = self.nodes + (displace if displace is not None else 0)
        if self.nlocal == 3:
            ax.triplot(pts[:, 0], pts[:, 1], self.elem, **kwargs)
        else:
            raise NotImplementedError

    def plotfield(self, field, displace=None, ax=None, **kwargs):
        if ax is None: ax = plt.gca()
        pts = self.nodes + (displace if displace is not None else 0)
        if len(field) != self.nnodes:
            raise ValueError("Field length mismatch")
        if self.nlocal == 3:
            return ax.tripcolor(pts[:, 0], pts[:, 1], self.elem, field,
                                shading="gouraud", **kwargs)
        else:
            raise NotImplementedError

# ───── LOAD MODES, FREQS, COEFFS ──────────────────────────────────────────────
mesh     = Mesh(mesh_file)
phi_all  = np.load(modes_file)
coeffs   = np.loadtxt(coeffs_file, delimiter=",", skiprows=1)
freqs    = np.loadtxt(freqs_file, delimiter=",", skiprows=1)

N        = mesh.nnodes
times    = np.arange(0, T + 1e-12, dt)
n_steps  = len(times)

# ───── PLOT SETUP (2x3 GRID) ──────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 12))
gs = fig.add_gridspec(3, 2)
axes = [fig.add_subplot(gs[i//2, i%2]) for i in range(4)]
ax_sum  = fig.add_subplot(gs[2, 0])
ax_true = fig.add_subplot(gs[2, 1])

# ───── PREPARE MODAL INFO ─────────────────────────────────────────────────────
modal_info = []
for idx in mode_indices:
    phi = phi_all[:, idx]
    freq = freqs[idx, 1]
    omega = 2 * np.pi * freq
    a_m, b_m = coeffs[idx, 1], coeffs[idx, 2]
    modal_info.append((phi, omega, a_m, b_m, freq))

# ───── UPDATE FUNCTION ────────────────────────────────────────────────────────
def update(i):
    t = times[i]
    if i % progress_every == 0:
        print(f"Rendering frame {i+1}/{n_steps}")

    u_sum = np.zeros(2 * N)
    for ax in axes: ax.cla()
    ax_sum.cla()
    ax_true.cla()

    for k, (phi, omega, a_m, b_m, freq) in enumerate(modal_info):
        u_k = deform_fac * (a_m * np.cos(omega * t) + (b_m / omega) * np.sin(omega * t)) * phi
        u_sum += u_k
        u_disp = u_k.reshape(N, 2)
        norm = np.linalg.norm(u_disp, axis=1)
        mesh.plotfield(norm, displace=u_disp, ax=axes[k], cmap="turbo")
        mesh.plot(displace=u_disp, ax=axes[k], color="k", lw=0.2)
        axes[k].set_title(f"Mode {mode_indices[k]+1} ({freq:.1f} Hz)")
        axes[k].set_xlim(-0.1, 0.1)
        axes[k].set_ylim(-0.08, 0.10)
        axes[k].set_aspect("equal")
        axes[k].grid(alpha=0.3)

    # SUM OF MODES
    u_disp = u_sum.reshape(N, 2)
    norm = np.linalg.norm(u_disp, axis=1)
    mesh.plotfield(norm, displace=u_disp, ax=ax_sum, cmap="turbo")
    mesh.plot(displace=u_disp, ax=ax_sum, color="k", lw=0.2)
    ax_sum.set_title("Sum of modes")
    ax_sum.set_xlim(-0.1, 0.1)
    ax_sum.set_ylim(-0.08, 0.10)
    ax_sum.set_aspect("equal")
    ax_sum.grid(alpha=0.3)
    ax_sum.set_xlabel(f"Time {t*1000:.2f} [ms]")

    # TRUE NUMERICAL FRAME (same t, but needs i * dt / true_dt)
    true_frame = int(round(t / dt * speedup))
    fname = os.path.join(anim_dir, f"animation_{true_frame}.txt")
    try:
        data = np.loadtxt(fname, skiprows=1, delimiter=",")
        u_true = deform_fac * data
        norm_true = np.linalg.norm(u_true, axis=1)
        mesh.plotfield(norm_true, displace=u_true, ax=ax_true, cmap="turbo")
        mesh.plot(displace=u_true, ax=ax_true, color="k", lw=0.2)
        ax_true.set_title("Newmark solution")
        ax_true.set_xlim(-0.1, 0.1)
        ax_true.set_ylim(-0.08, 0.10)
        ax_true.set_aspect("equal")
        ax_true.grid(alpha=0.3)
        ax_true.set_xlabel(f"Time {t*1000:.2f} [ms]")
    except Exception as e:
        print(f"Error loading frame {true_frame}: {e}")

# ───── RUN ANIMATION ─────────────────────────────────────────────────────────
ani = FuncAnimation(fig, update, frames=n_steps, interval=200, blit=False)
ani.save(gif_out, writer=PillowWriter(fps=50))
plt.close(fig)

print(f"Saved → {gif_out}")

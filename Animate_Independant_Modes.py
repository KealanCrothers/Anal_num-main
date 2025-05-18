#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ───── USER SETTINGS ──────────────────────────────────────────────────────────
mesh_file     = "plot/fork.txt"
modes_file    = "modes.npy"
freqs_file    = "frequencies.csv"
coeffs_file   = "coeffs.csv"

mode_indices  = [0, 1, 2, 3, 4, 5]  # Show these modes
T             = 0.008              # total animation duration [s]
dt            = 0.00001 * 8        # timestep per frame [s]
deform_fac    = 5e-3               # max ‖u‖ for every mode
gif_out       = "plot/animated_modes.gif"
progress_every = 50
# ───── END USER SETTINGS ──────────────────────────────────────────────────────

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

    def plotfield(self, field, displace=None, ax=None, **kwargs):
        if ax is None: ax = plt.gca()
        pts = self.nodes + (displace if displace is not None else 0)
        if len(field) != self.nnodes:
            raise ValueError("Field length mismatch")
        return ax.tripcolor(pts[:, 0], pts[:, 1], self.elem, field, shading="gouraud", **kwargs)

# ───── LOAD MODAL DATA ────────────────────────────────────────────────────────
mesh     = Mesh(mesh_file)
phi_all  = np.load(modes_file)               # shape (2N, nev)
freqs    = np.loadtxt(freqs_file, delimiter=",", skiprows=1)   # [mode, Hz]
coeffs   = np.loadtxt(coeffs_file, delimiter=",", skiprows=1)  # [mode, a, b]
N        = mesh.nnodes
times    = np.arange(0, T+1e-12, dt)
n_steps  = len(times)

# ───── SETUP FIGURE ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

modal_info = []
for idx in mode_indices:
    phi    = phi_all[:, idx]
    freq   = freqs[idx, 1]
    omega  = 2 * np.pi * freq
    a_m, b_m = coeffs[idx, 1], coeffs[idx, 2]
    u_static = a_m * phi  # t = 0
    u_reshaped = u_static.reshape(N, 2)
    max_norm = np.max(np.linalg.norm(u_reshaped, axis=1))
    scale = deform_fac / max_norm if max_norm > 0 else 1.0
    modal_info.append((phi, omega, a_m, b_m, freq, scale))

def update(i):
    t = times[i]
    if i % progress_every == 0:
        print(f"Frame {i+1}/{n_steps}")
    for ax in axes:
        ax.cla()

    for k, (phi, omega, a_m, b_m, freq, scale) in enumerate(modal_info):
        u_k = (a_m * np.cos(omega * t) + (b_m / omega) * np.sin(omega * t)) * phi
        u_disp = (u_k * scale).reshape(N, 2)
        norm = np.linalg.norm(u_disp, axis=1)
        mesh.plotfield(norm, displace=u_disp, ax=axes[k], cmap="turbo")
        mesh.plot(displace=u_disp, ax=axes[k], color="k", lw=0.3)
        axes[k].set_title(f"Mode {mode_indices[k]+1} ({freq:.1f} Hz)")
        axes[k].set_xlim(-0.06, 0.06)
        axes[k].set_ylim(-0.08, 0.10)
        axes[k].set_aspect("equal")
        axes[k].grid(alpha=0.3)

# ───── ANIMATE AND SAVE ───────────────────────────────────────────────────────
ani = FuncAnimation(fig, update, frames=n_steps, interval=100, blit=False)
ani.save(gif_out, writer=PillowWriter(fps=int(1/dt)))
plt.close(fig)
print(f"Saved → {gif_out}")

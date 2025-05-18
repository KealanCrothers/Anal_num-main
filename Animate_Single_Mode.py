#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ───── USER SETTINGS ──────────────────────────────────────────────────────────
mesh_file    = "plot/fork.txt"
modes_file   = "modes.npy"
freqs_file   = "frequencies.csv"
coeffs_file  = "coeffs.csv"

mode_idx     = 0       # 0-based index: fundamental = 0

import sys
# ───── READ ARGUMENTS ─────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <mode_idx>")
    print("       where <mode_idx> = 0 for fundamental mode, 1 for 2nd, etc.")
    sys.exit(1)

try:
    mode_idx = int(sys.argv[1])
except ValueError:
    print("Invalid mode index (must be integer)")
    sys.exit(1)


deform_fac   = 0.6       # same as total‐deformation GIF
gif_out      = "mode1.gif"

x_limits     = (-0.03, 0.03)
y_limits     = (-0.08, 0.10)
progress_every = 50      # print a message every this many frames
# ───── end USER SETTINGS ──────────────────────────────────────────────────────

# Load modal data
phi     = np.load(modes_file)[:, mode_idx]  # (2N,)
freqs   = np.loadtxt(freqs_file, delimiter=",", skiprows=1)
freq_hz = freqs[mode_idx,1]
omega   = 2*np.pi*freq_hz
coeffs  = np.loadtxt(coeffs_file, delimiter=",", skiprows=1)
a_m, b_m = coeffs[mode_idx,1], coeffs[mode_idx,2]

#normalize coefficients
a_m = 0.1

#normalize phi
phi = phi/np.linalg.norm(phi)

T            = 0.005   # total time to animate [s]
dt           = 0.00001*8  # time step between frames [s]


T = 1/freq_hz
dt = T/50

print(f"Animating mode {mode_idx+1} at {freq_hz:.6f} Hz for T={T}s, dt={dt}s")

# Build time array and frames
times   = np.arange(0, T+1e-12, dt)
n_steps = len(times)
print(f"→ {n_steps} frames")

# Mesh helper
class Mesh:
    def __init__(self, fname):
        with open(fname,"r") as f:
            self.nnodes = int(f.readline().split()[3])
            self.nodes = np.zeros((self.nnodes,2))
            for i in range(self.nnodes):
                _, coords = f.readline().split(":")
                self.nodes[i] = [float(x) for x in coords.split()]
            line=f.readline()
            while "triangles" not in line and "quads" not in line:
                line=f.readline()
            self.nlocal = 3 if "triangles" in line else 4
            self.nelem  = int(line.split()[3])
            self.elem   = np.zeros((self.nelem,self.nlocal),int)
            for i in range(self.nelem):
                _, rest = f.readline().split(":")
                self.elem[i] = [int(x) for x in rest.split()]

    def plot(self, displace=None, ax=None, **kwargs):
        if ax is None: ax=plt.gca()
        pts = self.nodes + (displace if displace is not None else 0)
        if self.nlocal==3:
            ax.triplot(pts[:,0], pts[:,1], self.elem, **kwargs)
        else:
            for tri in self.elem:
                x = np.append(pts[tri,0], pts[tri[0],0])
                y = np.append(pts[tri,1], pts[tri[0],1])
                ax.plot(x,y,**kwargs)

    def plotfield(self, field, displace=None, ax=None, **kwargs):
        if ax is None: ax=plt.gca()
        pts = self.nodes + (displace if displace is not None else 0)
        if len(field)!=self.nnodes:
            raise ValueError("Field length mismatch")
        if self.nlocal==3:
            return ax.tripcolor(pts[:,0], pts[:,1], self.elem, field,
                                shading="gouraud", **kwargs)
        else:
            raise NotImplementedError


mesh = Mesh(mesh_file)

# Set up figure
fig, ax = plt.subplots(figsize=(4,6))

def update(i):
    if i % progress_every == 0:
        print(f"  rendering frame {i+1}/{n_steps}")
    t = times[i]
    u2n = deform_fac*(a_m*np.cos(omega*t) + (b_m/omega)*np.sin(omega*t))*phi
    u    = u2n.reshape(mesh.nnodes,2)
    field= np.linalg.norm(u,axis=1)
    ax.cla()
    mesh.plot(displace=u, ax=ax, color="steelblue", lw=0.8)
    mesh.plotfield(field, displace=u, ax=ax, cmap="turbo")
    ax.set_title(f"Mode {mode_idx+1}, t={t*1e3:5.1f} ms")
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

ani = FuncAnimation(fig, update, frames=n_steps, interval=200, blit=False)
ani.save(gif_out, writer=PillowWriter(fps=50))
plt.close(fig)

print(f"Saved → {gif_out}")

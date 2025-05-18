import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class Mesh:
    nlocal: int
    nnodes: int
    nelem: int
    nodes: NDArray[np.float64]
    elem: NDArray[np.int32]

    def __init__(self, fname):
        self.fname = fname
        with open(fname, "r") as f:
            self.nnodes = int(f.readline().split(" ")[3])
            self.nodes = np.zeros((self.nnodes, 2))
            for i in range(self.nnodes):
                line = f.readline()
                parts = [s.strip() for s in line.split(':')]
                self.nodes[i] = [float(val.strip()) for val in parts[1].split()]
            line = f.readline()
            while ("triangles" not in line and "quads" not in line):
                line = f.readline()
            self.nlocal = 3 if "triangles" in line else 4
            self.nelem = int(line.split(" ")[3])
            self.elem = np.zeros((self.nelem, self.nlocal), dtype=np.int32)
            for i in range(self.nelem):
                line = f.readline()
                parts = [s.strip() for s in line.split(':')]
                self.elem[i] = [int(val.strip()) for val in parts[1].split()]

    def __str__(self):
        return (f"Mesh: {self.fname}\n"
                f"├─nnodes: {self.nnodes}\n"
                f"├─nelem: {self.nelem}\n"
                f"├─local: {self.nlocal}\n"
                f"├─nodes: array(shape=({self.nodes.shape[0]},{self.nodes.shape[1]}), dtype=np.float64)\n"
                f"└─elem: array(shape=({self.elem.shape[0]},{self.elem.shape[1]}), dtype=np.int32)\n")

    def __repr__(self) -> str:
        return self.__str__()

    def plot(self, displace=None, ax=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        coord = self.nodes if displace is None else np.array(displace) + self.nodes
        if self.nlocal == 3:
            ax.triplot(coord[:, 0], coord[:, 1], self.elem, *args, **kwargs)
        else:
            for e in self.elem:
                x = np.append(coord[e, 0], coord[e[0], 0])
                y = np.append(coord[e, 1], coord[e[0], 1])
                ax.plot(x, y, *args, **kwargs)

    def plotfield(self, field, displace=None, ax=None, *args, **kwargs):
        if ax is None:
            ax = plt.gca()
        coord = self.nodes if displace is None else np.array(displace) + self.nodes
        field = np.array(field)
        if len(field) != self.nnodes:
            raise ValueError("Field must be a 1D array with length equal to number of nodes")
        if self.nlocal == 3:
            mappable = ax.tripcolor(coord[:, 0], coord[:, 1], self.elem, field,
                                    shading="gouraud", *args, **kwargs)
            return mappable
        else:
            raise NotImplementedError("plotfield for quad meshes is not implemented in this example.")

def animate_deformation_to_gif(mesh, nframes=49, factor=5e4, animation_dir="./data/animation/",
                               gif_filename="animation.gif", xlim=(-1, 1), ylim=(-1, 1)):
    """
    Create an animated GIF of the deformed mesh with fixed axis limits.
    
    Parameters:
      - mesh: an instance of Mesh.
      - nframes: total number of frames.
      - factor: scaling factor for the displacement.
      - animation_dir: directory containing displacement files (UV1.txt, UV2.txt, ...).
      - gif_filename: output GIF filename.
      - xlim: tuple (xmin, xmax) to fix the x-axis.
      - ylim: tuple (ymin, ymax) to fix the y-axis.
    """
    fig, ax = plt.subplots()
    norig = 200
    def update(frame):
        if frame % 50 == 0:
            print(f"Processing frame {frame+1}/{nframes}")
        ax.cla()  # Clear axis for new frame.
        fname = f"{animation_dir}animation_{frame}.txt"
        try:
            uv = np.loadtxt(fname, skiprows=1, delimiter=",")
        except Exception as e:
            print(f"Error loading {fname}: {e}")
            return
        uv_norm = np.linalg.norm(uv, axis=1)
        mesh.plotfield(uv_norm, displace=uv*factor, ax=ax, cmap="turbo")
        mesh.plot(displace=uv*factor, ax=ax, lw=0.2, c="k")
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)
        t=0.01*frame
        ax.set_xlabel(f"Time {t:.2f} [ms]")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        return ax,


    # We take 1 frame every 8 frames to reduce the number of frames in the GIF.
    real_frames = np.arange(0, nframes, 15)
    ani = FuncAnimation(fig, update, frames=real_frames, interval=200, blit=False)
    
    writer = PillowWriter(fps=50)
    ani.save(gif_filename, writer=writer)
    plt.close(fig)
    print(f"Animation saved as {gif_filename}")

if __name__ == "__main__":
    mesh = Mesh("plot/fork.txt")
    print(mesh)
    
    # Set your fixed axis limits.
    fixed_xlim = (-0.1, 0.1)
    fixed_ylim = (-0.08, 0.1)
    
    animate_deformation_to_gif(mesh, nframes=4000, factor=1e2, animation_dir="./plot/animation/",
                               gif_filename="./plot/Solution.gif", xlim=fixed_xlim, ylim=fixed_ylim)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting
import os
import struct

matplotlib.use('TkAgg')


# === Load .hgt File ===
def load_hgt(filename):
    with open(filename, 'rb') as file:
        data = file.read()

    # Determine resolution from file size
    size = int(np.sqrt(len(data) / 2))  # each height value is 2 bytes
    fmt = '>' + 'h' * (size * size)  # big-endian signed short
    elevations = struct.unpack(fmt, data)
    elevation_array = np.array(elevations, dtype=np.int16).reshape((size, size))

    # Flip vertically so north is at the top
    return np.flipud(elevation_array)


# === Main plotting function ===
def plot_3d_terrain(elevation_data):
    size = elevation_data.shape[0]
    x = np.linspace(0, size - 1, size) * 30
    y = np.linspace(0, size - 1, size) * 30
    X, Y = np.meshgrid(x, y)
    Z = elevation_data

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface with color mapping
    surf = ax.plot_surface(X, Y, Z, cmap='terrain', linewidth=0, antialiased=False)

    # Add colorbar for elevation
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')

    ax.set_title('3D Terrain from .hgt File')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation (m)')

    plt.tight_layout()
    plt.show()


# === Run with your file ===
filename = 'N45E024_3.hgt'  # replace with your actual .hgt filename
if os.path.exists(filename):
    elevation_data = load_hgt(filename)
    plot_3d_terrain(elevation_data)
else:
    print(f"File not found: {filename}")

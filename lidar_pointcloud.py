#! /usr/bin/env python3.12

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import os
from matplotlib.colors import LinearSegmentedColormap

def combine_lidar_frames():
    # Find all LIDAR files in the directory
    lidar_files = sorted(glob.glob('training/dataset/lidar/*.csv'))

    if len(lidar_files) < 10:
        print("Need at least 10 LIDAR files for good 360° visualization")
        return

    # Load all files
    all_points = []
    for i, lidar_file in enumerate(lidar_files[:100]):  # Use first 10 files
        try:
            data = np.loadtxt(lidar_file, delimiter=',')
            if data.size > 0 and data.shape[1] >= 3:
                all_points.append(data)
        except Exception as e:
            print(f"Skipping {lidar_file}: {e}")

    if not all_points:
        print("No valid LIDAR data found")
        return

    # Combine all points
    combined_points = np.vstack(all_points)

    # Filter out extreme points (noise)
    distances = np.sqrt(combined_points[:, 0]**2 + combined_points[:, 1]**2 + combined_points[:, 2]**2)
    mask = (distances > 0.1) & (distances < 100)  # Keep points between 0.1m and 100m
    combined_points = combined_points[mask]

    # Create visualization
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d', facecolor='black')

    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list('lidar_cmap', ['blue', 'cyan', 'yellow', 'red'])

    # Plot all points with color based on distance
    distances = np.sqrt(combined_points[:, 0]**2 + combined_points[:, 1]**2 + combined_points[:, 2]**2)
    scatter = ax.scatter(
        combined_points[:, 0],
        combined_points[:, 1],
        combined_points[:, 2],
        c=distances,
        cmap=cmap,
        s=2,  # Smaller points for better visibility
        alpha=0.7,
        edgecolors=None  # No edges for cleaner look
    )

    # Add vehicle position
    ax.scatter([0], [0], [0.5], color='white', s=200, marker='^', label='Vehicle')

    # Add ground plane
    max_range = 50
    ground = np.array([[max_range, max_range, 0],
                      [max_range, -max_range, 0],
                      [-max_range, -max_range, 0],
                      [-max_range, max_range, 0]])
    ax.plot_trisurf(ground[:, 0], ground[:, 1], ground[:, 2], color='gray', alpha=0.1)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Distance from vehicle (m)', rotation=270, labelpad=15)

    # Set labels and title
    ax.set_title('Combined LIDAR Point Cloud - 360° View', fontsize=16, color='white', pad=20)
    ax.set_xlabel('X (m)', fontsize=12, color='white')
    ax.set_ylabel('Y (m)', fontsize=12, color='white')
    ax.set_zlabel('Z (m)', fontsize=12, color='white')

    # Adjust view
    ax.view_init(elev=33, azim=10)

    # Set equal aspect ratio
    max_val = 50
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(0, max_val)

    # Style adjustments
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.zaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    ax.grid(False)

    # Add context
    ax.text2D(0.02, 0.95, f"Combined from {len(all_points)} frames",
              transform=ax.transAxes, color='white', fontsize=10)
    ax.text2D(0.02, 0.92, f"{len(combined_points)} points total",
              transform=ax.transAxes, color='white', fontsize=10)

    plt.tight_layout()

    # Save and show
    output_file = 'combined_lidar_360.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='black')
    print(f"Combined 360° visualization saved as {output_file}")
    plt.show()

if __name__ == "__main__":
    combine_lidar_frames()

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# Load the point cloud
pcd = o3d.io.read_point_cloud("/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/gaussian-splatting/output/5d0f2b29-4/point_cloud/iteration_7000/point_cloud.ply")

# Extract points as a numpy array
points = np.asarray(pcd.points)

# If the point cloud has colors, extract them; otherwise, use Z-coordinates for grayscale
if pcd.has_colors():
    colors = np.asarray(pcd.colors)
else:
    # Normalize Z values to use as grayscale
    z_values = points[:, 2]
    grayscale = (z_values - z_values.min()) / (z_values.max() - z_values.min())
    colors = np.stack([grayscale, grayscale, grayscale], axis=-1)

# Plot the point cloud as a 2D projection (e.g., onto the XY plane)
plt.figure(figsize=(10, 10))
plt.scatter(points[:, 0], points[:, 1], c=colors, s=1, linewidths=0, marker='o')

# Remove axis for better visualization
plt.axis('off')

# Save the result as an image
plt.savefig('point_cloud_projection.png', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()

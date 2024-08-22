import open3d as o3d
import numpy as np

# Load your point cloud
pcd = o3d.io.read_point_cloud("/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/gaussian-splatting/output/5d0f2b29-4/point_cloud/iteration_7000/point_cloud.ply")

# If the point cloud has no colors or grayscale values need normalization
if not pcd.has_colors():
    print("No color information found. Assigning grayscale colors based on Z-coordinate.")
    points = np.asarray(pcd.points)
    # Normalize the Z-values to create a grayscale effect
    grayscale = (points[:, 2] - points[:, 2].min()) / (points[:, 2].max() - points[:, 2].min())
    colors = np.stack([grayscale, grayscale, grayscale], axis=-1)
    pcd.colors = o3d.utility.Vector3dVector(colors)

elif np.asarray(pcd.colors).max() > 1.0:
    # Normalize existing colors to the [0, 1] range
    print("Normalizing existing color values.")
    colors = np.asarray(pcd.colors)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])

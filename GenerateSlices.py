import numpy as np
import open3d as o3d

# Load your point cloud
pcd = o3d.io.read_point_cloud("/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/gaussian-splatting/output/5d0f2b29-4/point_cloud/iteration_7000/point_cloud.ply")

# Visualize the original point cloud
o3d.visualization.draw_geometries([pcd])

# Perform Poisson surface reconstruction to create a dense mesh
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)

# Check if the mesh has triangles
if len(mesh.triangles) == 0:
    print("Mesh has no triangles, check the input point cloud and adjust parameters.")
else:
    # Optionally remove low-density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    # Sample points from the dense mesh
    dense_pcd = mesh.sample_points_poisson_disk(number_of_points=100000)

    # Transfer the original colors to the dense point cloud
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    dense_colors = []
    for point in dense_pcd.points:
        [_, idx, _] = pcd_tree.search_knn_vector_3d(point, 1)
        dense_colors.append(pcd.colors[idx[0]])

    # Set the colors for the dense point cloud
    dense_pcd.colors = o3d.utility.Vector3dVector(np.array(dense_colors))

    # Save and visualize the dense point cloud with colors
    o3d.io.write_point_cloud("dense_point_cloud_colored.ply", dense_pcd)
    o3d.visualization.draw_geometries([dense_pcd])

    # Define the Z value for the slice and a tolerance
    z_value = 0.5
    tolerance = 0.01

    # Extract points that lie within the desired Z slice
    slice_points = np.asarray(dense_pcd.points)
    slice_points = slice_points[(slice_points[:, 2] > z_value - tolerance) & (slice_points[:, 2] < z_value + tolerance)]

    # Extract the corresponding colors
    slice_colors = np.asarray(dense_pcd.colors)
    slice_colors = slice_colors[(slice_points[:, 2] > z_value - tolerance) & (slice_points[:, 2] < z_value + tolerance)]

    # Convert the slice to a point cloud for visualization
    slice_pcd = o3d.geometry.PointCloud()
    slice_pcd.points = o3d.utility.Vector3dVector(slice_points)
    slice_pcd.colors = o3d.utility.Vector3dVector(slice_colors)

    # Save the slice as a point cloud
    o3d.io.write_point_cloud("slice_image.ply", slice_pcd)

    # Optionally, visualize the slice
    o3d.visualization.draw_geometries([slice_pcd])

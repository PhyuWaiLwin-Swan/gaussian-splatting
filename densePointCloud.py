import numpy as np
import open3d as o3d

# Load your point cloud
pcd = o3d.io.read_point_cloud("/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/gaussian-splatting/output/5d0f2b29-4/point_cloud/iteration_7000/point_cloud.ply")

# Visualize the original point cloud
o3d.visualization.draw_geometries([pcd])

# Perform Poisson surface reconstruction
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
    # Find the nearest neighbors in the original point cloud
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

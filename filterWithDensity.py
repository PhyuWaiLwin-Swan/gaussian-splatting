import numpy as np
from plyfile import PlyData, PlyElement

# Function to crop the Gaussian splatting point cloud by bounding box and max density threshold
def crop_gaussian_splatting(vert, min_bound, max_bound, density_field=None, max_density=None):
    """Crops the Gaussian splatting data while preserving additional fields and applying a max density threshold."""

    # Extract points (x, y, z)
    points = np.vstack([vert['x'], vert['y'], vert['z']]).T

    # Create a mask for points within the bounding box
    mask = (
            (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) &
            (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) &
            (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
    )

    # If max density filtering is requested, add it to the mask

    color_mask = ~((vert['f_dc_0'] == 1.0) & (vert['f_dc_1'] == 1.0) & (
                vert['f_dc_2'] == 1.0))  # Assuming white is (1.0, 1.0, 1.0)
    mask = mask & color_mask

        # Apply the mask to keep only points inside the bounding box and non-white points
    return vert[mask]

# Main processing function
def process_gaussian_splatting(ply_file_path, output_ply_path, min_bound, max_bound, density_field=None, max_density=None):
    """Processes a PLY file by cropping and saving it while preserving Gaussian splatting custom fields."""

    # Read the PLY file
    plydata = PlyData.read(ply_file_path)
    vert = plydata['vertex']

    # Crop the point cloud based on the bounding box and max density
    cropped_vert = crop_gaussian_splatting(vert, min_bound, max_bound, density_field, max_density)
    print(f"Cropped to {len(cropped_vert)} points.")

    # Save the cropped point cloud with all Gaussian splatting fields
    save_gaussian_splatting(output_ply_path, cropped_vert)
def save_gaussian_splatting(output_path, vertices):
    """Save a PLY file with custom fields for Gaussian splatting data (e.g., scale, rotation, opacity)."""

    # Get the available fields in the vertices
    fields = vertices.dtype.names

    # Create the PLY element with all fields preserved
    vertex_array = np.array([tuple(v[field] for field in fields) for v in vertices], dtype=vertices.dtype)
    ply_element = PlyElement.describe(vertex_array, 'vertex')

    # Write to the PLY file
    PlyData([ply_element]).write(output_path)
    print(f"Saved cropped Gaussian splatting data to: {output_path}")

# Example usage
if __name__ == "__main__":
    # Define the input PLY file path and output PLY file path
    input_ply = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/gaussian-splatting/output/d0f77868-6/point_cloud/iteration_7000/point_cloud.ply"
    output_ply = "cropped_gaussian_splatting_data2.ply"


    def save_gaussian_splatting(output_path, vertices):
        """Save a PLY file with custom fields for Gaussian splatting data (e.g., scale, rotation, opacity)."""

        # Get the available fields in the vertices
        fields = vertices.dtype.names

        # Create the PLY element with all fields preserved
        vertex_array = np.array([tuple(v[field] for field in fields) for v in vertices], dtype=vertices.dtype)
        ply_element = PlyElement.describe(vertex_array, 'vertex')

        # Write to the PLY file
        PlyData([ply_element]).write(output_path)
        print(f"Saved cropped Gaussian splatting data to: {output_path}")


    # Define cropping bounds (bounding box)
    min_bound = [-20.0, -20, -18.0]  # Adjust the bounds as necessary
    max_bound = [20.0, 20, 20.0]

    # Define max density filtering parameters (optional)
    density_field = 'opacity'  # Adjust this field according to your PLY file
    max_density = 2 # Adjust this to remove only the densest points (those exceeding the threshold)

    # Process the Gaussian splatting data by cropping and saving it
    process_gaussian_splatting(input_ply, output_ply, min_bound, max_bound, density_field, max_density)

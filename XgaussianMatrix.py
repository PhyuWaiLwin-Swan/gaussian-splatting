import math
import os
import json
import struct
import pydicom
import cv2
import numpy as np
from typing import NamedTuple
from PIL import Image
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


def calculate_rotation_matrix_with_translation(translation_vector):
    """
    Calculate a rotation matrix to align the local coordinate system (camera) with the world coordinate system.
    The translation vector represents the direction the camera is facing.
    World Coordinate System:
    - X-axis: Downward
    - Y-axis: Right
    - Z-axis: Forward
    """
    forward = np.array([translation_vector[0], translation_vector[1], 0])
    forward = -forward / np.linalg.norm(forward)
    world_up = np.array([0, 0, 1])
    right = np.cross(world_up, forward)
    right = right / np.linalg.norm(right)
    downward = np.cross(forward, right)
    downward = -downward / np.linalg.norm(downward)
    rotation_matrix = np.column_stack((downward, right, forward))
    return rotation_matrix


def calculate_fov_from_translation(T, width, height):
    """
    Calculate the horizontal (FovX) and vertical (FovY) field of view angles
    from the translation vector T and image size.
    """
    t_z = T[2]
    fov_x = 2 * np.arctan((width / 2) / abs(t_z))
    fov_y = 2 * np.arctan((height / 2) / abs(t_z))
    return np.degrees(fov_x), np.degrees(fov_y)


def extract_data_from_dicom_image(dicom_path, output_folder, white_background=True):
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_path)

        # Normalize pixel values to the range [0, 255]
        normalized_array = ((dicom_data.pixel_array - dicom_data.pixel_array.min()) /
                            (dicom_data.pixel_array.max() - dicom_data.pixel_array.min()) * 255).astype(np.uint8)
        normalized_array = cv2.rotate(normalized_array, cv2.ROTATE_90_CLOCKWISE)
        # normalized_array = np.where(normalized_array > 130, 1, 0).astype(np.uint8) * 255
        # resized_image = cv2.resize(normalized_array, resize_dim, interpolation=cv2.INTER_LINEAR)

        # Convert the image to RGBA (to add transparency)
        rgba_image = cv2.cvtColor(normalized_array, cv2.COLOR_GRAY2RGBA)

        # Define a threshold to make certain pixel values (e.g., black or near-black) transparent
        threshold = 80  # Set threshold for transparency, adjust as needed
        black_pixels = rgba_image[:, :, 0] <= threshold  # Find pixels below the threshold
        rgba_image[black_pixels, 0:3] = 255
        # Set the alpha channel (4th channel) to 0 for transparent pixels
        rgba_image[black_pixels, 3] = 255  # Set alpha to 0 (fully transparent)

        # Save as lossless PNG with transparency
        output_path = os.path.join(output_folder, os.path.basename(dicom_path).replace('.dcm', '.png'))
        cv2.imwrite(output_path, rgba_image)
        # Save as lossless PNG
        # output_path = os.path.join(output_folder, os.path.basename(dicom_path).replace('.dcm', '.png'))
        # cv2.imwrite(output_path, normalized_array)

        pixel_spacing = dicom_data.get((0x0018, 0x0090)).value
        detector_rows = dicom_data.get((0x0028, 0x0010)).value
        detector_columns = dicom_data[0x0028, 0x0011].value
        source_to_patient = struct.unpack('<f', dicom_data.get((0x7031, 0x1003)).value)[0]
        source_to_detector_distance = struct.unpack('<f', dicom_data.get((0x7031, 0x1031)).value)[0]
        col_width = struct.unpack('<f', dicom_data.get((0x7029, 0x1002)).value)[0]
        row_width = struct.unpack('<f', dicom_data.get((0x7029, 0x1006)).value)[0]
        theta = struct.unpack('<f', dicom_data.get((0x7031, 0x1001)).value)[0]
        table_movement = struct.unpack('<f', dicom_data.get((0x7031, 0x1002)).value)[0]+213
        z_adjustment = dicom_data.get((0x0018, 0x9311)).value
        source_to_detector_distance = source_to_detector_distance - source_to_patient

        # extr = np.array([
        #     [-np.sin(theta), np.cos(theta), 0, 0],
        #     [0, 0, -1, 0],
        #     [-np.cos(theta), -np.sin(theta), 1, source_to_patient],
        #     [0, 0, 0, 1]
        # ])
        # extr = np.array([
        #     [np.cos(theta), 0, np.sin(theta), 0],  # Row 1
        #     [0, 1, 0, table_movement*(-1)],  # Row 2 (Y-axis stays unchanged)
        #     [-np.sin(theta), 0, np.cos(theta), source_to_patient],  # Row 3
        #     [0, 0, 0, 1]  # Row 4 (homogeneous coordinate)
        # ])
        # extr = np.array([
        #     [1, 0, 0, 0],  # X-axis stays unchanged
        #     [0, np.cos(theta), -np.sin(theta), table_movement * (-1)],  # Rotation and translation in YZ plane
        #     [0, np.sin(theta), np.cos(theta), source_to_patient],  # Z-axis affected by rotation
        #     [0, 0, 0, 1]  # Homogeneous coordinate
        # ])

        extr = np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],          # Rotation in XY plane
            [np.sin(theta), np.cos(theta), 0, table_movement*(-1)],  # Y-axis translation
            [0, 0, 1, source_to_patient],                   # Z-axis remains unchanged
            [0, 0, 0, 1]                                    # Homogeneous coordinate
        ])

        R = extr[:3, :3]
        T = extr[:3, 3]
        detector_column_width = detector_columns * col_width
        detector_row_width = detector_rows * row_width

        # FovY = 2 * np.degrees(np.arctan(detector_row_width / (2 * source_to_detector_distance)))
        # FovX = 2 * np.degrees(np.arctan(detector_column_width / (2 * source_to_detector_distance)))
        #
        # FovX = source_to_detector_distance
        # FovY = source_to_detector_distance

        FovY = 0.1
        FovX = 0.1

        # Open PNG image to include in the camera info using 'with' statement
        with Image.open(output_path) as img:
            cam_info = CameraInfo(
                uid=dicom_data.InstanceNumber,
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=img.copy(),  # Copy image to use after closing
                image_path=output_path,
                image_name=os.path.basename(output_path),
                width=img.size[0],
                height=img.size[1]
            )

        return cam_info

    except Exception as e:
        print(f"Error converting {dicom_path} to lossless PNG: {e}")
        return None


# Convert DICOM images to lossless PNG and extract camera information
def getCameraInfos(batch_size=50):
    camera_infos = []
    dicom_files = [f for f in os.listdir(dicom_folder) if f.endswith('.dcm')]

    for i in range(0, len(dicom_files), batch_size):
        batch = dicom_files[i:i + batch_size]

        for filename in batch:
            dicom_path = os.path.join(dicom_folder, filename)
            camera_info = extract_data_from_dicom_image(dicom_path, output_folder)
            if camera_info:
                camera_infos.append(camera_info)

    return camera_infos


# Visualization function for the camera information (optional)
def visualize_camera_info(camera_infos):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, camera_info in enumerate(camera_infos):
        position = camera_info.T
        rotation_matrix = camera_info.R
        ax.scatter(position[0], position[1], position[2], color='red', s=50)

        direction = rotation_matrix[:, 2]
        ax.quiver(position[0], position[1], position[2],
                  direction[0], direction[1], direction[2],
                  color=f'C{idx % 10}', length=50, normalize=True)

    positions = np.array([camera_info.T for camera_info in camera_infos])
    ax.set_xlim([positions[:, 0].min(), positions[:, 0].max()])
    ax.set_ylim([positions[:, 1].min(), positions[:, 1].max()])
    ax.set_zlim([positions[:, 2].min(), positions[:, 2].max()])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


# Main program setup
dicom_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/CT-data/dicom/lung500_8500"
output_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/gaussian-splatting/images"
os.makedirs(output_folder, exist_ok=True)

camera_infos = getCameraInfos()

# Optional visualization
# visualize_camera_info(camera_infos)

print("Conversion complete.")

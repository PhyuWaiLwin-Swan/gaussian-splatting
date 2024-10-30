# random test with the dicom and synthesis data, not use in the project

import math
import os
import json
import struct

import pydicom
import cv2
import numpy as np
from typing import NamedTuple
from math import tan, radians, cos, sin

from PIL import Image


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


# Extract data from the dicom image
def extract_data_from_dicom_image(dicom_path, output_folder):
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_path)

        # Normalize pixel values to the range [0, 255]
        normalized_array = ((dicom_data.pixel_array - dicom_data.pixel_array.min()) /
                            (dicom_data.pixel_array.max() - dicom_data.pixel_array.min()) * 255).astype(np.uint8)
        normalized_array = cv2.rotate(normalized_array, cv2.ROTATE_90_CLOCKWISE)
        normalized_array = np.where(normalized_array > 120, 1, 0).astype(np.uint8) * 255
        # Convert DICOM to lossless PNG
        output_path = os.path.join(output_folder, os.path.basename(dicom_path).replace('.dcm', '.png'))
        cv2.imwrite(output_path, normalized_array)

        encrypt_type = 'little'
        if dicom_data.file_meta.TransferSyntaxUID == '1.2.840.10008.1.2.2':
            encrypt_type = 'big'


        pixel_spacing = dicom_data.get((0x0018, 0x0090)).value
        detector_rows = dicom_data.get((0x0028, 0x0010)).value
        detector_columns = dicom_data[0x0028, 0x0011].value
        source_to_patient = struct.unpack('<f', dicom_data.get((0x7031, 0x1003)).value)[0]
        source_to_detector_distance = struct.unpack('<f', dicom_data.get((0x7031, 0x1031)).value)[0]
        col_width = struct.unpack('<f', dicom_data.get((0x7029, 0x1002)).value)[0]
        row_width = struct.unpack('<f', dicom_data.get((0x7029, 0x1006)).value)[0]
        theta = struct.unpack('<f', dicom_data.get((0x7031, 0x1001)).value)[0]
        table_movement = struct.unpack('<f', dicom_data.get((0x7031, 0x1002)).value)[0]
        detector_shape = dicom_data.get((0x7029, 0x100B)).value
        typeofProjectData = dicom_data.get((0x7037, 0x1009)).value
        pitch = dicom_data.get((0x0018, 0x9311)).value
        # off_theta = struct.unpack('<f', dicom_data.get((0x7033, 0x100B)).value)[0]
        # theta += off_theta  # Apply offset to angle
        o_p = struct.unpack('<f', dicom_data.get((0x7033, 0x100D)).value)[0]
        o_z = struct.unpack('<f', dicom_data.get((0x7033, 0x100C)).value)[0]
        # p += o_p
        table_movement -= o_z
        detector_column_width = detector_columns * col_width
        detector_row_width = detector_rows * row_width

        bean_angle_x = math.atan((detector_columns * col_width / 2 ) /source_to_detector_distance)
        bean_angle_y = math.atan((detector_rows * row_width/2)/source_to_detector_distance)



        fov_x = math.tan(bean_angle_x) * source_to_patient * 2
        fov_y = math.tan(bean_angle_y) * source_to_patient * 2

        fov_x = bean_angle_x
        fov_y = bean_angle_y
        # Rotation matrix
        R = np.array([
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])
        phi1 = -np.pi*2
        R1 = np.array([[1.0, 0.0, 0.0],
                       [0.0, np.cos(phi1), -np.sin(phi1)],
                       [0.0, np.sin(phi1), np.cos(phi1)]])
        phi2 = np.pi*2
        R2 = np.array([[np.cos(phi2), -np.sin(phi2), 0.0],
                       [np.sin(phi2), np.cos(phi2), 0.0],
                       [0.0, 0.0, 1.0]])
        R3 = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                       [np.sin(theta), np.cos(theta), 0.0],
                       [0.0, 0.0, 1.0]])
        R = np.dot(np.dot(R3, R2), R1)
        # t = np.array([source_to_patient  * np.cos(theta), source_to_patient  * np.sin(theta), pitch/(2*math.pi) * theta])



        # R = np.array([
        #     [1,0, 0],
        #     [0,1, 0],
        #     [0, 0, 1]
        # ])
        #rotate around horizontal in world axis ,which is x
        # R = np.array([
        #     [1, 0, 0],
        #     [0, cos(theta), -sin(theta)],
        #     [0, sin(theta), cos(theta)]
        # ])
        #rotate on y axis
        # R = np.array([
        #     [cos(theta), 0, sin(theta)],
        #     [0, 1, 0],
        #     [-sin(theta), 0, cos(theta)]
        # ])
        # Translation vector




        x = -source_to_patient*math.sin(theta)
        y = -source_to_patient*math.cos(theta)
        z = table_movement - 213.60
        #source to pati
        # ent is the z direction in camera view, x is point up and down in rotaiton so it equal to y in world coordinate, z is always horizontal
        # t = np.array([z,y,-source_to_patient])  # Translation vector
        # t= np.array([z,y,-source_to_patient])
        # Calculate width and height in pixels
        # t = np.array([z, y, source_to_patient + pitch/(2*math.pi) * theta])
        # t = np.array([0, 0, source_to_patient + pitch / (2 * math.pi) * theta])
        t = np.array([0, 0, -z/2])
        # t = np.array(
            # [source_to_patient * np.cos(theta), source_to_patient * np.sin(theta), -z])
        # Extract camera information
        camera_info = CameraInfo(
            uid=dicom_data.InstanceNumber,  # Using InstanceNumber as UID
            R=R,  # Rotation matrix
            T=t,  # Translation vector
            FovY=fov_y,  # Field of view in the Y direction
            FovX=fov_x,  # Field of view in the X direction
            image=Image.open(output_path).convert('RGB'),  # Base64 encoded image
            image_path=output_path,
            image_name=os.path.basename(output_path),
            width=detector_column_width,
            height=detector_row_width,
        )

        # print(f"Converted {dicom_path} to lossless PNG: {output_path}")
        return camera_info
    except Exception as e:
        print(f"Error converting {dicom_path} to lossless PNG: {e}")
        return None

# Path to DICOM folder and output folder for PNG images
dicom_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/CT-data/dicom/lung_PD"
# dicom_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/CT-data/dicom/C001/1.2.840.113713.4.100.1.2.123467221304001792631562653249153/1.2.840.113713.4.100.1.2.259331308512386262319022488881735"
output_folder = ("/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/gaussian-splatting/images")
camera_info_filename = "camera_infos.json"

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Convert DICOM images to lossless PNG and extract camera information
def getCameraInfos():

    camera_infos = []
    for filename in os.listdir(dicom_folder):
        if filename.endswith('.dcm'):
            dicom_path = os.path.join(dicom_folder, filename)
            camera_info = extract_data_from_dicom_image(dicom_path, output_folder)
            if camera_info:
                camera_infos.append(camera_info)
    return camera_infos

print("Conversion complete.")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_camera(ax, position, rotation_matrix, color='blue', length=0.2):
    """Plot the camera position and its facing direction."""
    ax.scatter(position[0], position[1], position[2], color='red', s=50)

    # Plot the camera's facing direction
    # The direction is typically the third column of the rotation matrix
    direction = rotation_matrix[:, 2]  # Extract the forward direction (Z-axis in camera space)
    ax.quiver(position[0], position[1], position[2],
              direction[0], direction[1], direction[2],
              color=color, length=0.02, normalize=True)

    direction1 = rotation_matrix[:, 1]  # Extract the forward direction (Z-axis in camera space)
    ax.quiver(position[0], position[1], position[2],
              direction1[0], direction1[1], direction1[2],
              color='green', length=0.02, normalize=True)

    direction2 = rotation_matrix[:, 0]  # Extract the forward direction (Z-axis in camera space)
    ax.quiver(position[0], position[1], position[2],
                  direction2[0], direction2[1], direction2[2],
                  color='yellow', length=0.02, normalize=True)

def visualize_camera_info(camera_infos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for idx, camera_info in enumerate(camera_infos):
        # Extract the camera's position and rotation matrix
        position = camera_info.T
        rotation_matrix = camera_info.R

        # Plot the camera position and rotation
        plot_camera(ax, position, rotation_matrix, color=f'C{idx % 10}', length=120)

    # Set the axes limits based on the camera positions
    positions = np.array([camera_info.T for camera_info in camera_infos])
    ax.set_xlim([positions[:, 0].min(), positions[:, 0].max()])
    ax.set_ylim([positions[:, 1].min(), positions[:, 1].max()])
    ax.set_zlim([positions[:, 2].min(), positions[:, 2].max()])

    # Set the labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


# Retrieve camera infos
camera_infos = getCameraInfos()

# Visualize the camera movement
visualize_camera_info(camera_infos)



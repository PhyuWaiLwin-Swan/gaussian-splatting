import os
import json
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

        # Convert DICOM to lossless PNG
        output_path = os.path.join(output_folder, os.path.basename(dicom_path).replace('.dcm', '.png'))
        cv2.imwrite(output_path, normalized_array)

        # Extract relevant metadata
        detector_rows = int(dicom_data.Rows)
        detector_columns = int(dicom_data.Columns)
        source_to_detector_distance = float(dicom_data[0x0018, 0x1110].value)  # Replace with the correct tag
        source_to_patient_distance = float(dicom_data[0x0018, 0x1111].value)  # Replace with the correct tag

        # Calculate Field of View (FOV)
        pixel_spacing = dicom_data.PixelSpacing
        fov_x = detector_columns * pixel_spacing[0]
        fov_y = detector_rows * pixel_spacing[1]

        # Calculate intrinsic parameters
        f_x = detector_columns / (2 * tan(radians(fov_x) / 2))
        f_y = detector_rows / (2 * tan(radians(fov_y) / 2))
        c_x = detector_columns / 2
        c_y = detector_rows / 2

        K = np.array([
            [f_x, 0, c_x],
            [0, f_y, c_y],
            [0, 0, 1]
        ])

        # Define extrinsic parameters (assuming initial angle 0 for demonstration)
        theta = 0  # Angle in radians
        R = np.array( [
            [cos(theta), -sin(theta), 0],
            [sin(theta), cos(theta), 0],
            [0, 0, 1]
        ])

        t = np.array([0, 0, -source_to_patient_distance])  # Translation along z-axis


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
            width=normalized_array.shape[1],
            height=normalized_array.shape[0],
        )

        print(f"Converted {dicom_path} to lossless PNG: {output_path}")
        return camera_info
    except Exception as e:
        print(f"Error converting {dicom_path} to lossless PNG: {e}")
        return None

# Path to DICOM folder and output folder for PNG images
dicom_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/CT-data/dicom/lungCT-LC"
output_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/gaussian-splatting/images"
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

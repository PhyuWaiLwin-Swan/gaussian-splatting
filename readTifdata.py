import os
import uuid
import numpy as np
import cv2
from PIL import Image
from typing import NamedTuple

class CameraInfo(NamedTuple):
    uid: str
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


def extract_data_from_tif_image(tif_path, output_folder, col_width=0.05, row_width=0.05, source_to_patient=210.66, source_to_detector_distance=553.74, start_angle=0, angle_interval=0.5):
    try:
        # Read the .tif file
        image = Image.open(tif_path)
        image_array = np.array(image)

        # Normalize pixel values to the range [0, 255]
        normalized_array = ((image_array - image_array.min()) /
                            (image_array.max() - image_array.min()) * 255).astype(np.uint8)

        # Convert the image to RGBA (to add transparency)
        rgba_image = cv2.cvtColor(normalized_array, cv2.COLOR_GRAY2RGBA)

        # Define a threshold to make certain pixel values (e.g., black or near-black) transparent
        threshold = 80  # Set threshold for transparency, adjust as needed
        black_pixels = rgba_image[:, :, 0] >= threshold  # Find pixels below the threshold
        rgba_image[black_pixels, 0:3] = 255
        # Set the alpha channel (4th channel) to 0 for transparent pixels
        rgba_image[black_pixels, 3] = 255  # Set alpha to 0 (fully transparent)

        # Calculate the current angle for this image
        idx = int(os.path.basename(tif_path).split("_")[-1].replace(".tif", ""))  # Extract index from file name
        current_angle = start_angle + idx * angle_interval
        theta = np.radians(current_angle)  # Convert angle to radians

        # Save the image as PNG
        output_file = f"{os.path.basename(tif_path).replace('.tif', '')}_angle_{current_angle:.1f}.png"
        output_path = os.path.join(output_folder, output_file)
        cv2.imwrite(output_path, rgba_image)

        # print(f"Image saved to {output_path} with angle {current_angle:.1f} degrees (theta = {theta:.4f} radians)")

        # Create extrinsic matrix
        # extr = np.array([
        #     [-np.sin(theta), np.cos(theta), 0, 0],
        #     [0, 0, -1, 0],
        #     [-np.cos(theta), -np.sin(theta), 1, source_to_patient],
        #     [0, 0, 0, 1]
        # ]) # upside down y
        #     extr = np.array([
        #     [1, 0, 0, 0],
        #     [0, np.cos(theta), -np.sin(theta), 0],
        #     [0, np.sin(theta), np.cos(theta), 0],
        #     [0, 0, 0, 1]
        # ]) # x axis rotation
        extr = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],  # Row 1
            [0, 1, 0, 0],  # Row 2 (Y-axis stays unchanged)
            [-np.sin(theta), 0, np.cos(theta), source_to_patient],  # Row 3
            [0, 0, 0, 1]  # Row 4 (homogeneous coordinate)
        ]) # Y axis rotation #2e913bb56 # successful ge

        # extr = np.array([
        #     [np.cos(theta), -np.sin(theta), 0, 0],
        #     [np.sin(theta), np.cos(theta), 0, 0],
        #     [0, 0, 1, source_to_patient],
        #     [0, 0, 0, 1]
        # ]) # 83fad , z axis rotation

        R = extr[:3, :3]  # Rotation matrix
        T = extr[:3, 3]   # Translation vector

        # Calculate Field of View (FoV)
        detector_rows, detector_columns = rgba_image.shape[:2]
        detector_column_width = detector_columns * col_width
        detector_row_width = detector_rows * row_width

        FovY = 0.05
        FovX = 0.05

        # Open PNG image to include in the camera info
        with Image.open(output_path) as img:
            cam_info = CameraInfo(
                uid=str(uuid.uuid4()),  # Generate UUID
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=Image.open(output_path).convert('RGB'),
                image_path=output_path,
                image_name=os.path.basename(output_path),
                width=img.size[0],
                height=img.size[1]
            )

        return cam_info

    except Exception as e:
        print(f"Error processing {tif_path}: {e}")
        return None


def getCameraInfos():
    camera_infos = []

    # List all .tif files in the folder
    tif_files = [f for f in os.listdir(tif_folder) if f.endswith('.tif')]

    # Iterate over each .tif file
    for filename in tif_files:
        # Construct the full path to the .tif file
        tif_path = os.path.join(tif_folder, filename)

        # Extract camera information from the .tif image
        camera_info = extract_data_from_tif_image(tif_path, output_folder)

        # If camera_info is successfully extracted, add it to the list
        if camera_info:
            camera_infos.append(camera_info)

    return camera_infos


# Main program setup
tif_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/CT-data/dicom/walnatTif"
output_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/gaussian-splatting/images"
os.makedirs(output_folder, exist_ok=True)
getCameraInfos()
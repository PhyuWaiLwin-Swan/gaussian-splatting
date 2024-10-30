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


def extract_data_from_png_image(png_path, output_folder, col_width=0.05, row_width=0.05, source_to_patient=541, source_to_detector_distance=553.74, start_angle=0, angle_interval=5):
    try:
        # Read the .png file
        image = Image.open(png_path)
        image_array = np.array(image)

        # Normalize pixel values to the range [0, 255]
        normalized_array = ((image_array - image_array.min()) /
                            (image_array.max() - image_array.min()) * 255).astype(np.uint8)

        # Convert the image to RGBA (to add transparency)
        rgba_image = cv2.cvtColor(normalized_array, cv2.COLOR_GRAY2RGBA)

        # Define a threshold to make certain pixel values (e.g., black or near-black) transparent
        threshold = 80  # Set threshold for transparency, adjust as needed
        black_pixels = rgba_image[:, :, 0] <= threshold  # Find pixels below the threshold
        rgba_image[black_pixels, 0:3] = 255
        # Set the alpha channel (4th channel) to 0 for transparent pixels
        rgba_image[black_pixels, 3] = 255  # Set alpha to 0 (fully transparent)

        # Extract the numerical index from the filename
        filename = os.path.basename(png_path)
        idx = int(''.join(filter(str.isdigit, filename)))  # Extract digits from filename
        current_angle = start_angle + idx * angle_interval
        theta = np.radians(current_angle)  # Convert angle to radians

        output_file = f"{os.path.basename(png_path).replace('.png', '')}_angle_{current_angle:.1f}.png"
        output_path = os.path.join(output_folder, output_file)

        success = cv2.imwrite(output_path, rgba_image)
        if not success:
            raise IOError(f"Failed to write the image to {output_path}")

        # Create extrinsic matrix
        extr = np.array([
            [np.cos(theta), 0, np.sin(theta), 0],  # Row 1
            [0, 1, 0, 0],  # Row 2 (Y-axis stays unchanged)
            [-np.sin(theta), 0, np.cos(theta), source_to_patient],  # Row 3
            [0, 0, 0, 1]  # Row 4 (homogeneous coordinate)
        ])  # Y-axis rotation

        R = extr[:3, :3]  # Rotation matrix
        T = extr[:3, 3]   # Translation vector


        FovY = 0.1
        FovX = 0.1

        # Open PNG image to include in the camera info
        with Image.open(output_path) as img:
            cam_info = CameraInfo(
                uid=str(uuid.uuid4()),  # Generate UUID
                R=R,
                T=T,
                FovY=FovY,
                FovX=FovX,
                image=img.convert('RGB'),
                image_path=output_path,
                image_name=os.path.basename(output_path),
                width=img.size[0],
                height=img.size[1]
            )

        return cam_info

    except Exception as e:
        print(f"Error processing {png_path}: {e}")
        return None


def getCameraInfos():
    camera_infos = []

    # List all .png files in the folder
    png_files = [f for f in os.listdir(png_folder) if f.endswith('.png')and int(f.split('xray')[-1].split('.')[0]) % 1 == 0]

    # Iterate over each .png file
    for filename in png_files:
        # Construct the full path to the .png file
        png_path = os.path.join(png_folder, filename)

        # Extract camera information from the .png image
        camera_info = extract_data_from_png_image(png_path, output_folder)

        # If camera_info is successfully extracted, add it to the list
        if camera_info:
            camera_infos.append(camera_info)

    return camera_infos


# Main program setup
png_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/CT-data/png/knee"
output_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/gaussian-splatting/images"
os.makedirs(output_folder, exist_ok=True)
getCameraInfos()

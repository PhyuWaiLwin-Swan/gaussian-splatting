# This file has been use to convert the dicom image to the png format
# This purpose of this file to to convert the dicom image to the png format and extract the informaiton
import os
import pydicom
from PIL import Image
import cv2
import SimpleITK as sitk

# Function to convert DICOM to lossless JPEG
def convert_dicom_to_jpeg(dicom_path, output_path):
    try:
        # # Read DICOM file
        # dicom_data = pydicom.dcmread(dicom_path)
        #
        # # Convert pixel data to Pillow image
        # cv2.imwrite(output_path, dicom_data.pixel_array)
        # Save as lossless JPEG
        img = sitk.ReadImage(dicom_path)
        img = sitk.IntensityWindowing(img, -1000, 1000, 0, 255)
        img = sitk.Cast(img, sitk.sitkUInt8)
        sitk.WriteImage(img, output_path)


        print(f"Converted {dicom_path} to lossless JPEG: {output_path}")
    except Exception as e:
        print(f"Error converting {dicom_path} to lossless JPEG: {e}")

#Specifies the correct path for dicom folder and the output folder
dicom_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/CT-data/dicom/lungCT-LC"
output_folder = "/csse/users/pwl24/Desktop/fourth_year_2024/Seng_402/3dgs-CT/CT-data/png/lung-png"

# Create output folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Convert DICOM images to lossless Png
print("Converting DICOM images to lossless JPEG format...")
for filename in os.listdir(dicom_folder):
    if filename.endswith('.dcm'):
        dicom_path = os.path.join(dicom_folder, filename)
        output_path = os.path.join(output_folder, filename.replace('.dcm', '.png'))
        convert_dicom_to_jpeg(dicom_path, output_path)
print("Conversion complete.")

import os
from PIL import Image
import shutil

def split_image_folder(input_folder, output_folder, n):
    # Create output folders if they don't exist
    for i in range(n):
        folder_path = os.path.join(output_folder, f'part_{i + 1}')
        os.makedirs(folder_path, exist_ok=True)

    # List all JPG files in the input folder
    jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

    # Calculate the number of images per part
    images_per_part = len(jpg_files) // n

    # Split the list of JPG files into n parts
    image_parts = [jpg_files[i:i+images_per_part] for i in range(0, len(jpg_files), images_per_part)]

    # Process each part
    for i, part in enumerate(image_parts):
        # Copy each image to the corresponding output folder
        for image_file in part:
            input_path = os.path.join(input_folder, image_file)
            output_folder_path = os.path.join(output_folder, f'part_{i + 1}')
            output_path = os.path.join(output_folder_path, image_file)
            shutil.copy2(input_path, output_path)

if __name__ == "__main__":
    # Replace 'input_folder' and 'output_folder' with your actual folder paths
    input_folder = r'C:\Users\shatn\OneDrive\Desktop\Merged no tumor datasets - Copy'
    output_folder = r'C:\Users\shatn\OneDrive\Desktop\Merged no tumor datasets splited'
    
    # Replace 'n' with the number of folders you want
    n = 3
    
    split_image_folder(input_folder, output_folder, n)

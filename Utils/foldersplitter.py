import os
from PIL import Image

def split_image_folder(input_folder, output_folder, n):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all JPG files in the input folder
    jpg_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]

    # Calculate the number of images per part
    images_per_part = len(jpg_files) // n

    # Split the list of JPG files into n parts
    image_parts = [jpg_files[i:i+images_per_part] for i in range(0, len(jpg_files), images_per_part)]

    # Process each part
    for i, part in enumerate(image_parts):
        # Create a subfolder for each part
        subfolder = os.path.join(output_folder, f'part_{i + 1}')
        os.makedirs(subfolder)

        # Copy each image to the corresponding subfolder
        for image_file in part:
            input_path = os.path.join(input_folder, image_file)
            output_path = os.path.join(subfolder, image_file)
            img = Image.open(input_path)
            img.save(output_path)

if __name__ == "__main__":
    # Replace 'input_folder' and 'output_folder' with your actual folder paths
    input_folder = 'path/to/your/input/folder'
    output_folder = 'path/to/your/output/folder'
    
    # Replace 'n' with the number of parts you want
    n = 3
    
    split_image_folder(input_folder, output_folder, n)

import os
import shutil
import random

def split_images(input_folder, output_folder_1, output_folder_2, split_ratio=0.7):
    # Create output folders if they don't exist
    os.makedirs(output_folder_1, exist_ok=True)
    os.makedirs(output_folder_2, exist_ok=True)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))]

    # Calculate the number of images for each folder based on the split ratio
    num_images_folder_1 = int(len(image_files) * split_ratio)
    num_images_folder_2 = len(image_files) - num_images_folder_1

    # Randomly shuffle the list of image files
    random.shuffle(image_files)

    # Copy images to the output folders based on the split ratio
    for i, image_file in enumerate(image_files):
        source_path = os.path.join(input_folder, image_file)
        if i < num_images_folder_1:
            destination_path = os.path.join(output_folder_1, image_file)
        else:
            destination_path = os.path.join(output_folder_2, image_file)
        shutil.copy2(source_path, destination_path)

    print(f"Splitting complete. {num_images_folder_1} images in {output_folder_1} and {num_images_folder_2} images in {output_folder_2}.")

# Example usage
input_folder_path = r'/home/homam/Desktop/no tumor/'
output_folder_1_path = r'/home/homam/Desktop/no tumor 70%/'
output_folder_2_path = r'/home/homam/Desktop/no tumor 30%/'

split_images(input_folder_path, output_folder_1_path, output_folder_2_path, split_ratio=0.7)

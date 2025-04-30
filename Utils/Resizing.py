from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(input_folder, filename)
            print(f"Processing image: {img_path}")
            img = Image.open(img_path)
            img_resized = img.resize(target_size, Image.LANCZOS)  # Updated line    # Updated line

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, filename)
            print(f"Saving resized image to: {output_path}")
            img_resized.save(output_path, 'JPEG')

# Specify input and output folders and target size
input_folder = r'Drop the path of the folder here'
output_folder = r'Drop the path of the folder here'
target_size = (X, X)  # Adjust the size as needed

# Resize images
resize_images(input_folder, output_folder, target_size)

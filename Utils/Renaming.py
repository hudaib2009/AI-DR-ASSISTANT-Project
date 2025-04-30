import os
import shutil

def rename_jpg_images(source_folder, destination_folder):
    # Ensure the destination folder exists, create it if necessary
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # List all files in the source folder
    files = os.listdir(source_folder)

    # Counter for renaming sequentially
    counter = 1

    # Iterate through each file and check if it's a JPG image
    for file in files:
        if file.lower().endswith(".jpg"):
            # Generate a new name with a sequential number
            new_name = f"{counter}.jpg"

            # Create the full path for the source and destination files
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, new_name)

            # Rename the file by moving it to the destination folder
            shutil.move(source_path, destination_path)

            print(f"Renamed: {file} -> {new_name}")

            # Increment the counter for the next file
            counter += 1

# Replace 'source_folder' and 'destination_folder' with your actual folder paths
source_folder = r'Drop The Path Of The Folder Here'
destination_folder = r'Drop The Path Of The Folder Here'

rename_jpg_images(source_folder, destination_folder)
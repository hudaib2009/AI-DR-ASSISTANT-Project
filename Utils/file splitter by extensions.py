import os
import shutil

def split_files_by_extension(source_folder, extension1_folder, extension2_folder):
    # Create folders if they don't exist
    os.makedirs(extension1_folder, exist_ok=True)
    os.makedirs(extension2_folder, exist_ok=True)

    # List all files in the source folder
    files = os.listdir(source_folder)

    for file in files:
        # Check if the file is a .jpg or .cr2 file
        if file.lower().endswith(".extension1"):
            shutil.move(os.path.join(source_folder, file), os.path.join(extension1_folder, file))
        elif file.lower().endswith(".extension2"):
            shutil.move(os.path.join(source_folder, file), os.path.join(extension2_folder, file))

# Specify your folder paths
source_folder = r'/path/to/source/folder'
extension1_folder = r'/path/to/extension1/folder'
extension2_folder = r'/path/to/extension2/folder'

# Call the function to split files
split_files_by_extension(source_folder, extension1_folder, extension2_folder)

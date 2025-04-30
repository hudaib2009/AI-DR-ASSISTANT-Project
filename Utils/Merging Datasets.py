import os
import shutil

# Source folders
folder1 = r'C:\Users\shatn\OneDrive\Desktop\rename no tumor - Copy'
folder2 = r'C:\Users\shatn\OneDrive\Desktop\rename no tumor 2 - Copy'
folder3 = r'C:\Users\shatn\OneDrive\Desktop\rename no tumor 3 - Copy'

# Destination folder
destination_folder = r'C:\Users\shatn\OneDrive\Desktop\Merged Datasets'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Copy and rename images from folder1
for i, filename in enumerate(os.listdir(folder1)):
    src_path = os.path.join(folder1, filename)
    dest_path = os.path.join(destination_folder, f'{i + 1}.jpg')
    shutil.copy(src_path, dest_path)

# Copy and rename images from folder2
for i, filename in enumerate(os.listdir(folder2)):
    src_path = os.path.join(folder2, filename)
    dest_path = os.path.join(destination_folder, f'{i + 1 + len(os.listdir(folder1))}.jpg')
    shutil.copy(src_path, dest_path)

# Copy and rename images from folder3
for i, filename in enumerate(os.listdir(folder3)):
    src_path = os.path.join(folder3, filename)
    dest_path = os.path.join(destination_folder, f'{i + 1 + len(os.listdir(folder1)) + len(os.listdir(folder2))}.jpg')
    shutil.copy(src_path, dest_path)

print("Images merged and renamed successfully.")

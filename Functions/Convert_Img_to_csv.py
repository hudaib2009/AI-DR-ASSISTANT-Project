import pandas as pd
import cv2 # OpenCV library for image processing
import os
import numpy as np
from tqdm import tqdm # Optional: for progress bar
import math # To calculate number of batches

# --- Configuration ---
# !! IMPORTANT: Update these paths to match your system !!
# Make sure this points to the CSV file containing ALL your data rows
csv_file_path = r'Data/Data_Entry_2017.csv' # <--- CHANGE THIS
# Make sure this points to the folder containing all the corresponding images
image_folder_path = r'Data/Merged_Images' # Use raw string for Windows paths
# Choose a name for the output file
output_csv_path = 'Data/Images_Batched.csv' # Changed name slightly to avoid overwrite

image_size = (128, 128)
batch_size = 50000 # Process 50000 images at a time
# !! IMPORTANT: Specify the exact name of the column containing the labels !!
label_column_name = 'Finding_Labels' # <--- CONFIRM THIS is the correct column name in your CSV
# --- End Configuration ---

# --- Load Full Index Data ---
print(f"Attempting to read the full CSV index file from: {csv_file_path}")
try:
    # This line reads your ACTUAL CSV file specified above
    df_full_index = pd.read_csv(csv_file_path)
    print("Full CSV index data loaded successfully.")
    print(f"Total images to process: {len(df_full_index)}")
    print(f"Index DataFrame shape: {df_full_index.shape}")
    # print(f"Columns: {df_full_index.columns.tolist()}") # Optional: uncomment to see all columns

    # --- Verify Label Column Exists ---
    if label_column_name not in df_full_index.columns:
        print(f"*** FATAL ERROR: The specified label column '{label_column_name}' does not exist in the input CSV '{csv_file_path}'. ***")
        print(f"Available columns are: {df_full_index.columns.tolist()}")
        print("Please update the 'label_column_name' variable in the script.")
        exit()
    else:
        print(f"Confirmed label column '{label_column_name}' exists.")

except FileNotFoundError:
    print(f"*** Error: CSV file not found at '{csv_file_path}'. Please check the path. ***")
    exit() # Stop the script if the file isn't found
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- Prepare for Batch Processing ---
total_images = len(df_full_index)
num_batches = math.ceil(total_images / batch_size)
# Get metadata columns, EXCLUDING the Image_Index AND the specified label column (we'll add label back later)
metadata_columns = df_full_index.columns.drop(['Image_Index', label_column_name], errors='ignore')
num_pixels = image_size[0] * image_size[1]
pixel_columns = [f'pixel_{i}' for i in range(num_pixels)]

print(f"\nStarting image processing in {num_batches} batches of size {batch_size}...")
print(f"Output will have pixel columns first, then metadata, with '{label_column_name}' as the final column.")

# --- Main Batch Processing Loop ---
for i in range(num_batches):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, total_images) # Ensure end index doesn't exceed total
    batch_df_index = df_full_index.iloc[start_index:end_index] # Get the slice for this batch

    print(f"\n--- Processing Batch {i+1}/{num_batches} (Rows {start_index} to {end_index-1}) ---")

    # Reset lists for each batch
    processed_metadata_list = []
    pixel_data_list = []
    original_filenames_list = []
    label_list = [] # List to store labels separately

    # Inner loop processes only the images in the current batch
    for index, row in tqdm(batch_df_index.iterrows(), total=batch_df_index.shape[0], desc=f"Batch {i+1}"):
        image_filename = row['Image_Index']
        image_path = os.path.join(image_folder_path, image_filename)

        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"\nWarning: Image file not found, skipping row {index} ({start_index + batch_df_index.index.get_loc(index)} overall): {image_path}")
            continue # Skip this row

        try:
            # Read the image in grayscale mode directly
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print(f"\nWarning: Failed to load image (corrupted?), skipping row {index} ({start_index + batch_df_index.index.get_loc(index)} overall): {image_path}")
                continue # Skip if image loading failed

            # Resize the image
            img_resized = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)

            # Flatten the image into a 1D array (128*128 = 16384 pixels)
            pixel_values = img_resized.flatten()

            # Store pixel data and metadata for this row
            pixel_data_list.append(pixel_values)
            # Get metadata BUT EXCLUDE the label column for now
            metadata = row[metadata_columns].to_dict()
            processed_metadata_list.append(metadata)
            original_filenames_list.append(image_filename) # Store the original filename
            label_list.append(row[label_column_name]) # Store the label separately

        except Exception as e:
            global_row_index = start_index + batch_df_index.index.get_loc(index) # Calculate overall index for error message
            print(f"\nError processing image {image_path} (original CSV row index: {index}, overall image index: {global_row_index}): {e}")
            # Decide how to handle errors, e.g., skip the row
            continue

    # --- Post-Processing (for this batch) ---
    print(f"\nBatch {i+1} processing finished. Processed {len(pixel_data_list)} images successfully in this batch.")

    # Check if any data was processed in this batch
    if not pixel_data_list: # Check pixel data list as it's fundamental
        print(f"No images were successfully processed in batch {i+1}. Skipping save for this batch.")
        continue

    # Create DataFrames from the collected lists for THIS BATCH ONLY
    print(f"Creating DataFrame for batch {i+1}...")

    pixel_df_batch = pd.DataFrame(pixel_data_list, columns=pixel_columns)
    metadata_df_batch = pd.DataFrame(processed_metadata_list) # Contains metadata MINUS label

    # Add the original image filename and the LABEL column to the metadata part
    metadata_df_batch['Original_Image_Index'] = original_filenames_list
    # Add the label column here (it will be initially placed based on DataFrame construction)
    metadata_df_batch[label_column_name] = label_list

    # Concatenate the pixel DataFrame and the metadata DataFrame side-by-side for THIS BATCH
    final_df_batch = pd.concat([pixel_df_batch, metadata_df_batch], axis=1)

    # --- Reorder columns to put the SPECIFIED LABEL column last ---
    print(f"Ensuring '{label_column_name}' column is at the end for batch {i+1}.")
    cols = final_df_batch.columns.tolist()
    # Check if label column is already last (might happen depending on concat order)
    if cols[-1] != label_column_name:
      if label_column_name in cols:
          cols.remove(label_column_name)
          cols.append(label_column_name)
          final_df_batch = final_df_batch[cols] # Reindex with new column order
      else:
          # This shouldn't happen due to the earlier check, but good safeguard
          print(f"*** Warning: Label column '{label_column_name}' was not found in the batch DataFrame just before saving. This is unexpected. ***")


    print(f"DataFrame for batch {i+1} created with shape: {final_df_batch.shape}")
    print(f"Final columns for batch {i+1} (last 5): ...{final_df_batch.columns.tolist()[-5:]}") # Show last few columns

    # --- Save/Append Batch Results ---
    try:
        # If it's the first batch (i=0), write with header
        if i == 0:
            final_df_batch.to_csv(output_csv_path, index=False, mode='w', header=True)
            print(f"Successfully saved first batch (with header) to: {output_csv_path}")
        # For subsequent batches, append without header
        else:
            final_df_batch.to_csv(output_csv_path, index=False, mode='a', header=False)
            print(f"Successfully appended batch {i+1} to: {output_csv_path}")
    except Exception as e:
        print(f"\nError saving/appending batch {i+1} data to CSV: {e}")
        print("Stopping script to prevent data loss or corruption.")
        exit() # Stop if saving fails

print("\n--- All Batches Processed ---")
print(f"Final processed data saved in batches to: {output_csv_path}")

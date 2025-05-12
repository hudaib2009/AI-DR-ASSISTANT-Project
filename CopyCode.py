import pandas as pd

input_file = 'Data/Images_Batched.csv'   # Replace with your source CSV filename
output_file = 'CopiedFirst1000Rows.csv'  # Output file name

df = pd.read_csv(input_file, nrows=1000) # Choose how many Rows

df.to_csv(output_file, index=False)

print(f"First 1000 rows copied to {output_file}")

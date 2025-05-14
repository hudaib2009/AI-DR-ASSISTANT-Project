import pandas as pd

ROWS = 7500  # Number of rows to copy
input_file = 'Data/Images_Batched.csv'   # Replace with your source CSV filename
output_file = f'CopiedFirst{ROWS}Rows.csv'  # Output file name

df = pd.read_csv(input_file, nrows=ROWS)

df.to_csv(output_file, index=False)

print(f"First {ROWS} rows copied to {output_file}")

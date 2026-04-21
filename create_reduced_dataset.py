#Reduced files are saved to: Analytes_reduced/
import os
import pandas as pd

# Folder containing the original full-size CSV files
INPUT_DIR = "Analytes"
# Folder where the reduced CSVs will be saved (created if it doesn't exist)
OUTPUT_DIR = "Analytes_reduced"
# Maximum number of data rows to keep per file (not counting the header)
MAX_ROWS = 1000
# Create the output directory if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Collect all CSV files in the input directory
csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

for filename in csv_files:
    input_path  = os.path.join(INPUT_DIR,  filename)
    output_path = os.path.join(OUTPUT_DIR, filename)
    # Read the full CSV into a DataFrame
    df = pd.read_csv(input_path, low_memory=False)
    original_rows = len(df)
    # Keep only the first MAX_ROWS rows
    df_reduced = df.head(MAX_ROWS)
    # Write the reduced DataFrame to the output folder.
    df_reduced.to_csv(output_path, index=False)
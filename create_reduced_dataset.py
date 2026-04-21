"""
create_reduced_dataset.py

This script creates a reduced version of each analyte CSV dataset.
Each output file contains only the first 1,000 data rows (plus the header),
while preserving the exact same file structure and column format as the originals.

Reduced files are saved to: Analytes_reduced/
"""

import os
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

# Folder containing the original full-size CSV files
INPUT_DIR = "Analytes"

# Folder where the reduced CSVs will be saved (created if it doesn't exist)
OUTPUT_DIR = "Analytes_reduced"

# Maximum number of data rows to keep per file (not counting the header)
MAX_ROWS = 1000

# ── Setup ─────────────────────────────────────────────────────────────────────

# Create the output directory if it doesn't already exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Collect all CSV files in the input directory
csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".csv")]

print(f"Found {len(csv_files)} CSV file(s) in '{INPUT_DIR}/'")
print(f"Reducing each to {MAX_ROWS} rows → saving to '{OUTPUT_DIR}/'\n")

# ── Process each file ─────────────────────────────────────────────────────────

for filename in csv_files:
    input_path  = os.path.join(INPUT_DIR,  filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    # Read the full CSV into a DataFrame
    df = pd.read_csv(input_path, low_memory=False)

    original_rows = len(df)

    # Keep only the first MAX_ROWS rows
    df_reduced = df.head(MAX_ROWS)

    # Write the reduced DataFrame to the output folder.
    # index=False ensures the row numbers are not written as an extra column.
    df_reduced.to_csv(output_path, index=False)

    print(f"  {filename:<20}  {original_rows:>6} rows  →  {len(df_reduced)} rows  ✓")

# ── Done ──────────────────────────────────────────────────────────────────────

print(f"\nDone. Reduced files are in '{OUTPUT_DIR}/'")

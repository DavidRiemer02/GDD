import pandas as pd
from tqdm import tqdm
import os
from RandomForest.Utils.cleanCSV import clean_csv

def read_large_data(input_file, output_file_path, sample_size: int = 5000):
    """
    samples `sample_size` rows (or fewer if the file is small), and overwrites the original file.
    
    Args:
        input_file (str): Path to the CSV file.
        output_file_path (str): Path to save the cleaned final file.
        sample_size (int): Number of rows to sample for the final output.
    """
    chunk_size = 1_000_000
    print(f"Reading and processing: {input_file}")
    
    # Estimate total rows for progress bar
    total_rows = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    csv_reader = pd.read_csv(input_file, chunksize=chunk_size, on_bad_lines='skip', engine='python', encoding='utf-8')
    
    final_chunks = []
    for chunk in tqdm(csv_reader, total=total_rows // chunk_size + 1, desc="Reading"):
        final_chunks.append(chunk)
    
    final_data = pd.concat(final_chunks, ignore_index=True)

    if len(final_data) <= sample_size:
        sampled_data = final_data
        print(f"Only {len(final_data)} rows found. Sampling skipped.")
    else:
        sampled_data = final_data.sample(n=sample_size, random_state=42)

    # Overwrite input file with sampled data
    sampled_data.to_csv(input_file, index=False)

    # Clean the overwritten file
    clean_csv(input_file_path=input_file, output_file_path=output_file_path)
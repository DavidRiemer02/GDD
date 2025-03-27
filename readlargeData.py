# Load libraries
import pandas as pd
import random
from tqdm import tqdm

input_file = 'TrainingData/realData/naio_10_fcp_ii1.csv'
output_file = 'results/sample_5000.csv'

# Sampling configuration
chunk_size = 1000000
sample_size = 5000
reservoir = []
total_seen = 0

# Count total rows (optional, only for tqdm progress bar)
total_rows = sum(1 for _ in open(input_file, 'r')) - 1  # exclude header

# Open file in chunks and apply reservoir sampling
csv_reader = pd.read_csv(input_file, chunksize=chunk_size)

for chunk in tqdm(csv_reader, total=total_rows // chunk_size + 1, desc="Sampling"):
    for _, row in chunk.iterrows():
        total_seen += 1
        if len(reservoir) < sample_size:
            reservoir.append(row)
        else:
            r = random.randint(0, total_seen - 1)
            if r < sample_size:
                reservoir[r] = row

# Convert to DataFrame and write to CSV
sample_df = pd.DataFrame(reservoir)
sample_df.to_csv(output_file, index=False)

print(f"✅ Sampled {len(sample_df)} rows to {output_file}")

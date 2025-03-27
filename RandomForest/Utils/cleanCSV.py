import csv
import numpy as np
import time
import os

def clean_csv_quotes(input_file_path, output_file_path, enforce_consistent_columns=True):
    """
    Cleans a CSV file and logs cleaning performance:
    - Detects input delimiter
    - Handles multiple encodings (utf-8-sig, ISO-8859-1, cp1252)
    - Replaces empty/whitespace-only strings with 'NaN'
    - Normalizes quotes and trims fields
    - Ensures consistent column count (optional)
    - Always outputs comma-separated UTF-8 CSV
    - Logs processing time, rows, columns, and file size to 'performance/cleanlog.csv'
    """
    log_file_path = "performance/cleanlog.csv"
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    encodings_to_try = ['utf-8-sig', 'ISO-8859-1', 'cp1252']
    start_time = time.time()

    for encoding in encodings_to_try:
        try:
            with open(input_file_path, 'r', encoding=encoding, newline='') as infile:
                sample = infile.read(2048)
                infile.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=[',', ';', '\t'])
                except csv.Error:
                    dialect = csv.excel
                    dialect.delimiter = ','

                reader = csv.reader(infile, dialect)
                cleaned_rows = []
                max_columns = 0

                for row in reader:
                    cleaned_row = []
                    for field in row:
                        cleaned = field.strip().replace('"', '')  # remove all double quotes, no matter where
                        cleaned = "NaN" if cleaned == "" or cleaned.isspace() else cleaned
                        cleaned_row.append(cleaned)
                    cleaned_rows.append(cleaned_row)
                    max_columns = max(max_columns, len(cleaned_row))

            if enforce_consistent_columns:
                for i, row in enumerate(cleaned_rows):
                    if len(row) < max_columns:
                        cleaned_rows[i] += ["NaN"] * (max_columns - len(row))
                    elif len(row) > max_columns:
                        cleaned_rows[i] = row[:max_columns]

            with open(output_file_path, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONE, escapechar='\\')
                writer.writerows(cleaned_rows)

            duration = round(time.time() - start_time, 4)
            file_size = os.path.getsize(input_file_path)
            num_rows = len(cleaned_rows)
            num_cols = max_columns

            log_entry = [input_file_path, output_file_path, encoding, file_size, num_rows, num_cols, duration]

            # Write log entry
            write_header = not os.path.exists(log_file_path)
            with open(log_file_path, 'a', newline='', encoding='utf-8') as log_file:
                log_writer = csv.writer(log_file)
                if write_header:
                    log_writer.writerow(["InputFile", "OutputFile", "EncodingUsed", "SizeBytes", "Rows", "Columns", "CleanTimeSeconds"])
                log_writer.writerow(log_entry)

            print(f"✅ Cleaned CSV saved to: {output_file_path}")
            return

        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"⚠️ Error processing file with encoding {encoding}: {e}")
            return

    print("❌ Failed to decode the file with known encodings. Please check encoding manually.")

clean_csv_quotes("TrainingData/fakeData/anne_car_data_1.csv", "results/result.csv")
import csv

def clean_csv_quotes(input_file_path, output_file_path):
    """
    Cleans a CSV file:
    - Handles files with unknown encodings (tries utf-8-sig, ISO-8859-1, cp1252)
    - Preserves quoted fields containing commas
    - Strips surrounding quotes and trims whitespace inside fields
    - Writes a well-formatted CSV
    """
    encodings_to_try = ['utf-8-sig', 'ISO-8859-1', 'cp1252']
    
    for encoding in encodings_to_try:
        try:
            cleaned_rows = []

            with open(input_file_path, 'r', encoding=encoding, newline='') as infile:
                reader = csv.reader(infile)
                for row in reader:
                    cleaned_row = [field.strip().replace('""', '"') for field in row]
                    cleaned_rows.append(cleaned_row)

            with open(output_file_path, 'w', encoding='utf-8', newline='') as outfile:
                writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerows(cleaned_rows)

            print(f"✅ Fully cleaned CSV saved to: {output_file_path}")
            return
        except UnicodeDecodeError:
            continue  # Try next encoding
        except Exception as e:
            print(f"⚠️ Error processing CSV with encoding {encoding}: {e}")
            return

    print(f"❌ Failed to decode the file with known encodings. Please check encoding manually.")

import csv
import os

def clean_csv_quotes(input_file_path, output_file_path):
    """
    Cleans a CSV file while ensuring that quoted fields remain intact.
    """
    try:
        with open(input_file_path, 'r', newline='', encoding='utf-8-sig') as infile, \
             open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.reader(infile, quotechar='"', delimiter=',', skipinitialspace=True)
            writer = csv.writer(outfile, quotechar='"', quoting=csv.QUOTE_MINIMAL, delimiter=',')

            for row in reader:
                # Ensure all columns are cleaned and quotes remain intact
                cleaned_row = [col.strip() for col in row]  # Strip spaces
                writer.writerow(cleaned_row)

        print(f"✅ Cleaned CSV saved to: {output_file_path}")

    except Exception as e:
        print(f"❌ Error processing '{input_file_path}': {e}")


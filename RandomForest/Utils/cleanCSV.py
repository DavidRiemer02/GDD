import csv

def clean_csv_quotes(input_file_path, output_file_path):
    """
    Removes all empty quotes "" from a CSV file and writes the cleaned content to a new file.

    Args:
    - input_file_path: str, path to the input CSV file.
    - output_file_path: str, path to the output cleaned CSV file.
    """

    with open(input_file_path, 'r', newline='', encoding='utf-8') as infile, \
         open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            # Remove empty quotes and strip unnecessary spaces
            cleaned_row = [cell.replace('""', '').strip() for cell in row]
            writer.writerow(cleaned_row)

    print(f"Cleaned CSV saved to: {output_file_path}")
import csv
import os
import argparse # Added argparse
import logging # Added logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def combine_csv_files(file1, file2, output_file, file1_has_header=True, file2_has_header=True):
    """
    Combines two CSV files into a single output CSV file.
    Uses the header from the first file if file1_has_header is True.

    Args:
        file1 (str): Path to the first input CSV file.
        file2 (str): Path to the second input CSV file.
        output_file (str): Path to the combined output CSV file.
        file1_has_header (bool): Whether the first file has a header to be used.
        file2_has_header (bool): Whether the second file has a header to be skipped.
    """
    if not os.path.exists(file1):
        logging.error(f"Input file {file1} not found.")
        return False
    if not os.path.exists(file2):
        logging.error(f"Input file {file2} not found.")
        return False

    try:
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            files_processed = 0
            total_rows_written = 0
            header_written = False

            # Process first file
            try:
                with open(file1, 'r', newline='') as infile1:
                    reader1 = csv.reader(infile1)
                    rows_written_file1 = 0
                    if file1_has_header:
                        header = next(reader1) 
                        writer.writerow(header)
                        total_rows_written += 1
                        header_written = True
                    for row in reader1:
                        writer.writerow(row)
                        rows_written_file1 += 1
                    total_rows_written += rows_written_file1
                    logging.info(f"Processed {file1}: Wrote {rows_written_file1} data rows (Header: {file1_has_header}).")
                    files_processed += 1
            except Exception as e:
                logging.error(f"Error processing {file1}: {e}")
                return False

            # Process second file
            try:
                with open(file2, 'r', newline='') as infile2:
                    reader2 = csv.reader(infile2)
                    rows_written_file2 = 0
                    if file2_has_header:
                        if not header_written: # If first file had no header, use this one if available
                            header = next(reader2)
                            writer.writerow(header)
                            total_rows_written +=1
                            header_written = True
                        else: # First file had header, so skip this one
                            next(reader2) 
                    
                    for row in reader2:
                        writer.writerow(row)
                        rows_written_file2 += 1
                    total_rows_written += rows_written_file2
                    logging.info(f"Processed {file2}: Wrote {rows_written_file2} data rows (Header skipped: {file2_has_header and header_written}).")
                    files_processed += 1
            except Exception as e:
                logging.error(f"Error processing {file2}: {e}")
                # Optionally, decide if this is a critical failure or can be skipped

        final_data_rows = total_rows_written - 1 if header_written else total_rows_written
        logging.info(f"Successfully combined {files_processed} files into {output_file}. Total data rows written: {final_data_rows}.")
        return True

    except Exception as e:
        logging.error(f"Error writing to {output_file}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two CSV files.")
    parser.add_argument("--file1", type=str, required=True, help="Path to the first input CSV file (provides header).")
    parser.add_argument("--file2", type=str, required=True, help="Path to the second input CSV file (data to append).")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the combined output CSV file.")
    parser.add_argument("--no_header_file1", action='store_false', dest='file1_has_header', help="Specify if the first file does NOT have a header.")
    parser.add_argument("--no_header_file2", action='store_false', dest='file2_has_header', help="Specify if the second file does NOT have a header (to be skipped).")
    parser.set_defaults(file1_has_header=True, file2_has_header=True)

    args = parser.parse_args()

    # Example default paths if not running from CLI (for direct script run, though CLI is preferred)
    # These would typically be passed via CLI as per argparse setup.
    # file_5k = 'cleaned_data/AP_Data_Comparison_5K_050525_cleaned.csv'
    # generated_file = 'cleaned_data/generated_comparison_data.csv'
    # combined_output_file = 'cleaned_data/combined_comparison_data.csv'

    if not (args.file1 and args.file2 and args.output_file):
        logging.error("File paths for file1, file2, and output_file must be provided.")
    else:
        combine_csv_files(args.file1, args.file2, args.output_file, args.file1_has_header, args.file2_has_header) 
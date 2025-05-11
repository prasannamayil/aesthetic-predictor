import pandas as pd
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_OUTPUT_DIR = 'cleaned_data'

def clean_csv_files(input_files, output_dir):
    """
    Cleans specified CSV files by dropping rows with any NaN values.

    Args:
        input_files (list): List of paths to input CSV files.
        output_dir (str): Directory to save the cleaned CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Starting data cleaning process. Output directory: {output_dir}")

    for input_file in input_files:
        try:
            logging.info(f"Processing file: {input_file}")
            if not os.path.exists(input_file):
                logging.warning(f"  File not found - {input_file}. Skipping.")
                continue

            df = pd.read_csv(input_file)
            original_rows = len(df)
            logging.info(f"  Original number of rows: {original_rows}")

            df_cleaned = df.dropna()
            cleaned_rows = len(df_cleaned)
            logging.info(f"  Number of rows after cleaning: {cleaned_rows}")
            logging.info(f"  Number of rows dropped: {original_rows - cleaned_rows}")

            base_name = os.path.basename(input_file)
            name, ext = os.path.splitext(base_name)
            output_file_path = os.path.join(output_dir, f"{name}_cleaned{ext}")

            df_cleaned.to_csv(output_file_path, index=False)
            logging.info(f"  Cleaned data saved to: {output_file_path}")

        except FileNotFoundError: # Should be caught by os.path.exists, but as a fallback
            logging.error(f"  Error: File not found - {input_file}")
        except Exception as e:
            logging.error(f"  An error occurred while processing {input_file}: {e}")

    logging.info("Data cleaning process finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean CSV files by dropping rows with NaN values.")
    parser.add_argument("--input_files", nargs='+', required=True,
                        help="List of input CSV file paths to clean.")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save cleaned files. Default: {DEFAULT_OUTPUT_DIR}")
    
    args = parser.parse_args()
    clean_csv_files(args.input_files, args.output_dir)
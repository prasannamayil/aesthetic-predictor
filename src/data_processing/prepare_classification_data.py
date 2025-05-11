import argparse
import os
import sys
import logging
import pandas as pd # Added import for pd

# Add src directory to Python path to allow importing download_utils
# This assumes the script is run from the root of the project or src/data_processing directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR) # This should be the 'src' directory
# If script is in src/data_processing, then SRC_DIR is src. Project root is parent of src.
PROJECT_ROOT = os.path.dirname(SRC_DIR)

# Heuristic to find project root if script is run from different locations
# This is a common pattern but might need adjustment based on actual execution context
if os.path.basename(PROJECT_ROOT) == 'aesthetic-predictor': # Or your repo name
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, SRC_DIR) # Ensure src is also in path for `from data_processing import ...`
elif os.path.basename(SRC_DIR) == 'aesthetic-predictor': # if running from src/
    sys.path.insert(0, SRC_DIR)
    PROJECT_ROOT = SRC_DIR # In this case src is the project root
else: # Fallback if structure is unexpected, assumes download_utils is in the same dir or already in path
    pass

from data_processing import download_utils # Changed from download_utils to data_processing.download_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_OUTPUT_ROOT_DIR = 'data' # Will store as data/classification/
DEFAULT_CSV_PATH = 'cleaned_data/AP_Data_Rating_1400_050525_cleaned.csv'
DEFAULT_URL_COL = 'url'
DEFAULT_RATING_COL = 'rating'
DEFAULT_CLASSES = ['good', 'neutral', 'bad']

def main_classification_prep(
    csv_path,
    url_col,
    rating_col,
    classes,
    output_root_dir,
    test_split_size=0.1,
    val_split_size=0.0, # No validation split by default, matching old script
    random_state=42,
    user_agent='ClassificationDataPrep/1.0'
):
    """Main function to prepare data for classification."""
    task_type = 'classification'
    logging.info(f"Starting data preparation for CLASSIFICATION task.")
    logging.info(f"Input CSV: {csv_path}")
    logging.info(f"Output root: {output_root_dir}")
    logging.info(f"Classes: {classes}")

    # Construct the specific output directory for this task type
    # e.g., data/classification/
    task_specific_output_dir = os.path.join(output_root_dir, task_type)
    os.makedirs(task_specific_output_dir, exist_ok=True)

    # 1. Load and Prepare DataFrame
    try:
        df = download_utils.prepare_dataframe(csv_path, url_col, rating_col, classes)
    except Exception as e:
        logging.error(f"Failed to prepare DataFrame: {e}. Aborting.")
        return

    if df.empty:
        logging.error("No data left after filtering by classes or other cleaning. Aborting.")
        return

    # 2. Split Data
    try:
        train_df, val_df, test_df = download_utils.split_data(
            df, 
            test_split_size=test_split_size, 
            val_split_size=val_split_size, 
            stratify_col=rating_col, # Stratify by rating for classification
            random_state=random_state
        )
    except ValueError as e:
        logging.error(f"Failed to split data: {e}. This can happen if classes are too small for stratification. Aborting.")
        return
    
    all_splits_data = {'train': train_df, 'val': val_df, 'test': test_df}
    total_successes = 0
    total_errors = 0

    # 3. Download and Organize for each split
    for split_name, split_df in all_splits_data.items():
        if split_df.empty:
            logging.info(f"Split '{split_name}' is empty. Skipping download.")
            continue

        # e.g., data/classification/train/
        split_output_dir = os.path.join(task_specific_output_dir, split_name)
        # download_utils.download_and_organize_images will create subdirs for classes here
        
        s, e, _ = download_utils.download_and_organize_images(
            df=split_df,
            output_base_dir=split_output_dir, # This dir will contain class subfolders
            url_col=url_col,
            task_type=task_type,
            label_col=rating_col,
            split_name=split_name,
            classes=classes,
            user_agent=user_agent
        )
        total_successes += s
        total_errors += e

    logging.info(f"--- Classification Data Preparation Summary ---")
    logging.info(f"Total images successfully processed: {total_successes}")
    logging.info(f"Total errors encountered: {total_errors}")
    logging.info(f"Processed classification data saved under: {task_specific_output_dir}")
    logging.info("Classification data preparation finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare image data for CLIP classification training.")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH, 
                        help=f"Path to the input CSV file. Default: {DEFAULT_CSV_PATH}")
    parser.add_argument("--url_col", type=str, default=DEFAULT_URL_COL, 
                        help=f"Name of the column containing image URLs. Default: {DEFAULT_URL_COL}")
    parser.add_argument("--rating_col", type=str, default=DEFAULT_RATING_COL, 
                        help=f"Name of the column containing class ratings. Default: {DEFAULT_RATING_COL}")
    parser.add_argument("--classes", nargs='+', default=DEFAULT_CLASSES, 
                        help=f"List of class names. Default: {' '.join(DEFAULT_CLASSES)}")
    parser.add_argument("--output_root_dir", type=str, default=DEFAULT_OUTPUT_ROOT_DIR, 
                        help=f"Base directory to save processed data. Default: {DEFAULT_OUTPUT_ROOT_DIR}")
    parser.add_argument("--test_split_size", type=float, default=0.1, 
                        help="Proportion of data for the test set. Default: 0.1")
    parser.add_argument("--val_split_size", type=float, default=0.0, 
                        help="Proportion of data for the validation set. Default: 0.0")
    parser.add_argument("--random_state", type=int, default=42, 
                        help="Random state for reproducibility. Default: 42")
    parser.add_argument("--user_agent", type=str, default="ClassificationDataPrep/1.0",
                        help="User agent for download requests.")

    args = parser.parse_args()

    if not (0 <= args.test_split_size < 1 and 0 <= args.val_split_size < 1):
        logging.error("Split sizes must be between 0 and 1.")
        sys.exit(1)
    if (args.test_split_size + args.val_split_size) >= 1:
        logging.error("Sum of test and validation split sizes must be less than 1.")
        sys.exit(1)

    main_classification_prep(
        csv_path=args.csv_path,
        url_col=args.url_col,
        rating_col=args.rating_col,
        classes=args.classes,
        output_root_dir=args.output_root_dir,
        test_split_size=args.test_split_size,
        val_split_size=args.val_split_size,
        random_state=args.random_state,
        user_agent=args.user_agent
    ) 
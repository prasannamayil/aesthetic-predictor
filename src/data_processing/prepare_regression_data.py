import argparse
import os
import sys
import logging
import pandas as pd

# Add src directory to Python path (similar to prepare_classification_data.py)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if os.path.basename(PROJECT_ROOT) == 'aesthetic-predictor': # Or your repo name
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, SRC_DIR)
elif os.path.basename(SRC_DIR) == 'aesthetic-predictor':
    sys.path.insert(0, SRC_DIR)
    PROJECT_ROOT = SRC_DIR
else:
    pass

from data_processing import download_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_OUTPUT_ROOT_DIR = 'data'  # Will store as data/regression/
DEFAULT_CSV_PATH = 'url_elo_rankings.csv' # Output from get_elo.py
DEFAULT_URL_COL = 'url'
DEFAULT_SCORE_COL = 'scaled_score_1_10' # Or 'elo_rating' depending on preference


def main_regression_prep(
    csv_path,
    url_col,
    score_col,
    output_root_dir,
    test_split_size=0.1,
    val_split_size=0.1, 
    random_state=42,
    user_agent='RegressionDataPrep/1.0'
):
    """Main function to prepare data for regression task."""
    task_type = 'regression'
    logging.info(f"Starting data preparation for REGRESSION task.")
    logging.info(f"Input CSV: {csv_path}")
    logging.info(f"Output root: {output_root_dir}")
    logging.info(f"Score column: {score_col}")

    task_specific_output_dir = os.path.join(output_root_dir, task_type)
    os.makedirs(task_specific_output_dir, exist_ok=True)

    # 1. Load and Prepare DataFrame
    try:
        # For regression, we don't filter by classes, so `classes` arg is None
        df = download_utils.prepare_dataframe(csv_path, url_col, score_col, classes=None)
    except Exception as e:
        logging.error(f"Failed to prepare DataFrame: {e}. Aborting.")
        return

    if df.empty:
        logging.error("No data left after initial cleaning. Aborting.")
        return
        
    # Ensure score column is numeric for regression
    try:
        df[score_col] = pd.to_numeric(df[score_col])
    except ValueError as e:
        logging.error(f"Score column '{score_col}' could not be converted to numeric: {e}. Aborting.")
        return

    # 2. Split Data
    # For regression, stratification is usually not done on the score directly unless binned.
    # Passing stratify_col=None for now.
    try:
        train_df, val_df, test_df = download_utils.split_data(
            df, 
            test_split_size=test_split_size, 
            val_split_size=val_split_size, 
            stratify_col=None, # No stratification for regression on continuous score
            random_state=random_state
        )
    except ValueError as e:
        logging.error(f"Failed to split data: {e}. Aborting.")
        return
    
    all_splits_data = {'train': train_df, 'val': val_df, 'test': test_df}
    total_successes = 0
    total_errors = 0

    # 3. Download and Organize for each split
    for split_name, split_df in all_splits_data.items():
        if split_df.empty:
            logging.info(f"Split '{split_name}' is empty. Skipping download.")
            continue

        split_output_dir = os.path.join(task_specific_output_dir, split_name)
        os.makedirs(split_output_dir, exist_ok=True) # Base for this split, e.g. data/regression/train/
        
        s, e, metadata = download_utils.download_and_organize_images(
            df=split_df,
            output_base_dir=split_output_dir, # Images will go into split_output_dir/images/
            url_col=url_col,
            task_type=task_type,
            label_col=score_col, # This will be the 'score' in metadata
            split_name=split_name,
            classes=None, # Not used for regression
            user_agent=user_agent
        )
        total_successes += s
        total_errors += e

        if metadata:
            meta_df = pd.DataFrame(metadata)
            # Save metadata CSV directly in the split_output_dir (e.g., data/regression/train/train_metadata.csv)
            meta_csv_path = os.path.join(split_output_dir, f"{split_name}_metadata.csv")
            try:
                meta_df.to_csv(meta_csv_path, index=False)
                logging.info(f"Regression metadata for {split_name} saved to {meta_csv_path}")
            except Exception as csv_err:
                logging.error(f"Failed to save metadata CSV for {split_name}: {csv_err}")

    logging.info(f"--- Regression Data Preparation Summary ---")
    logging.info(f"Total images successfully processed: {total_successes}")
    logging.info(f"Total errors encountered: {total_errors}")
    logging.info(f"Processed regression data saved under: {task_specific_output_dir}")
    logging.info("Regression data preparation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare image data for CLIP regression training.")
    parser.add_argument("--csv_path", type=str, default=DEFAULT_CSV_PATH, 
                        help=f"Path to the input CSV file with URLs and scores. Default: {DEFAULT_CSV_PATH}")
    parser.add_argument("--url_col", type=str, default=DEFAULT_URL_COL, 
                        help=f"Name of the column containing image URLs. Default: {DEFAULT_URL_COL}")
    parser.add_argument("--score_col", type=str, default=DEFAULT_SCORE_COL, 
                        help=f"Name of the column containing numerical scores. Default: {DEFAULT_SCORE_COL}")
    parser.add_argument("--output_root_dir", type=str, default=DEFAULT_OUTPUT_ROOT_DIR, 
                        help=f"Base directory to save processed data. Default: {DEFAULT_OUTPUT_ROOT_DIR}")
    parser.add_argument("--test_split_size", type=float, default=0.1, 
                        help="Proportion of data for the test set. Default: 0.1")
    parser.add_argument("--val_split_size", type=float, default=0.1, 
                        help="Proportion of data for the validation set. Default: 0.1")
    parser.add_argument("--random_state", type=int, default=42, 
                        help="Random state for reproducibility. Default: 42")
    parser.add_argument("--user_agent", type=str, default="RegressionDataPrep/1.0",
                        help="User agent for download requests.")

    args = parser.parse_args()
    
    if not (0 <= args.test_split_size < 1 and 0 <= args.val_split_size < 1):
        logging.error("Split sizes must be between 0 and 1.")
        sys.exit(1)
    if (args.test_split_size + args.val_split_size) >= 1:
        logging.error("Sum of test and validation split sizes must be less than 1.")
        sys.exit(1)

    main_regression_prep(
        csv_path=args.csv_path,
        url_col=args.url_col,
        score_col=args.score_col,
        output_root_dir=args.output_root_dir,
        test_split_size=args.test_split_size,
        val_split_size=args.val_split_size,
        random_state=args.random_state,
        user_agent=args.user_agent
    ) 
import pandas as pd
import requests
import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import hashlib
from PIL import Image
import io
import logging
from urllib.parse import urlparse

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def generate_safe_filename(url, prefix="", suffix_from_url=True, hash_len=16):
    """
    Generates a unique and safe filename from a URL.
    Uses SHA256 hash of the URL for uniqueness and filesystem safety.
    Optionally, tries to append the original extension or defaults to .jpg.

    Args:
        url (str): The URL to generate a filename for.
        prefix (str): Optional prefix for the filename.
        suffix_from_url (bool): If True, try to get extension from URL.
        hash_len (int): Length of the hash to use in the filename.

    Returns:
        str: A safe filename.
    """
    if not url:
        return f"{prefix}invalid_url_{hashlib.sha256(os.urandom(16)).hexdigest()[:hash_len]}.jpg"

    url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
    filename_core = f"{prefix}{url_hash[:hash_len]}"

    if suffix_from_url:
        try:
            parsed_url = urlparse(url)
            original_filename = os.path.basename(parsed_url.path)
            _, ext = os.path.splitext(original_filename)
            if ext.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']:
                return f"{filename_core}{ext.lower()}"
            else:
                return f"{filename_core}.jpg"  # Default to .jpg
        except Exception:
            return f"{filename_core}.jpg"  # Fallback
    return f"{filename_core}.jpg"


def download_and_verify_image(url, target_dir, filename, timeout=20, user_agent=None):
    """
    Downloads an image from a URL, verifies it, and saves it.

    Args:
        url (str): The URL of the image.
        target_dir (str): Directory to save the image.
        filename (str): Filename to save the image as.
        timeout (int): Request timeout in seconds.
        user_agent (str, optional): User agent string for the request.

    Returns:
        str or None: The full path to the saved image if successful, None otherwise.
    """
    filepath = os.path.join(target_dir, filename)
    if os.path.exists(filepath):
        logging.debug(f"File already exists: {filepath}. Skipping download.")
        # Optional: Verify existing file
        try:
            img = Image.open(filepath)
            img.verify()
            return filepath # Assume valid if it opens and verifies
        except (IOError, SyntaxError, Image.UnidentifiedImageError) as e:
            logging.warning(f"Existing file {filepath} is corrupted: {e}. Attempting re-download.")
            try:
                os.remove(filepath)
            except OSError as remove_err:
                logging.error(f"Could not remove corrupted existing file {filepath}: {remove_err}")
                return None # Cannot proceed if corrupted file cannot be removed


    headers = {'User-Agent': user_agent} if user_agent else {}

    try:
        response = requests.get(url, stream=True, timeout=timeout, headers=headers)
        response.raise_for_status()

        content_type = response.headers.get('content-type')
        if not content_type or not content_type.lower().startswith('image/'):
            logging.warning(f"URL content type is not image ({content_type}) for {url}. Attempting to save anyway.")

        image_data_bytes = response.content
        
        # Validate image data with PIL before saving
        try:
            img = Image.open(io.BytesIO(image_data_bytes))
            img.verify()  # Verify integrity
            # To save, we need to reopen the image after verify or use the original bytes
            # We will save directly from bytes after verification
        except (IOError, SyntaxError, Image.UnidentifiedImageError) as img_err:
            logging.warning(f"Failed to validate image data from {url}. Error: {img_err}")
            return None

        os.makedirs(target_dir, exist_ok=True)
        with open(filepath, 'wb') as f:
            f.write(image_data_bytes)
        
        # Final check on the saved file
        try:
            final_img = Image.open(filepath)
            final_img.verify()
        except (IOError, SyntaxError, Image.UnidentifiedImageError) as final_img_err:
            logging.error(f"Saved file {filepath} from {url} is corrupted: {final_img_err}. Deleting.")
            try:
                os.remove(filepath)
            except OSError as os_err:
                logging.error(f"Could not remove corrupted saved file {filepath}: {os_err}")
            return None
            
        return filepath

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {url}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error downloading {url}: {e}")
        return None


def prepare_dataframe(csv_path, url_col, label_col=None, classes=None):
    """
    Loads and prepares the initial DataFrame from a CSV file.
    Handles missing values and filters by classes if provided.
    """
    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(df)} rows from {csv_path}")
    except FileNotFoundError:
        logging.error(f"CSV file not found at {csv_path}")
        raise
    except Exception as e:
        logging.error(f"Error reading CSV file {csv_path}: {e}")
        raise

    required_cols = [url_col]
    if label_col:
        required_cols.append(label_col)

    for col in required_cols:
        if col not in df.columns:
            logging.error(f"Required column '{col}' not found in {csv_path}.")
            raise ValueError(f"Missing required column: {col}")

    df.dropna(subset=[url_col], inplace=True) # URLs are essential
    if label_col:
        df.dropna(subset=[label_col], inplace=True)
        if classes: # For classification task
            df[label_col] = df[label_col].astype(str).str.lower()
            original_count = len(df)
            df = df[df[label_col].isin(classes)]
            filtered_count = len(df)
            if original_count != filtered_count:
                logging.warning(f"Filtered out {original_count - filtered_count} rows with labels not in {classes}.")
    
    if df.empty:
        logging.error("DataFrame is empty after initial preparation. No valid data to process.")
        raise ValueError("No data to process after initial cleaning and filtering.")
        
    logging.info(f"Prepared DataFrame with {len(df)} valid rows.")
    return df

def split_data(df, test_split_size=0.1, val_split_size=0.0, stratify_col=None, random_state=42):
    """
    Splits DataFrame into train, validation, and test sets.
    """
    if not (0 <= test_split_size < 1 and 0 <= val_split_size < 1):
        raise ValueError("Split sizes must be between 0 and 1.")
    if (test_split_size + val_split_size) >= 1:
        raise ValueError("Sum of test and validation split sizes must be less than 1.")

    splits = {}
    remaining_df = df.copy()

    if test_split_size > 0:
        train_val_df, test_df = train_test_split(
            remaining_df,
            test_size=test_split_size,
            stratify=remaining_df[stratify_col] if stratify_col and stratify_col in remaining_df else None,
            random_state=random_state
        )
        splits['test'] = test_df
        remaining_df = train_val_df
    else:
        splits['test'] = pd.DataFrame(columns=df.columns) # Empty DF if no test split

    if val_split_size > 0 and not remaining_df.empty:
        # Calculate validation split size relative to the remaining data
        if (1 - test_split_size) == 0: # Avoid division by zero if test_split is 1 (though disallowed)
             relative_val_split = 0
        else:
            relative_val_split = val_split_size / (1 - test_split_size)
        
        if relative_val_split >= 1.0 and len(remaining_df) > 0: # If val_split makes it take all remaining data
            logging.warning("Validation split size is too large for the remaining data. Assigning all remaining to validation.")
            splits['val'] = remaining_df
            splits['train'] = pd.DataFrame(columns=df.columns) # Empty train
            remaining_df = pd.DataFrame(columns=df.columns)
        elif relative_val_split > 0 and len(remaining_df) > 1 : # Need at least 2 samples to split for train and val
            train_df, val_df = train_test_split(
                remaining_df,
                test_size=relative_val_split,
                stratify=remaining_df[stratify_col] if stratify_col and stratify_col in remaining_df else None,
                random_state=random_state
            )
            splits['train'] = train_df
            splits['val'] = val_df
        else: # Not enough data for validation split or val_split_size is 0
            splits['train'] = remaining_df
            splits['val'] = pd.DataFrame(columns=df.columns) # Empty DF
    else: # No validation split needed or no data left for it
        splits['train'] = remaining_df
        splits['val'] = pd.DataFrame(columns=df.columns) # Empty DF

    logging.info(f"Data split: Train ({len(splits['train'])}), Validation ({len(splits['val'])}), Test ({len(splits['test'])}).")
    return splits['train'], splits['val'], splits['test']


def download_and_organize_images(
    df,
    output_base_dir,
    url_col,
    task_type, # 'classification' or 'regression'
    label_col=None, # Required for classification, used for score in regression for metadata
    split_name='train', # 'train', 'val', or 'test'
    classes=None, # List of class names for classification (e.g. ['good', 'bad'])
    user_agent='Mozilla/5.0',
    skip_existing=True
):
    """
    Downloads images from a DataFrame subset (train, val, or test) and organizes them.

    For 'classification': organizes into subfolders per class.
    For 'regression': saves all images into one folder and creates a metadata CSV.

    Args:
        df (pd.DataFrame): DataFrame for the current split (e.g., train_df).
        output_base_dir (str): Base directory for this split's data (e.g., 'data/processed/train').
        url_col (str): Name of the column containing image URLs.
        task_type (str): 'classification' or 'regression'.
        label_col (str, optional): Column with class labels or scores.
        split_name (str): Name of the current split (e.g., 'train', 'val', 'test').
        classes (list, optional): List of class names (for classification).
        user_agent (str): User agent for downloads.
        skip_existing (bool): If True, skip downloading if file exists. (Note: download_and_verify_image has its own existence check)

    Returns:
        tuple: (number_of_successful_downloads, number_of_errors, list_of_metadata_for_regression)
    """
    success_count = 0
    error_count = 0
    metadata_list = [] # For regression task

    if df.empty:
        logging.info(f"No images to process for {split_name} set.")
        return 0, 0, []

    logging.info(f"--- Processing {split_name} set ({len(df)} images) ---")

    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"Downloading {split_name}"):
        url = row[url_col]
        label = row[label_col] if label_col and label_col in row else None
        
        original_filename_prefix = f"{split_name}_{index}_" # More descriptive prefix
        filename = generate_safe_filename(url, prefix=original_filename_prefix)

        if task_type == 'classification':
            if not label or label not in classes:
                logging.warning(f"Skipping image due to missing or invalid class label: {label} for URL {url}")
                error_count +=1
                continue
            target_dir = os.path.join(output_base_dir, str(label))
        elif task_type == 'regression':
            target_dir = os.path.join(output_base_dir, "images") # All images in one subfolder
        else:
            logging.error(f"Invalid task_type: {task_type}. Must be 'classification' or 'regression'.")
            error_count += df.shape[0] # Count all as errors for this call
            return 0, error_count, [] # Early exit

        os.makedirs(target_dir, exist_ok=True)
        
        saved_image_path = download_and_verify_image(url, target_dir, filename, user_agent=user_agent)

        if saved_image_path:
            success_count += 1
            if task_type == 'regression':
                # Store path relative to the 'images' subdirectory of the split specific dir
                relative_image_path = os.path.join("images", filename)
                metadata_list.append({'image_path': relative_image_path, 'score': label, 'original_url': url})
        else:
            error_count += 1
            logging.debug(f"Failed to download or verify image from URL: {url}")

    if error_count > 0:
        logging.warning(f"Finished {split_name} set with {error_count} download/verification errors.")
    else:
        logging.info(f"Finished {split_name} set successfully. Downloaded {success_count} images.")
    
    return success_count, error_count, metadata_list

# Example of how this module might be called (to be put in separate scripts)
# def main_process_data(
#     csv_path, url_col, output_root_dir, task_type,
#     label_col=None, classes=None, # Classification specific
#     test_split_size=0.1, val_split_size=0.1, # Split sizes
#     random_state=42, user_agent='DefaultImageDownloader/1.0'
# ):
#     logging.info(f"Starting data processing for task: {task_type}")
#     logging.info(f"Output directory: {output_root_dir}")

#     # 1. Load and Prepare DataFrame
#     try:
#         df = prepare_dataframe(csv_path, url_col, label_col, classes if task_type == 'classification' else None)
#     except Exception as e:
#         logging.error(f"Failed to prepare DataFrame: {e}. Aborting.")
#         return

#     # 2. Split Data
#     stratify_col = label_col if task_type == 'classification' and label_col else None
#     try:
#         train_df, val_df, test_df = split_data(df, test_split_size, val_split_size, stratify_col, random_state)
#     except Exception as e:
#         logging.error(f"Failed to split data: {e}. Aborting.")
#         return
    
#     all_splits_data = {'train': train_df, 'val': val_df, 'test': test_df}
#     total_successes = 0
#     total_errors = 0

#     # 3. Download and Organize for each split
#     for split_name, split_df in all_splits_data.items():
#         if split_df.empty:
#             logging.info(f"Split '{split_name}' is empty. Skipping download.")
#             continue

#         split_output_dir = os.path.join(output_root_dir, task_type, split_name)
#         os.makedirs(split_output_dir, exist_ok=True)
        
#         s, e, metadata = download_and_organize_images(
#             df=split_df,
#             output_base_dir=split_output_dir,
#             url_col=url_col,
#             task_type=task_type,
#             label_col=label_col,
#             split_name=split_name,
#             classes=classes if task_type == 'classification' else None,
#             user_agent=user_agent
#         )
#         total_successes += s
#         total_errors += e

#         if task_type == 'regression' and metadata:
#             meta_df = pd.DataFrame(metadata)
#             meta_csv_path = os.path.join(split_output_dir, f"{split_name}_metadata.csv")
#             try:
#                 meta_df.to_csv(meta_csv_path, index=False)
#                 logging.info(f"Regression metadata for {split_name} saved to {meta_csv_path}")
#             except Exception as csv_err:
#                 logging.error(f"Failed to save metadata CSV for {split_name}: {csv_err}")

#     logging.info(f"--- Overall Summary ---")
#     logging.info(f"Total images successfully processed: {total_successes}")
#     logging.info(f"Total errors encountered: {total_errors}")
#     logging.info(f"Processed data saved under: {os.path.join(output_root_dir, task_type)}")
#     logging.info("Data processing finished.") 
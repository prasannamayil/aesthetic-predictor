import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import clip
import os
from PIL import Image, ImageDraw, ImageFont
import argparse
import pandas as pd
import random
import math
from tqdm import tqdm
import sys
import logging # Added logging

# Add src directory to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR) # This is 'src'
PROJECT_ROOT = os.path.dirname(SRC_DIR) # This is project root

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CLIP regression model.")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Path to the saved model checkpoint (.pth file), e.g., results/checkpoints/clip_regressor_linear.pth")
    parser.add_argument("--data_root_dir", type=str, default="data",
                        help="Root directory for processed data (e.g., 'data/').")
    parser.add_argument("--task_name", type=str, default="regression",
                        help="Task name, used for subfolder in data_root_dir (e.g., 'regression').")
    parser.add_argument("--split_name", type=str, default="test", choices=["train", "val", "test"],
                        help="Which data split to evaluate (train, val, or test).")
    parser.add_argument("--clip_model_name", type=str, default="ViT-L/14", 
                        help="CLIP model architecture used during training.")
    parser.add_argument("--was_full_model_finetuned", action='store_true', 
                        help="Set if the checkpoint is for a full fine-tuned CLIP model.")
    parser.add_argument("--score_col_name", type=str, default="score",
                        help="Name of the score column in the metadata CSV.")
    parser.add_argument("--num_samples_visualize", type=int, default=25, 
                        help="Number of random samples to include in the visualization grid. Set to 0 to disable visualization.")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Base directory to save evaluation outputs (plots, CSVs).")
    parser.add_argument("--output_plot_name", type=str, default="evaluation_plot.png", 
                        help="Filename for the composite output image. Will be saved in results_dir/plots/.")
    parser.add_argument("--output_csv_name", type=str, default="evaluation_predictions.csv",
                        help="Filename for the CSV with all predictions. Will be saved in results_dir/evaluation_outputs/.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--image_size_viz", type=int, default=224, 
                        help="Size to resize images to for the output visualization grid.")
    parser.add_argument("--font_size_viz", type=int, default=14, 
                        help="Font size for score labels in visualization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    # Construct full paths for data and outputs
    args.eval_data_dir = os.path.join(args.data_root_dir, args.task_name, args.split_name)
    args.output_plot_dir = os.path.join(args.results_dir, "plots")
    args.output_csv_dir = os.path.join(args.results_dir, "evaluation_outputs")
    os.makedirs(args.output_plot_dir, exist_ok=True)
    os.makedirs(args.output_csv_dir, exist_ok=True)
    args.full_output_plot_path = os.path.join(args.output_plot_dir, args.output_plot_name)
    args.full_output_csv_path = os.path.join(args.output_csv_dir, args.output_csv_name)

    return args

# --- Custom Dataset (adapted from train_regressor.py) ---
class EvalImageScoreDataset(Dataset):
    def __init__(self, split_dir, score_col_name, split_name='test'): # No transform needed here
        self.split_dir = split_dir
        self.image_base_dir = os.path.join(split_dir, "images")
        metadata_csv_path = os.path.join(split_dir, f"{split_name}_metadata.csv")
        self.score_col_name = score_col_name

        try:
            self.metadata = pd.read_csv(metadata_csv_path)
            if 'image_path' not in self.metadata.columns or self.score_col_name not in self.metadata.columns:
                logging.error(f"Eval Metadata CSV {metadata_csv_path} must contain 'image_path' and '{self.score_col_name}'.")
                self.metadata = pd.DataFrame()
                return
            self.metadata[self.score_col_name] = pd.to_numeric(self.metadata[self.score_col_name], errors='coerce')
            self.metadata.dropna(subset=[self.score_col_name, 'image_path'], inplace=True)
        except FileNotFoundError:
            logging.error(f"Eval Metadata CSV not found: {metadata_csv_path}")
            self.metadata = pd.DataFrame()
        except Exception as e:
            logging.error(f"Error reading eval metadata CSV {metadata_csv_path}: {e}")
            self.metadata = pd.DataFrame()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()
        try:
            row = self.metadata.iloc[idx]
            img_rel_path = row['image_path']
            full_img_path = os.path.join(self.split_dir, img_rel_path)
            if not os.path.exists(full_img_path):
                 full_img_path_fallback = os.path.join(self.image_base_dir, os.path.basename(img_rel_path))
                 if os.path.exists(full_img_path_fallback): full_img_path = full_img_path_fallback
                 else: logging.warning(f"Image not found: {full_img_path} or {full_img_path_fallback}. Skipping item."); return None
            
            score = float(row[self.score_col_name])
            return full_img_path, score # Return path for direct loading, and score
        except Exception as e:
            logging.warning(f"Error loading item at index {idx} from {self.split_dir}: {e}. Skipping.")
            return None

# --- Model Definition (for loading state dict) ---
class CLIPWithRegressor(nn.Module):
    def __init__(self, clip_model, regressor_head):
        super().__init__()
        self.clip_model = clip_model
        self.regressor = regressor_head
    def forward(self, images):
        image_features = self.clip_model.encode_image(images).float()
        return self.regressor(image_features).squeeze(-1)

# --- Main Evaluation Script ---
if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {DEVICE}")

    # --- Load CLIP Model Base and Preprocessor ---
    logging.info(f"Loading base CLIP model: {args.clip_model_name}")
    try:
        model_base, preprocess = clip.load(args.clip_model_name, device='cpu')
        model_base.eval()
    except Exception as e: logging.error(f"Error loading base CLIP model: {e}"); exit(1)

    # --- Prepare Model for Inference ---
    feature_dim = model_base.visual.output_dim
    regression_head = nn.Linear(feature_dim, 1)
    model_eval_display_name = ""

    if args.was_full_model_finetuned:
        logging.info("Loading full fine-tuned model state from checkpoint...")
        model_eval = CLIPWithRegressor(model_base, regression_head)
        model_eval_display_name = "CLIPWithRegressor (Fine-tuned)"
    else:
        logging.info("Loading linear regression head state from checkpoint...")
        model_eval = regression_head # Base model used separately for feature extraction
        model_eval_display_name = "RegressionHead (Linear Probe)"

    try:
        logging.info(f"Loading checkpoint: {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location='cpu')
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        if any(k.startswith('module.') for k in state_dict.keys()):
            logging.info("Removed 'module.' prefix from DataParallel checkpoint.")
        model_eval.load_state_dict(new_state_dict)
        logging.info(f"Successfully loaded state_dict into {model_eval_display_name}.")
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found: {args.checkpoint_path}"); exit(1)
    except Exception as e:
        logging.error(f"Error loading model state_dict: {e}"); exit(1)

    model_eval.to(DEVICE).eval()
    if not args.was_full_model_finetuned:
        model_base.to(DEVICE).eval() # Ensure base CLIP is also on device and eval for linear probe

    # --- Load Test Data ---
    logging.info(f"Loading evaluation data from: {args.eval_data_dir} (split: {args.split_name})")
    eval_dataset = EvalImageScoreDataset(args.eval_data_dir, args.score_col_name, split_name=args.split_name)

    if len(eval_dataset) == 0:
        logging.error(f"No data loaded from {args.eval_data_dir}. Exiting."); exit(1)
    logging.info(f"Loaded {len(eval_dataset)} samples for evaluation.")

    # --- Get Predictions for ALL samples ---
    all_predictions = []
    # Use a DataLoader for batching if dataset is large, for simplicity direct iteration for now
    # Could wrap eval_dataset with a Dataloader if needed for batching predictions
    logging.info("Generating predictions for all evaluation samples...")
    with torch.no_grad():
        for i in tqdm(range(len(eval_dataset)), desc="Predicting all samples"):
            item = eval_dataset[i]
            if item is None: continue
            img_path, actual_score = item
            try:
                image = Image.open(img_path).convert('RGB')
                processed_image = preprocess(image).unsqueeze(0).to(DEVICE)
                if args.was_full_model_finetuned:
                    predicted_score = model_eval(processed_image).item()
                else:
                    image_features = model_base.encode_image(processed_image).float()
                    predicted_score = model_eval(image_features).item()
                all_predictions.append({
                    'image_path': os.path.relpath(img_path, PROJECT_ROOT), # Store relative path
                    'actual_score': actual_score,
                    'predicted_score': predicted_score
                })
            except Exception as e:
                logging.warning(f"Error processing {img_path} for prediction: {e}. Skipping.")

    if not all_predictions:
        logging.error("No predictions were generated. Exiting."); exit(1)
    
    predictions_df = pd.DataFrame(all_predictions)
    try:
        predictions_df.to_csv(args.full_output_csv_path, index=False)
        logging.info(f"All predictions saved to: {args.full_output_csv_path}")
    except Exception as e:
        logging.error(f"Error saving predictions CSV: {e}")

    # Calculate overall metrics (e.g., MSE, MAE, Correlation)
    if 'actual_score' in predictions_df.columns and 'predicted_score' in predictions_df.columns:
        actuals = predictions_df['actual_score']
        preds = predictions_df['predicted_score']
        mse = ((actuals - preds) ** 2).mean()
        mae = (actuals - preds).abs().mean()
        correlation = actuals.corr(preds)
        logging.info(f"Overall Metrics: MSE={mse:.4f}, MAE={mae:.4f}, Correlation={correlation:.4f}")
    else:
        logging.warning("Could not calculate overall metrics due to missing score columns.")

    # --- Create Composite Image for Visualization (if num_samples_visualize > 0) ---
    if args.num_samples_visualize > 0 and not predictions_df.empty:
        logging.info(f"Preparing visualization for {args.num_samples_visualize} samples...")
        num_to_viz = min(args.num_samples_visualize, len(predictions_df))
        
        # Sample from the predictions_df for visualization
        # Ensure 'image_path' is absolute for Image.open by joining with PROJECT_ROOT if it was made relative
        viz_df = predictions_df.sample(n=num_to_viz, random_state=args.seed)

        cols = math.ceil(math.sqrt(num_to_viz * 1.0)) # Adjusted for better layout
        rows = math.ceil(num_to_viz / cols)
        img_w, img_h = args.image_size_viz, args.image_size_viz
        text_h_viz = args.font_size_viz + 10
        total_w = cols * img_w
        total_h = rows * (img_h + text_h_viz)
        composite_img = Image.new('RGB', (total_w, total_h), color='white')
        draw = ImageDraw.Draw(composite_img)
        try: font = ImageFont.truetype("arial.ttf", args.font_size_viz)
        except IOError: logging.warning("Arial font not found. Using default PIL font."); font = ImageFont.load_default()

        current_x, current_y = 0, 0
        for i, row_data in viz_df.iterrows():
            # Construct absolute path if image_path in df is relative to project root
            abs_img_path = os.path.join(PROJECT_ROOT, row_data['image_path']) if not os.path.isabs(row_data['image_path']) else row_data['image_path']
            try:
                img = Image.open(abs_img_path).convert('RGB').resize((img_w, img_h))
                composite_img.paste(img, (current_x, current_y))
                text = f"P: {row_data['predicted_score']:.2f} / A: {row_data['actual_score']:.2f}"
                draw.text((current_x + 5, current_y + img_h + 2), text, fill='black', font=font)
            except Exception as e:
                logging.warning(f"Error processing image {abs_img_path} for composite: {e}")
                draw.rectangle([current_x, current_y, current_x+img_w, current_y+img_h], outline="red", fill="lightgray")
                draw.text((current_x+5, current_y+5), "Error", fill='red', font=font)
            
            current_x += img_w
            if (i + 1) % cols == 0: # Note: iterrows index might not be sequential from 0 if sampled
                                     # This logic needs to be based on count of processed items for viz
                current_x = 0
                current_y += img_h + text_h_viz
        
        # Adjusting the loop for grid positioning based on item count for visualization
        # Re-doing paste logic for robust grid layout
        composite_img = Image.new('RGB', (total_w, total_h), color='white') # Re-init for clean paste
        draw = ImageDraw.Draw(composite_img)
        for idx, (_, row_data) in enumerate(viz_df.iterrows()):
            abs_img_path = os.path.join(PROJECT_ROOT, row_data['image_path']) if not os.path.isabs(row_data['image_path']) else row_data['image_path']
            grid_x = (idx % cols) * img_w
            grid_y = (idx // cols) * (img_h + text_h_viz)
            try:
                img = Image.open(abs_img_path).convert('RGB').resize((img_w, img_h))
                composite_img.paste(img, (grid_x, grid_y))
                text = f"P: {row_data['predicted_score']:.2f} / A: {row_data['actual_score']:.2f}"
                draw.text((grid_x + 5, grid_y + img_h + 2), text, fill='black', font=font)
            except Exception as e:
                logging.warning(f"Error processing image {abs_img_path} for composite: {e}")
                draw.rectangle([grid_x, grid_y, grid_x+img_w, grid_y+img_h], outline="red", fill="lightgray")
                draw.text((grid_x+5, grid_y+5), "Error", fill='red', font=font)

        try:
            composite_img.save(args.full_output_plot_path)
            logging.info(f"Composite visualization saved to: {args.full_output_plot_path}")
        except Exception as e: logging.error(f"Error saving composite image: {e}")
    elif args.num_samples_visualize == 0:
        logging.info("Visualization disabled as per --num_samples_visualize=0.")
    else:
        logging.info("Skipping visualization as no prediction results are available.")

    logging.info("Evaluation finished.") 
import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import datasets, transforms # Not needed for custom dataset
from torch.utils.data import DataLoader, Dataset # Added Dataset
import clip # Use openai/clip library
import os
from tqdm.auto import tqdm
from PIL import Image
import argparse
import pandas as pd # Added pandas for reading metadata CSV
import sys # Added sys

# Add src directory to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR) # This is 'src'
PROJECT_ROOT = os.path.dirname(SRC_DIR) # This is project root

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# from utils import some_utility # Example

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear readout or fine-tune CLIP for image score regression.")
    parser.add_argument("--data_root_dir", type=str, default="data",
                        help="Root directory for processed data (e.g., 'data/', which contains 'data/regression/').")
    parser.add_argument("--task_name", type=str, default="regression", 
                        help="Task name, used for subfolder in data_root_dir.")
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints",
                        help="Directory to save trained model checkpoints.")
    parser.add_argument("--checkpoint_name", type=str, default="clip_regressor.pth",
                        help="Filename for the saved model checkpoint.")
    parser.add_argument("--clip_model_name", type=str, default="ViT-L/14", 
                        help="CLIP model architecture (e.g., ViT-B/32, ViT-L/14).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate. Use smaller (e.g., 1e-5) if fine-tuning full model.")
    parser.add_argument("--finetune_full_model", action='store_true', 
                        help="Fine-tune entire CLIP model instead of just the regression head.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers.")
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "l1"], 
                        help="Regression loss function (mse or l1).")
    parser.add_argument("--score_col_name", type=str, default="score", 
                        help="Name of the score column in metadata CSVs (e.g., 'score', 'scaled_score_1_10').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    if args.finetune_full_model and args.learning_rate > 1e-4:
        print(f"Warning: Fine-tuning full model with LR {args.learning_rate}. Consider smaller LR.")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    base, ext = os.path.splitext(args.checkpoint_name)
    mode_suffix = "_finetuned" if args.finetune_full_model else "_linear"
    args.full_checkpoint_save_path = os.path.join(args.checkpoint_dir, f"{base}{mode_suffix}{ext}")

    # Data directories based on new structure for regression task
    # e.g. data/regression/train/, data/regression/val/
    args.train_split_dir = os.path.join(args.data_root_dir, args.task_name, "train")
    args.val_split_dir = os.path.join(args.data_root_dir, args.task_name, "val")
    args.test_split_dir = os.path.join(args.data_root_dir, args.task_name, "test") # For potential future use

    return args


# --- Custom Dataset for Regression ---
class ImageScoreDataset(Dataset):
    def __init__(self, split_dir, score_col_name, transform=None, split_name='train'): # Added split_name for clarity
        """
        Args:
            split_dir (string): Path to the split directory (e.g., data/regression/train).
                                This directory should contain an 'images' subfolder and a metadata CSV.
            score_col_name (string): Name of the column in metadata CSV that contains scores.
            transform (callable, optional): Optional transform.
            split_name (string): Name of the split ('train', 'val', 'test') for logging and metadata filename.
        """
        self.split_dir = split_dir
        self.image_base_dir = os.path.join(split_dir, "images")
        metadata_csv_path = os.path.join(split_dir, f"{split_name}_metadata.csv")
        self.transform = transform
        self.score_col_name = score_col_name

        try:
            self.metadata = pd.read_csv(metadata_csv_path)
            if 'image_path' not in self.metadata.columns or self.score_col_name not in self.metadata.columns:
                logging.error(f"Metadata CSV {metadata_csv_path} must contain 'image_path' and '{self.score_col_name}' columns.")
                self.metadata = pd.DataFrame()
                return

            self.metadata[self.score_col_name] = pd.to_numeric(self.metadata[self.score_col_name], errors='coerce')
            original_len = len(self.metadata)
            self.metadata.dropna(subset=[self.score_col_name, 'image_path'], inplace=True)
            if len(self.metadata) < original_len:
                logging.warning(f"Dropped {original_len - len(self.metadata)} rows from {metadata_csv_path} due to NaNs in image_path or non-numeric scores.")
        
        except FileNotFoundError:
            logging.error(f"Metadata CSV not found: {metadata_csv_path}")
            self.metadata = pd.DataFrame()
        except Exception as e:
            logging.error(f"Error reading metadata CSV {metadata_csv_path}: {e}")
            self.metadata = pd.DataFrame()

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx): idx = idx.tolist()

        img_rel_path = 'unknown'
        try:
            row = self.metadata.iloc[idx]
            img_rel_path = row['image_path'] # This path is relative to self.image_base_dir parent usually, e.g. "images/filename.jpg"
                                           # The download_utils saves it as os.path.join("images", filename)
            
            # image_path from metadata should be like "images/xyz.jpg"
            # So, we join split_dir with this relative path. self.image_base_dir is split_dir/images
            # Therefore, the image full path is os.path.join(self.split_dir, img_rel_path)
            full_img_path = os.path.join(self.split_dir, img_rel_path)
            
            if not os.path.exists(full_img_path):
                 # Fallback for older metadata that might just have filename, not images/filename.jpg
                 full_img_path_fallback = os.path.join(self.image_base_dir, os.path.basename(img_rel_path))
                 if os.path.exists(full_img_path_fallback):
                     full_img_path = full_img_path_fallback
                 else:
                    logging.warning(f"Image file not found at {full_img_path} or {full_img_path_fallback}. Skipping row {idx}.")
                    return None

            image = Image.open(full_img_path).convert('RGB')
            score = float(row[self.score_col_name])
        except Exception as e:
            logging.warning(f"Error loading sample idx {idx}, path {img_rel_path} under {self.split_dir}. Error: {e}. Skipping.")
            return None

        if self.transform: image = self.transform(image)
        return image, torch.tensor(score, dtype=torch.float32)


# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count()
    print(f"Using device: {DEVICE}, GPUs: {NUM_GPUS}")
    print(f"Checkpoint save path: {args.full_checkpoint_save_path}")

    # --- Load CLIP Model and Preprocessor ---
    print(f"Loading CLIP model: {args.clip_model_name}...")
    try:
        load_device = "cpu" if (args.finetune_full_model and DEVICE == "cuda" and NUM_GPUS > 1) else DEVICE
        model, preprocess = clip.load(args.clip_model_name, device=load_device)
    except Exception as e: print(f"Error loading CLIP: {e}"); exit(1)
    print(f"CLIP model loaded to {load_device}.")

    # --- Prepare Dataset and DataLoader ---
    print(f"Loading training data from: {args.train_split_dir}")
    train_dataset = ImageScoreDataset(args.train_split_dir, args.score_col_name, preprocess, split_name='train')
    if not train_dataset or len(train_dataset) == 0:
        print(f"Error: No training data loaded from {args.train_split_dir}. Check paths and metadata format."); exit(1)
    print(f"Found {len(train_dataset)} training images.")

    val_dataset = None
    if os.path.exists(args.val_split_dir) and os.path.exists(os.path.join(args.val_split_dir, "val_metadata.csv")):
        print(f"Loading validation data from: {args.val_split_dir}")
        val_dataset = ImageScoreDataset(args.val_split_dir, args.score_col_name, preprocess, split_name='val')
        if val_dataset and len(val_dataset) > 0:
            print(f"Found {len(val_dataset)} validation images.")
        else:
            print(f"Validation data found at {args.val_split_dir} but no valid samples loaded."); val_dataset = None
    else: print("Validation data directory or metadata not found. Proceeding without validation.")

    def safe_collate(batch):
        batch = [item for item in batch if item is not None]
        return torch.utils.data.dataloader.default_collate(batch) if batch else (None, None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True, collate_fn=safe_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True, collate_fn=safe_collate) if val_dataset else None

    # --- Define the Regression Head ---
    feature_dim = model.visual.output_dim
    print(f"CLIP feature dimension: {feature_dim}")
    regression_head = nn.Linear(feature_dim, 1) # Output a single score

    # --- Configure Model for Training Mode ---
    # Same CLIPWithClassifier can be reused, just with regression_head and 1 output unit.
    class CLIPWithRegressor(nn.Module):
        def __init__(self, clip_model, regressor_head, do_finetune):
            super().__init__()
            self.clip_model = clip_model
            self.regressor = regressor_head
            self.do_finetune = do_finetune
        def forward(self, images):
            if self.do_finetune:
                image_features = self.clip_model.encode_image(images).float()
            else:
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images).float()
            return self.regressor(image_features).squeeze(-1) # Squeeze last dim for regression loss

    if args.finetune_full_model:
        print("Configuring for fine-tuning full model.")
        for param in model.parameters(): param.requires_grad = True
        model.to(DEVICE)
    else:
        print("Configuring for linear probe on regression head.")
        for param in model.parameters(): param.requires_grad = False
        model.to(DEVICE) # Base model frozen on device
    
    regression_head.to(DEVICE) # Regression head always trainable and on device

    model_to_train = CLIPWithRegressor(model, regression_head, args.finetune_full_model)

    if args.finetune_full_model:
        optimizer = optim.AdamW(model_to_train.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.AdamW(model_to_train.regressor.parameters(), lr=args.learning_rate)

    if DEVICE == "cuda" and NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs with DataParallel.")
        model_to_train = nn.DataParallel(model_to_train).to(DEVICE)
    else:
        model_to_train.to(DEVICE)

    # --- Loss Function ---
    criterion = nn.MSELoss() if args.loss_type == "mse" else nn.L1Loss()
    print(f"Using {args.loss_type.upper()}Loss.")

    # --- Training Loop ---
    print(f"Starting training for {args.num_epochs} epochs...")
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        model_to_train.train()
        if not args.finetune_full_model:
            actual_model = model_to_train.module if isinstance(model_to_train, nn.DataParallel) else model_to_train
            actual_model.clip_model.eval()
        
        running_loss = 0.0
        processed_batches = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")
        for images, scores in train_pbar:
            if images is None or scores is None: continue
            images, scores = images.to(DEVICE), scores.to(DEVICE).float()
            optimizer.zero_grad()
            outputs = model_to_train(images)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            processed_batches += 1
            train_pbar.set_postfix({'loss': f'{running_loss / processed_batches:.4f}'})
        avg_train_loss = running_loss / processed_batches if processed_batches > 0 else 0

        avg_val_loss_epoch = float('inf')
        if val_loader:
            model_to_train.eval()
            val_loss = 0.0
            val_batches = 0
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")
            with torch.no_grad():
                for images, scores in val_pbar:
                    if images is None or scores is None: continue
                    images, scores = images.to(DEVICE), scores.to(DEVICE).float()
                    outputs = model_to_train(images)
                    loss = criterion(outputs, scores)
                    val_loss += loss.item()
                    val_batches += 1
                    val_pbar.set_postfix({'loss': f'{val_loss / val_batches:.4f}'})
            avg_val_loss_epoch = val_loss / val_batches if val_batches > 0 else float('inf')

            if avg_val_loss_epoch < best_val_loss:
                best_val_loss = avg_val_loss_epoch
                print(f"New best val_loss: {best_val_loss:.4f}. Saving model to {args.full_checkpoint_save_path}")
                try:
                    save_dict = model_to_train.module.state_dict() if isinstance(model_to_train, nn.DataParallel) else model_to_train.state_dict()
                    torch.save(save_dict, args.full_checkpoint_save_path)
                except Exception as e: print(f"Error saving model: {e}")
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss_epoch:.4f}")

    # Save final model if no validation was performed
    if not val_loader:
        print(f"No validation. Saving final model to {args.full_checkpoint_save_path}")
        try:
            save_dict = model_to_train.module.state_dict() if isinstance(model_to_train, nn.DataParallel) else model_to_train.state_dict()
            torch.save(save_dict, args.full_checkpoint_save_path)
        except Exception as e: print(f"Error saving final model: {e}")
    elif val_loader and avg_val_loss_epoch >= best_val_loss:
         print(f"Final epoch val_loss {avg_val_loss_epoch:.4f} was not better than best {best_val_loss:.4f}. Best model already saved.")

    print("Training finished.") 
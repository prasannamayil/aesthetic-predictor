import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import clip # Use openai/clip library
import os
from tqdm.auto import tqdm
from PIL import Image
import argparse
import sys # Added sys for path modification

# Add src directory to Python path to allow importing potential utils or other src modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR) # This is 'src'
PROJECT_ROOT = os.path.dirname(SRC_DIR) # This is project root

# Add project root and src to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# from utils import some_utility # Example if you have utils.py in src/utils/

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a linear readout or fine-tune CLIP for image classification.")
    # Adjusted default data_dir to point to the new structure
    parser.add_argument("--data_root_dir", type=str, default="data", 
                        help="Root directory for processed data (e.g., 'data/', which contains 'data/classification/').")
    parser.add_argument("--task_name", type=str, default="classification", help="Task name, used for subfolder in data_root_dir.")
    # Adjusted default checkpoint_save_path and made it a directory
    parser.add_argument("--checkpoint_dir", type=str, default="results/checkpoints", 
                        help="Directory to save trained model checkpoints.")
    parser.add_argument("--checkpoint_name", type=str, default="clip_classifier.pth", 
                        help="Filename for the saved model checkpoint.")
    parser.add_argument("--clip_model_name", type=str, default="ViT-L/14", 
                        help="CLIP model architecture to use (e.g., ViT-B/32, ViT-L/14).")
    parser.add_argument("--num_classes", type=int, default=3, 
                        help="Number of output classes.")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training and validation.")
    parser.add_argument("--num_epochs", type=int, default=6, 
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Learning rate. Use a smaller value (e.g., 1e-5) if --finetune_full_model is set.")
    parser.add_argument("--validation_split_ratio", type=float, default=0.1, # Renamed from validation_split
                        help="Fraction of training data from 'train' folder to use for validation (0 to disable).")
    parser.add_argument("--finetune_full_model", action='store_true', 
                        help="If set, fine-tune the entire CLIP model instead of just the linear readout.")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    if args.finetune_full_model and args.learning_rate > 1e-4:
        print(f"Warning: Fine-tuning the full model with a relatively high learning rate ({args.learning_rate}). Consider using a smaller LR (e.g., 1e-5 or 1e-6).")

    # Construct full checkpoint path and ensure directory exists
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    base, ext = os.path.splitext(args.checkpoint_name)
    mode_suffix = "_finetuned" if args.finetune_full_model else "_linear"
    args.full_checkpoint_save_path = os.path.join(args.checkpoint_dir, f"{base}{mode_suffix}{ext}")

    # Construct data directories based on new structure
    args.train_data_dir = os.path.join(args.data_root_dir, args.task_name, "train")
    args.test_data_dir = os.path.join(args.data_root_dir, args.task_name, "test") # For potential future use
    # Validation data will be split from args.train_data_dir

    return args

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_GPUS = torch.cuda.device_count()

    print(f"Using device: {DEVICE}, Number of GPUs: {NUM_GPUS}")
    print(f"Training data directory: {args.train_data_dir}")
    print(f"Checkpoint will be saved to: {args.full_checkpoint_save_path}")

    # --- Load CLIP Model and Preprocessor ---
    print(f"Loading CLIP model: {args.clip_model_name}...")
    try:
        load_device = "cpu" if (args.finetune_full_model and DEVICE == "cuda" and NUM_GPUS > 1) else DEVICE
        model, preprocess = clip.load(args.clip_model_name, device=load_device)
    except Exception as e:
        print(f"Error loading CLIP model: {e}")
        exit(1)
    print(f"CLIP model loaded successfully to {load_device}.")

    # --- Prepare Dataset and DataLoader ---
    class SafeImageFolder(datasets.ImageFolder):
        def __getitem__(self, index):
            try:
                return super().__getitem__(index)
            except Exception as e:
                path, _ = self.samples[index]
                print(f"Warning: Skipping image {path} due to error: {e}. Returning None.")
                return None # Returning None, collate_fn will handle it

    def safe_collate(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None, None 
        return torch.utils.data.dataloader.default_collate(batch)

    print(f"Loading data from: {args.train_data_dir}")
    if not os.path.isdir(args.train_data_dir):
        print(f"Error: Training directory not found at {args.train_data_dir}")
        print(f"Please ensure data is prepared using 'src/data_processing/prepare_classification_data.py' first.")
        exit(1)

    try:
        full_train_dataset = SafeImageFolder(args.train_data_dir, transform=preprocess)

        if len(full_train_dataset) == 0:
            print(f"Error: No images found in {args.train_data_dir}.")
            exit(1)

        # Dynamically set num_classes if not matching, or warn.
        if len(full_train_dataset.classes) != args.num_classes:
            print(f"Warning: Argument --num_classes is {args.num_classes}, but found {len(full_train_dataset.classes)} classes in data: {full_train_dataset.classes}. Using detected number of classes: {len(full_train_dataset.classes)}.")
            args.num_classes = len(full_train_dataset.classes)
        
        print(f"Found {len(full_train_dataset)} total images in {args.num_classes} classes.")
        print(f"Class mapping: {full_train_dataset.class_to_idx}")

        val_dataset = None
        if args.validation_split_ratio > 0:
            total_size = len(full_train_dataset)
            val_size = int(args.validation_split_ratio * total_size)
            train_size = total_size - val_size
            if val_size > 0 and train_size > 0:
                print(f"Splitting training data: {train_size} train, {val_size} validation.")
                train_dataset, val_dataset = torch.utils.data.random_split(
                    full_train_dataset, [train_size, val_size],
                    generator=torch.Generator().manual_seed(args.seed)
                )
            else:
                print("Not enough data for validation split, using full dataset for training.")
                train_dataset = full_train_dataset
        else:
            print("No validation split requested, using full dataset for training.")
            train_dataset = full_train_dataset

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, collate_fn=safe_collate
        )

        val_loader = None
        if val_dataset:
            val_loader = DataLoader(
                val_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, collate_fn=safe_collate
            )

    except Exception as e:
        print(f"Error creating dataset or dataloader: {e}")
        exit(1)

    # --- Define the Linear Readout Model ---
    feature_dim = model.visual.output_dim
    print(f"CLIP image feature dimension: {feature_dim}")
    linear_classifier = nn.Linear(feature_dim, args.num_classes)

    # --- Configure Model for Training Mode ---
    if args.finetune_full_model:
        print("Configuring for fine-tuning the full CLIP model + classifier.")
        for param in model.parameters(): param.requires_grad = True
        model.to(DEVICE) 
    else:
        print("Configuring for linear probe (only training the classifier head).")
        for param in model.parameters(): param.requires_grad = False
        model.to(DEVICE) # Base model to device, frozen

    linear_classifier.to(DEVICE) # Classifier to device, always trainable

    # Combine CLIP and classifier into a single module for easier handling with DataParallel
    class CLIPWithClassifier(nn.Module):
        def __init__(self, clip_model, classifier_head, do_finetune):
            super().__init__()
            self.clip_model = clip_model
            self.classifier = classifier_head
            self.do_finetune = do_finetune

        def forward(self, images):
            if self.do_finetune:
                image_features = self.clip_model.encode_image(images).float()
            else:
                with torch.no_grad(): # Ensure no gradients for base model if not fine-tuning
                    image_features = self.clip_model.encode_image(images).float()
            return self.classifier(image_features)

    # Instantiate the combined model
    # The `model` var (CLIP) is already on `load_device` or `DEVICE`
    # `linear_classifier` is on `DEVICE`
    # The combined model will be moved to DEVICE again after DP wrapping if needed
    model_to_train = CLIPWithClassifier(model, linear_classifier, args.finetune_full_model)

    # Optimizer: select parameters based on fine-tuning mode
    if args.finetune_full_model:
        optimizer = optim.AdamW(model_to_train.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.AdamW(model_to_train.classifier.parameters(), lr=args.learning_rate) # Only classifier params

    # --- Multi-GPU Setup (DataParallel) ---
    if DEVICE == "cuda" and NUM_GPUS > 1:
        print(f"Using {NUM_GPUS} GPUs via DataParallel.")
        model_to_train = nn.DataParallel(model_to_train).to(DEVICE)
    else:
        model_to_train.to(DEVICE) # Ensure single device model is on DEVICE

    criterion = nn.CrossEntropyLoss()

    # --- Training Loop ---
    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in range(args.num_epochs):
        model_to_train.train() # Set combined model to train mode
        if not args.finetune_full_model:
             # If linear probing, keep the CLIP part of the custom module in eval mode
             # Accessing .module if DataParallel was used
            actual_model = model_to_train.module if isinstance(model_to_train, nn.DataParallel) else model_to_train
            actual_model.clip_model.eval()

        running_loss = 0.0
        processed_batches = 0
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Train]")

        for inputs, labels in train_progress_bar:
            if inputs is None or labels is None: continue
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model_to_train(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            processed_batches += 1
            train_progress_bar.set_postfix({'loss': f'{running_loss / processed_batches:.4f}'})
        
        avg_train_loss = running_loss / processed_batches if processed_batches > 0 else 0

        # --- Validation Phase ---
        avg_val_loss = -1.0
        if val_loader:
            model_to_train.eval() # Full model to eval for validation
            val_loss = 0.0
            val_batches = 0
            val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.num_epochs} [Val]")

            with torch.no_grad():
                for inputs, labels in val_progress_bar:
                    if inputs is None or labels is None: continue
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model_to_train(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_batches += 1
                    val_progress_bar.set_postfix({'loss': f'{val_loss / val_batches:.4f}'})
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0

        print(f"Epoch [{epoch+1}/{args.num_epochs}], Train Loss: {avg_train_loss:.4f}", end="")
        if val_loader: print(f", Val Loss: {avg_val_loss:.4f}")
        else: print(" (No validation)")

    # --- Save the Trained Model ---
    # Save the state_dict of the underlying model if DataParallel was used
    final_model_state_dict = model_to_train.module.state_dict() if isinstance(model_to_train, nn.DataParallel) else model_to_train.state_dict()
    print(f"Training finished. Saving model checkpoint to: {args.full_checkpoint_save_path}")
    try:
        torch.save(final_model_state_dict, args.full_checkpoint_save_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")

    # To save only the classifier head (useful for linear probe):
    if not args.finetune_full_model:
        classifier_head_path = os.path.join(args.checkpoint_dir, f"{os.path.splitext(args.checkpoint_name)[0]}_classifier_head_linear.pth")
        try:
            classifier_state = model_to_train.module.classifier.state_dict() if isinstance(model_to_train, nn.DataParallel) else model_to_train.classifier.state_dict()
            torch.save(classifier_state, classifier_head_path)
            print(f"Classifier head (linear probe) saved separately to: {classifier_head_path}")
        except Exception as e:
            print(f"Error saving classifier head separately: {e}") 
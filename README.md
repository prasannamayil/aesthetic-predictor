# Aesthetic Predictor with CLIP

This project provides tools to train and evaluate CLIP-based models for two primary tasks:
1.  **Image Classification:** Predict categorical aesthetic labels (e.g., good, neutral, bad) for images.
2.  **Image Regression:** Predict a numerical aesthetic score for images, typically derived from an ELO ranking system.

The models can be trained either by fine-tuning the entire CLIP model or by training only a linear readout layer on top of frozen CLIP features.

## Project Structure

```
.aesthetic-predictor/
├── data/                             # Root for raw and processed datasets
│   ├── classification/               # Processed data for classification
│   │   ├── train/
│   │   │   ├── good/
│   │   │   ├── neutral/
│   │   │   └── bad/
│   │   └── test/
│   │       └── ... (similar structure)
│   └── regression/                   # Processed data for regression
│       ├── train/
│       │   ├── images/
│       │   └── train_metadata.csv
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── val_metadata.csv
│   │   └── test/
│   │       ├── images/
│   │       └── test_metadata.csv
│   ├── cleaned_data/                     # Directory for cleaned input CSVs (e.g., ratings, comparisons)
│   ├── notebooks/                        # Jupyter notebooks for exploration and visualization
│   │   ├── asthetics_predictor.ipynb
│   │   ├── evaluate_and_visualize.ipynb
│   │   └── scratch.ipynb
│   ├── results/
│   │   ├── checkpoints/                  # Saved model checkpoints
│   │   ├── evaluation_outputs/           # CSV files with evaluation predictions
│   │   └── plots/                        # Plots generated during evaluation or analysis
│   ├── src/
│   │   ├── data_processing/              # Scripts for data downloading, cleaning, and preparation
│   │   │   ├── download_utils.py         # Core utilities for downloading and processing
│   │   │   ├── prepare_classification_data.py # Script to prepare data for classification
│   │   │   ├── prepare_regression_data.py  # Script to prepare data for regression
│   │   │   ├── combine_comparisons.py    # Utility to merge comparison CSVs for ELO
│   │   │   └── elo_processing.py         # Script to calculate ELO scores from comparisons
│   │   ├── training/                     # Scripts for model training
│   │   │   ├── train_classifier.py
│   │   │   └── train_regressor.py
│   │   ├── evaluation/                   # Scripts for model evaluation
│   │   │   └── evaluate_regressor.py     # (Classifier evaluation might be in notebooks or need a new script)
│   │   └── utils.py                      # General utility functions (e.g., loading aesthetic models)
│   ├── LICENSE
│   └── README.md                         # This file
└── requirements.txt                  # Python dependencies
```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd aesthetic-predictor
    ```

2.  **Create a Python environment and activate it:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    Ensure you have PyTorch installed (preferably with CUDA support). Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/).
    Then, install the other requirements:
    ```bash
    pip install -r requirements.txt
    ```
    The `clip-by-openai` package might occasionally need to be installed directly from GitHub if the PyPI version has issues:
    ```bash
    pip install git+https://github.com/openai/CLIP.git
    ```

## Workflow

The general workflow involves:
1.  **Preparing Input CSVs:** Ensure your initial data (ratings for classification, pairwise comparisons for regression) is in CSV format and placed in a suitable location (e.g., `cleaned_data/`).
2.  **Data Processing:** Run scripts from `src/data_processing/` to download images, calculate ELO scores (for regression), and structure the data into the `data/classification/` or `data/regression/` directories.
3.  **Training:** Use scripts from `src/training/` to train your models. Checkpoints will be saved in `results/checkpoints/`.
4.  **Evaluation:** Use scripts from `src/evaluation/` (or notebooks) to evaluate trained models. Outputs are saved in `results/evaluation_outputs/` and `results/plots/`.

### 1. Data Processing

#### a) For Classification Task (Good/Neutral/Bad Ratings)

   - **Input:** A CSV file (e.g., `cleaned_data/my_ratings.csv`) with columns for image URLs and their corresponding class labels (e.g., 'good', 'neutral', 'bad').
   - **Script:** `src/data_processing/prepare_classification_data.py`
   - **Action:** Downloads images, splits them into train/test (and optionally validation) sets, and organizes them into class-specific subdirectories under `data/classification/`.
   - **Example Command:**
     ```bash
     python src/data_processing/prepare_classification_data.py \
         --csv_path cleaned_data/my_ratings.csv \
         --url_col 'image_url_column_name' \
         --rating_col 'rating_column_name' \
         --classes good neutral bad \
         --output_root_dir data \
         --test_split_size 0.15 \
         --val_split_size 0.0 # No validation images from this script, train script will split train further
     ```

#### b) For Regression Task (ELO Scores)

   This is a two-step process:

   **Step 1: Generate ELO Scores (if you have pairwise comparison data)**
   - **Input:** A CSV file (e.g., `cleaned_data/pairwise_comparisons.csv`) containing pairwise image comparisons. This file should have columns for the URLs of the two images being compared and a column indicating the winner.
       - If you have multiple comparison files, you can merge them first using `src/data_processing/combine_comparisons.py`.
         ```bash
         python src/data_processing/combine_comparisons.py # (Modify paths inside the script)
         ```
   - **Script:** `src/data_processing/elo_processing.py`
   - **Action:** Calculates ELO ratings and scaled scores (1-10) for each unique image URL based on the comparisons.
   - **Output:** A CSV file (e.g., `url_elo_rankings.csv` at the project root by default) with columns like `url`, `elo_rating`, `rank`, `scaled_score_1_10`.
   - **Example Command:**
     ```bash
     python src/data_processing/elo_processing.py \
         --input_csv cleaned_data/combined_comparison_data.csv \
         --output_csv url_elo_rankings.csv \
         --url_a_col 'image1_url' \
         --url_b_col 'image2_url' \
         --winner_col 'result' # Assumes 1.0 if image A wins, else B wins
     ```

   **Step 2: Prepare Regression Dataset (Images and Scores)**
   - **Input:** The CSV file generated by `elo_processing.py` (e.g., `url_elo_rankings.csv`).
   - **Script:** `src/data_processing/prepare_regression_data.py`
   - **Action:** Downloads images, splits data into train/val/test sets, saves images into `data/regression/<split>/images/`, and creates corresponding metadata CSVs (e.g., `data/regression/train/train_metadata.csv`) linking image paths to scores.
   - **Example Command:**
     ```bash
     python src/data_processing/prepare_regression_data.py \
         --csv_path url_elo_rankings.csv \
         --url_col 'url' \
         --score_col 'scaled_score_1_10' \
         --output_root_dir data \
         --test_split_size 0.1 \
         --val_split_size 0.1
     ```

### 2. Training Models

All training scripts save model checkpoints to `results/checkpoints/`. The filename will indicate whether it was a linear probe or full fine-tuning.

#### a) Training a Classifier
   - **Script:** `src/training/train_classifier.py`
   - **Input Data:** Processed classification data from `data/classification/`.
   - **Example (Linear Probe):**
     ```bash
     python src/training/train_classifier.py \
         --data_root_dir data \
         --task_name classification \
         --checkpoint_name my_classifier.pth \
         --clip_model_name "ViT-L/14" \
         --batch_size 64 \
         --num_epochs 5 \
         --learning_rate 1e-3 \
         --validation_split_ratio 0.1
     ```
   - **Example (Full Fine-tuning):**
     ```bash
     python src/training/train_classifier.py \
         --data_root_dir data \
         --task_name classification \
         --checkpoint_name my_classifier_finetuned.pth \
         --clip_model_name "ViT-L/14" \
         --batch_size 16 \
         --num_epochs 3 \
         --learning_rate 1e-5 \
         --finetune_full_model \
         --validation_split_ratio 0.1
     ```

#### b) Training a Regressor (Predicting Aesthetic Scores)
   - **Script:** `src/training/train_regressor.py`
   - **Input Data:** Processed regression data from `data/regression/` (expects `train_metadata.csv`, `val_metadata.csv` and corresponding image folders).
   - **Example (Linear Probe):**
     ```bash
     python src/training/train_regressor.py \
         --data_root_dir data \
         --task_name regression \
         --checkpoint_name my_regressor.pth \
         --clip_model_name "ViT-L/14" \
         --score_col_name "scaled_score_1_10" \
         --batch_size 32 \
         --num_epochs 10 \
         --learning_rate 1e-3 \
         --loss_type "mse"
     ```
   - **Example (Full Fine-tuning):**
     ```bash
     python src/training/train_regressor.py \
         --data_root_dir data \
         --task_name regression \
         --checkpoint_name my_regressor_finetuned.pth \
         --clip_model_name "ViT-L/14" \
         --score_col_name "scaled_score_1_10" \
         --batch_size 16 \
         --num_epochs 5 \
         --learning_rate 1e-5 \
         --finetune_full_model \
         --loss_type "mse"
     ```

### 3. Evaluating Models

#### a) Evaluating a Regressor
   - **Script:** `src/evaluation/evaluate_regressor.py`
   - **Action:** Loads a trained regression model, predicts scores on a test set, calculates metrics (MSE, MAE, Correlation), saves all predictions to a CSV in `results/evaluation_outputs/`, and generates a visualization plot in `results/plots/`.
   - **Example Command:**
     ```bash
     python src/evaluation/evaluate_regressor.py \
         --checkpoint_path results/checkpoints/my_regressor_linear.pth \
         --data_root_dir data \
         --task_name regression \
         --split_name test \
         --clip_model_name "ViT-L/14" \
         --score_col_name "scaled_score_1_10" \
         # Add --was_full_model_finetuned if evaluating a fine-tuned checkpoint
         --num_samples_visualize 25 \
         --output_plot_name regressor_eval.png \
         --output_csv_name regressor_predictions.csv
     ```

#### b) Evaluating a Classifier
   (Currently, specific evaluation scripts for the classifier might be found within the notebooks, or a dedicated script can be added to `src/evaluation/`. The general approach would involve loading a trained classifier, predicting on the `data/classification/test` set, and calculating metrics like accuracy, precision, recall, F1-score, and a confusion matrix.)


## Utilities

- **`src/utils.py`:** Contains helper functions, including:
    - `get_aesthetic_model()`: Downloads and loads a pre-trained aesthetic model (linear layer on CLIP features) from the LAION-AI/aesthetic-predictor repository.
    - `get_aesthetic_score()`: Calculates an aesthetic score for a PIL image using a provided CLIP model and the loaded aesthetic linear model.

## Notebooks

The `notebooks/` directory contains Jupyter notebooks for various purposes, such as:
-   Initial data exploration and cleaning.
-   Model evaluation and visualization (e.g., `evaluate_and_visualize.ipynb`).
-   Prototyping and scratch work.

Remember to adjust paths and parameters in the example commands to match your specific filenames, column names, and desired configurations.

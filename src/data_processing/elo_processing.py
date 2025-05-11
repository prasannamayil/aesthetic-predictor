import pandas as pd
import numpy as np
import math
import argparse # Added argparse
import os # Added os to construct paths relative to script if needed

# --- Configuration ---
# Default paths, can be overridden by command-line arguments
DEFAULT_INPUT_CSV = 'cleaned_data/combined_comparison_data.csv'
DEFAULT_OUTPUT_CSV = 'url_elo_rankings.csv'
# Column names are kept as globals for now, but could also be args
URL_A_COL = 'image1_url'
URL_B_COL = 'image2_url'
WINNER_COL = 'result'

# Elo Configuration
INITIAL_RATING = 1500
K_FACTOR = 32

# --- Elo Functions ---
def probability(rating1, rating2):
    """Calculates the expected probability of player 1 winning against player 2."""
    return 1.0 / (1.0 + math.pow(10, (rating2 - rating1) / 400))

def update_elo(winner_rating, loser_rating, k_factor):
    """Calculates the new Elo ratings for winner and loser."""
    prob_winner = probability(loser_rating, winner_rating)
    prob_loser = probability(winner_rating, loser_rating)

    new_winner_rating = winner_rating + k_factor * (1 - prob_winner)
    new_loser_rating = loser_rating + k_factor * (0 - prob_loser)

    return new_winner_rating, new_loser_rating

def calculate_elo_ratings(df, url_a_col, url_b_col, winner_col, initial_rating, k_factor):
    """Processes comparisons and calculates Elo ratings."""
    elo_ratings = {}
    all_urls = pd.concat([df[url_a_col], df[url_b_col]]).unique()
    for url in all_urls:
        if pd.notna(url):
            elo_ratings[str(url)] = initial_rating

    print(f"Processing {len(df)} comparisons...")
    for index, row in df.iterrows():
        url_a_val = row[url_a_col]
        url_b_val = row[url_b_col]
        winner_val = row[winner_col]

        if pd.isna(url_a_val) or pd.isna(url_b_val) or pd.isna(winner_val):
            print(f"Warning: Skipping row {index} due to missing URL(s) or winner.")
            continue
        
        url_a = str(url_a_val)
        url_b = str(url_b_val)

        # Determine the actual winner URL based on winner_col value
        # Assuming winner_col == 1.0 means url_a wins, otherwise url_b wins
        # This logic needs to be robust based on how winner_col is defined.
        # The original script had: winner = str(row[url_a_col]) if row[winner_col] == 1.0 else str(row[url_b_col])
        # This assumes winner_col directly indicates the winner choice (e.g. 1 for A, 2 for B, or 0 for B, 1 for A)
        # Let's refine this: if winner_col contains the URL of the winner, that's simpler.
        # If winner_col is 'image1_url' or 'image2_url' string, use that.
        # If winner_col is 1 or 0 (or 1.0 / 0.0), interpret accordingly.
        
        # Current logic from file: winner = str(row[url_a_col]) if row[winner_col] == 1.0 else str(row[url_b_col])
        # This means if the value in winner_col is 1.0, url_a is the winner.
        # If it's anything else (e.g. 0.0 or 2.0 if that's how B is marked), url_b is the winner.
        # This interpretation should be clarified or made more robust if data format varies.
        actual_winner_url = str(url_a) if winner_val == 1.0 else str(url_b)
        
        if url_a not in elo_ratings or url_b not in elo_ratings:
            print(f"Warning: Skipping row {index}. One or both URLs ('{url_a}', '{url_b}') not found in initial list.")
            continue

        if actual_winner_url == url_a:
            winner_id = url_a
            loser_id = url_b
        elif actual_winner_url == url_b:
            winner_id = url_b
            loser_id = url_a
        else:
            print(f"Warning: Skipping row {index}. Winner ({actual_winner_url}) could not be resolved from '{url_a}' or '{url_b}'. Check winner_col interpretation.")
            continue

        winner_current_rating = elo_ratings[winner_id]
        loser_current_rating = elo_ratings[loser_id]

        new_winner_rating, new_loser_rating = update_elo(
            winner_current_rating, loser_current_rating, k_factor
        )
        elo_ratings[winner_id] = new_winner_rating
        elo_ratings[loser_id] = new_loser_rating
    
    results_df = pd.DataFrame(list(elo_ratings.items()), columns=['url', 'elo_rating'])
    results_df = results_df.sort_values(by='elo_rating', ascending=False).reset_index(drop=True)
    results_df['rank'] = results_df.index + 1
    return results_df

def scale_ratings(df, elo_col='elo_rating', scaled_col='scaled_score_1_10'):
    """Scales Elo ratings to a 1-10 range."""
    min_rating = df[elo_col].min()
    max_rating = df[elo_col].max()
    if max_rating == min_rating:
        df[scaled_col] = 5.5
    else:
        df[scaled_col] = 1 + 9 * (df[elo_col] - min_rating) / (max_rating - min_rating)
    df[scaled_col] = df[scaled_col].round(2)
    return df

def main():
    parser = argparse.ArgumentParser(description="Calculate Elo ratings from pairwise comparison data.")
    parser.add_argument("--input_csv", type=str, default=DEFAULT_INPUT_CSV,
                        help=f"Path to the input CSV file with comparison data. Default: {DEFAULT_INPUT_CSV}")
    parser.add_argument("--output_csv", type=str, default=DEFAULT_OUTPUT_CSV,
                        help=f"Path to save the output CSV file with Elo rankings. Default: {DEFAULT_OUTPUT_CSV}")
    parser.add_argument("--url_a_col", type=str, default=URL_A_COL,
                        help=f"Column name for the first URL in comparison. Default: {URL_A_COL}")
    parser.add_argument("--url_b_col", type=str, default=URL_B_COL,
                        help=f"Column name for the second URL in comparison. Default: {URL_B_COL}")
    parser.add_argument("--winner_col", type=str, default=WINNER_COL,
                        help=f"Column name indicating the winner (e.g., result is 1.0 if URL A wins, else URL B wins). Default: {WINNER_COL}")
    parser.add_argument("--initial_rating", type=int, default=INITIAL_RATING,
                        help=f"Initial Elo rating for all items. Default: {INITIAL_RATING}")
    parser.add_argument("--k_factor", type=int, default=K_FACTOR,
                        help=f"K-factor for Elo calculation. Default: {K_FACTOR}")
    
    args = parser.parse_args()

    # --- Load Data ---
    print(f"Loading data from: {args.input_csv}")
    try:
        df = pd.read_csv(args.input_csv)
        print(f"Successfully loaded data with {len(df)} rows from {args.input_csv}")
    except FileNotFoundError:
        print(f"Error: File not found at {args.input_csv}")
        return # Changed exit() to return for better testability if wrapped
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # --- Validate Columns ---
    required_cols = [args.url_a_col, args.url_b_col, args.winner_col]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Missing one or more required columns in the CSV: {required_cols}. Found: {list(df.columns)}")
        return
    
    # --- Calculate Elo Ratings ---
    print("Initializing and calculating Elo ratings...")
    results_df = calculate_elo_ratings(df, args.url_a_col, args.url_b_col, args.winner_col, 
                                     args.initial_rating, args.k_factor)

    # --- Scale Elo Ratings (1-10) ---
    if not results_df.empty:
        print("Scaling Elo ratings...")
        results_df = scale_ratings(results_df)
        # --- Display Results ---
        print("\n--- Elo Ranking Results ---")
        print(results_df.to_string())
    else:
        print("No results to scale or display. Elo calculation might have yielded an empty DataFrame.")

    # --- Save Results ---
    if not results_df.empty:
        try:
            results_df.to_csv(args.output_csv, index=False)
            print(f"\nResults saved to {args.output_csv}")
        except Exception as e:
            print(f"\nError saving results to CSV: {e}")
    else:
        print(f"No results to save. Output file {args.output_csv} will not be created/updated.")

if __name__ == "__main__":
    main()
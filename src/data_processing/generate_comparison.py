import csv
import itertools
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_INPUT_RATING_FILE = 'cleaned_data/AP_Data_Rating_1400_050525_cleaned.csv'
DEFAULT_OUTPUT_COMPARISON_FILE = 'cleaned_data/generated_comparison_data.csv'

def generate_comparison_data(rating_file, output_file, 
                             good_label='good', bad_label='bad', neutral_label='neutral',
                             url_col='url', rating_col='rating'):
    """
    Reads a CSV file with image ratings and generates a comparison CSV file.
    Hierarchy: good > bad, good > neutral, bad > neutral.

    Args:
        rating_file (str): Path to the input rating CSV file.
        output_file (str): Path to the output comparison CSV file.
        good_label (str): Label for 'good' ratings.
        bad_label (str): Label for 'bad' ratings.
        neutral_label (str): Label for 'neutral' ratings.
        url_col (str): Name of the URL column in the input CSV.
        rating_col (str): Name of the rating column in the input CSV.
    """
    ratings = {good_label: [], bad_label: [], neutral_label: []}
    valid_ratings = [good_label, bad_label, neutral_label]

    try:
        with open(rating_file, 'r', newline='') as infile:
            reader = csv.DictReader(infile)
            if url_col not in reader.fieldnames or rating_col not in reader.fieldnames:
                logging.error(f"Input file {rating_file} must contain '{url_col}' and '{rating_col}' columns.")
                return False
            for row in reader:
                url = row[url_col]
                rating = row[rating_col].lower()
                if rating in ratings:
                    ratings[rating].append(url)
                else:
                    logging.warning(f"Unknown rating '{row[rating_col]}' for url {url}. Skipping.")
    except FileNotFoundError:
        logging.error(f"Input file {rating_file} not found.")
        return False
    except Exception as e:
        logging.error(f"Error reading {rating_file}: {e}")
        return False

    logging.info(f"Read ratings: {good_label}={len(ratings[good_label])}, "
                 f"{bad_label}={len(ratings[bad_label])}, "
                 f"{neutral_label}={len(ratings[neutral_label])}")

    if not any(ratings.values()):
        logging.error("No valid ratings found in the input file. Cannot generate comparisons.")
        return False

    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['image1_url', 'image2_url', 'result']) # result=1.0 means image1 wins
            count = 0

            # Good > Bad
            for good_url, bad_url in itertools.product(ratings[good_label], ratings[bad_label]):
                writer.writerow([good_url, bad_url, 1.0])
                count += 1

            # Good > Neutral
            for good_url, neutral_url in itertools.product(ratings[good_label], ratings[neutral_label]):
                writer.writerow([good_url, neutral_url, 1.0])
                count += 1

            # Bad > Neutral
            for bad_url, neutral_url in itertools.product(ratings[bad_label], ratings[neutral_label]):
                writer.writerow([bad_url, neutral_url, 1.0])
                count += 1

        logging.info(f"Successfully generated {count} comparisons in {output_file}")
        return True
    except Exception as e:
        logging.error(f"Error writing to {output_file}: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate pairwise comparison data from a rating CSV.")
    parser.add_argument("--input_rating_file", type=str, default=DEFAULT_INPUT_RATING_FILE,
                        help=f"Path to the input rating CSV. Default: {DEFAULT_INPUT_RATING_FILE}")
    parser.add_argument("--output_comparison_file", type=str, default=DEFAULT_OUTPUT_COMPARISON_FILE,
                        help=f"Path to the output comparison CSV. Default: {DEFAULT_OUTPUT_COMPARISON_FILE}")
    parser.add_argument("--url_col", type=str, default='url', help="Name of the URL column.")
    parser.add_argument("--rating_col", type=str, default='rating', help="Name of the rating column.")
    parser.add_argument("--good_label", type=str, default='good', help="Label for good ratings.")
    parser.add_argument("--bad_label", type=str, default='bad', help="Label for bad ratings.")
    parser.add_argument("--neutral_label", type=str, default='neutral', help="Label for neutral ratings.")

    args = parser.parse_args()

    generate_comparison_data(
        args.input_rating_file, 
        args.output_comparison_file,
        good_label=args.good_label,
        bad_label=args.bad_label,
        neutral_label=args.neutral_label,
        url_col=args.url_col,
        rating_col=args.rating_col
    ) 
import logging
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)
import argparse

from data_pipeline import pipeline
from constants import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        default=False,
        action="store_true",
        help="Use full TinyStores dataset instead of the small one.",
    )
    args = parser.parse_args()

    full_training_data = args.full
    
    # TODO: also validate on full dataset
    if full_training_data:
        logging.info("Using full TinyStores dataset.")
        path = "data/TinyStoriesV2-GPT4-train.txt"
    else:
        logging.info("Using small dev dataset.")
        path = "data/data_train.txt"

    data = pipeline(path, "data/data_validation.txt")

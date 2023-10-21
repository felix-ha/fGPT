import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(message)s")
file_handler = logging.FileHandler("log.txt")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)




import argparse

from data_pipeline import pipeline
from constants import *
import wget
import os


if __name__ == "__main__":

    logging.info("This is a test log entry.")
    logging.info("This is a test log entry.")
    logging.info("This is a test log entry.")
    logging.info("This is a test log entry.")
    logging.info("This is a test log entry.")
    quit()


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        default=False,
        action="store_true",
        help="Use full TinyStores dataset instead of the small one.",
    )
    args = parser.parse_args()

    full_training_data = args.full
   
    if full_training_data:
        logging.info("Using full TinyStores dataset.")
        path_train = "data/TinyStoriesV2-GPT4-train.txt"
        path_validation = "data/TinyStoriesV2-GPT4-valid.txt"
        if not os.path.exists(path_train):
            logging.info("Downloading full TinyStores training dataset.")
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
            wget.download(url, path_train)
        if not os.path.exists(path_validation):
            logging.info("Downloading full TinyStores validation dataset.")
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
            wget.download(url, path_validation)
    else:
        logging.info("Using small dev dataset.")
        path_train = "data/data_train.txt"
        path_validation = "data/data_validation.txt"

    data = pipeline(path_train, path_validation)

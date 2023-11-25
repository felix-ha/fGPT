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
from torch.utils.data import DataLoader
from data_prep import collate_fn
from tokenizer import create_encoder, create_decoder, split_tokens_raw
from data_prep import read_from_json, get_token_int_dicts

from dionysus.training import TrainingConfig, train
from model import (
    LanguageModelDataset,
    simpleGPT,
    cross_entropy_language_model,
    generate,
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        default=False,
        action="store_true",
        help="Use full TinyStores dataset instead of the small one.",
    )
    parser.add_argument(
        "--ratio",
        default=1.0,
        type=float,
        help="Ratio of the data to use for training.",
    )
    parser.add_argument(
        "--splits",
        default=2,
        type=int,
        help="Ratio of the data to use for training.",
    )

    args = parser.parse_args()

    if args.full:
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

    path = "datapipeline"
    pipeline(path_train, path_validation, path, args.ratio, args.splits)

    device = "cpu"

    token_to_int, int_to_token = get_token_int_dicts(path)
    texts_ids_train = read_from_json(os.path.join(path, "texts_ids_train.json"))
    texts_ids_validation = read_from_json(
        os.path.join(path, "texts_ids_validation.json")
    )

    vocab_size = len(int_to_token)
    n_positions = max([len(text_ids) for text_ids in texts_ids_train])

    encoder = create_encoder(token_to_int, END_OF_TEXT, TOKEN_TO_REMOVE, UNK)
    decoder = create_decoder(int_to_token)

    dataset_train = LanguageModelDataset(texts_ids_train)
    dataset_validation = LanguageModelDataset(texts_ids_validation)

    dataloader_train = DataLoader(
        dataset_train, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    dataloader_validation = DataLoader(
        dataset_validation, batch_size=4, shuffle=False, collate_fn=collate_fn
    )

    stop_token_id = token_to_int[END_OF_TEXT]

    model = simpleGPT(
        vocab_size=vocab_size,
        n_embd=8,
        num_heads=4,
        block_size=n_positions,
        n_layer=2,
        dropout=0.1,
        device=device,
    )

    loss_func = cross_entropy_language_model

    train_config = TrainingConfig(
        model=model,
        epochs=5,
        loss_func=loss_func,
        training_loader=dataloader_train,
        validation_loader=dataloader_validation,
        optimizer="AdamW",
        device=device,
        colab=False,
        save_model=True,
        tar_result=True,
        save_path="runs",
        model_name="GPT-2",
        progress_bar=True,
    )

    train(train_config)

    prompt = "Alice was so tired when she got back home so she went"
    output, choices = generate(
        model,
        prompt,
        encoder,
        decoder,
        stop_token_id=stop_token_id,
        max_n=5,
        choices_per_step=3,
    )

    logging.info(f"\n{choices}")
    logging.info(f"Promt: {prompt}")
    logging.info(f"Model: {output}")

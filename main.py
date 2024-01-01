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
from fgpt.constants import *
import wget
import os
import torch
from torch.utils.data import DataLoader
from data_prep import collate_fn
from tokenizer import create_encoder, create_decoder, split_tokens_raw
from data_prep import read_from_json, get_token_int_dicts

from dionysus.training import TrainingConfig, train
from fgpt.model import (
    LanguageModelDataset,
    simpleGPT,
    cross_entropy_language_model,
    generate,
)


def get_model(vocab_size, n_positions, device):
    return simpleGPT(
        vocab_size=vocab_size,
        n_embd=768,
        num_heads=4,
        block_size=n_positions,
        n_layer=4,
        dropout=0.1,
        device=device,
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
        default=100,
        type=int,
        help="Ratio of the data to use for training.",
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Number of epochs for training.",
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
    pipeline(
        path_train,
        path_validation,
        path,
        args.ratio,
        args.splits,
        token_to_int_file="datapipeline/token_to_int.json",
        int_to_token_file="datapipeline/int_to_token.json",
    )

    device = "gpu" if torch.cuda.is_available() else "cpu"

    token_to_int, int_to_token = get_token_int_dicts(path)
    texts_ids_train = read_from_json(os.path.join(path, "texts_ids_train.json"))
    texts_ids_validation = read_from_json(
        os.path.join(path, "texts_ids_validation.json")
    )

    dataset_info = read_from_json(os.path.join(path, "dataset_info.json"))
    vocab_size = dataset_info["vocab_size"]
    n_positions = dataset_info["n_positions"]

    encoder = create_encoder(token_to_int, END_OF_TEXT, TOKEN_TO_REMOVE, UNK)
    decoder = create_decoder(int_to_token)

    dataset_train = LanguageModelDataset(texts_ids_train)
    dataset_validation = LanguageModelDataset(texts_ids_validation)

    dataloader_train = DataLoader(
        dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn
    )
    dataloader_validation = DataLoader(
        dataset_validation, batch_size=8, shuffle=False, collate_fn=collate_fn
    )

    stop_token_id = token_to_int[END_OF_TEXT]

    model = get_model(vocab_size, n_positions, device)

    loss_func = cross_entropy_language_model

    train_config = TrainingConfig(
        model=model,
        epochs=args.epochs,
        loss_func=loss_func,
        training_loader=dataloader_train,
        validation_loader=dataloader_validation,
        optimizer="AdamW",
        device=device,
        force_write_logs=False,
        save_model=True,
        tar_result=True,
        save_path="runs",
        model_name="fGPT",
        progress_bar=True,
        checkpoint_step=1,
        checkpoint_step_batch=500
    )

    logging.info(f"maybe set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32")

    train(train_config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of parameters: {num_params}")

import time
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import DataLoader

from tokenizer import (
    split_tokens_raw,
    clean_tokens,
    get_unique_tokens,
    create_token_to_int_dicts,
    create_encoder,
    create_decoder,
)
from data_prep import (
    load_file,
    split_corpus,
    texts_to_input_ids,
    collate_fn,
    LanguageModelDataset,
)
from constants import *

import logging

logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)


def print_unique_characters(corpus: str):
    unique_chars = set(corpus.replace(END_OF_TEXT, ""))
    sorted_chars = sorted(unique_chars)
    logging.info(f"Unique characters: {sorted_chars}")


def replace_characters(corpus, replacment_dict):
    logging.info("start replace_characters")
    for char, replacement in replacment_dict.items():
        logging.info(f"replace {char} with {replacement}")
        corpus = corpus.replace(char, replacement)
    logging.info("end replace_characters")
    return corpus


@dataclass
class Data:
    """
    Result from data pipeline that builds a data set for a language model.

    Args:
        n_positions:  The maximum sequence length that this model can be used. Determined by the maximum length of the texts in the training set
        vocab_size: Size of the vocabulary.
        token_to_int: Mapping from tokens to integers.
        encoder: Callable that converts a string to a list of tokens (int).
        int_to_token: Mapping from integers to tokens.
        decoder: Callable that converts a list of integers to a string.
        dataloader_train: Dataloader for the training set.
        dataloader_validation: Dataloader for the validation set.
    """

    n_positions: int
    vocab_size: int
    token_to_int: dict
    encoder: callable
    int_to_token: dict
    decoder: callable
    dataloader_train: torch.utils.data.DataLoader
    dataloader_validation: torch.utils.data.DataLoader


def pipeline(file_path_train, file_path_validation, ratio=1.0, number_splits_for_sub_corpus=10):
    start_time = time.perf_counter()

    # Create vocabular, i. e. token <-> int mappings^
    logging.info("start reading file")
    corpus_train_raw = load_file(file_path_train, ratio)
    logging.info("read file")

    print_unique_characters(corpus_train_raw)
    corpus_train_clean = replace_characters(corpus_train_raw, CHARACTER_REPLACEMENTS)
    tokens_raw = split_tokens_raw(corpus_train_clean, DELIMTERS, number_splits_for_sub_corpus)
    tokens_all = clean_tokens(tokens_raw, TOKEN_TO_REMOVE)
    tokens_unique = get_unique_tokens(tokens_all, vocab_size=10_000)
    token_to_int, int_to_token = create_token_to_int_dicts(tokens_unique, UNK)
    vocab_size = len(int_to_token)
    logging.info(f"Size of vocabulary: {vocab_size}")
  

    encoder = create_encoder(token_to_int, DELIMTERS, TOKEN_TO_REMOVE, UNK)
    decoder = create_decoder(int_to_token)

    # Split whole corpus after character replacments in CHARACTER_REPLACEMENTS accoring to END_OF_TEXT token
    texts_train = split_corpus(corpus_train_clean, END_OF_TEXT)
    logging.info(f"Number of stories: {len(texts_train)}")

    with open(file_path_validation, "r", encoding="utf8") as file:
        corpus_validation_raw = file.read()
    texts_validation = split_corpus(corpus_validation_raw, END_OF_TEXT)


    quit()
    # Convert texts to input IDs
    texts_ids_train = texts_to_input_ids(texts_train, encoder)
    texts_ids_validation = texts_to_input_ids(texts_validation, encoder)

    n_positions = max([len(text_ids) for text_ids in texts_ids_train])
    logging.info(f"Maxmial size of a text: {n_positions}")
    logging.info(
        f"Average length of stories: {np.mean([len(text_ids) for text_ids in texts_ids_train]):.1f}"
    )

    dataset_train = LanguageModelDataset(texts_ids_train)
    dataset_validation = LanguageModelDataset(texts_ids_validation)

    dataloader_train = DataLoader(
        dataset_train, batch_size=2, shuffle=True, collate_fn=collate_fn
    )
    dataloader_validation = DataLoader(
        dataset_validation, batch_size=2, shuffle=False, collate_fn=collate_fn
    )

    end_time = time.perf_counter()
    logging.info(f"Time taken: {end_time - start_time:.6f} seconds")

    return Data(
        n_positions=n_positions,
        vocab_size=vocab_size,
        token_to_int=token_to_int,
        encoder=encoder,
        int_to_token=int_to_token,
        decoder=decoder,
        dataloader_train=dataloader_train,
        dataloader_validation=dataloader_validation,
    )

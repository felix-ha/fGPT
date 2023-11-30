import time
import numpy as np
from pathlib import Path
import os
import tarfile


from tokenizer import (
    split_tokens_raw,
    clean_tokens,
    get_unique_tokens,
    create_token_to_int_dicts,
    create_encoder,
)
from data_prep import write_to_json, load_file, split_corpus, texts_to_input_ids
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


def tar_folder(path):
    """
    Creates a .tar file for a given folder
    """
    with tarfile.open(f"{path}.tar", "w") as tar:
        tar.add(path, arcname=os.path.basename(path))


def pipeline(
    file_path_train,
    file_path_validation,
    path,
    ratio=1.0,
    number_splits_for_sub_corpus=10,
):
    start_time = time.perf_counter()

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Create vocabular, i. e. token <-> int mappings^
    logging.info("start reading file")
    corpus_train_raw = load_file(file_path_train, ratio)
    logging.info("read file")

    print_unique_characters(corpus_train_raw)
    corpus_train_clean = replace_characters(corpus_train_raw, CHARACTER_REPLACEMENTS)
    tokens_raw = split_tokens_raw(
        corpus_train_clean, END_OF_TEXT, number_splits_for_sub_corpus
    )
    tokens_all = clean_tokens(tokens_raw, TOKEN_TO_REMOVE)
    tokens_unique = get_unique_tokens(tokens_all, vocab_size=10_000)
    token_to_int, int_to_token = create_token_to_int_dicts(
        tokens_unique, UNK, END_OF_TEXT
    )
    vocab_size = len(int_to_token)
    logging.info(f"Size of vocabulary: {vocab_size}")

    encoder = create_encoder(token_to_int, END_OF_TEXT, TOKEN_TO_REMOVE, UNK)

    # Split whole corpus after character replacments in CHARACTER_REPLACEMENTS accoring to END_OF_TEXT token
    texts_train = split_corpus(corpus_train_clean, END_OF_TEXT)
    logging.info(f"Number of stories: {len(texts_train)}")

    with open(file_path_validation, "r", encoding="utf8") as file:
        corpus_validation_raw = file.read()
        if ratio < 1.0:
            corpus_validation_raw = corpus_validation_raw[:int(len(corpus_validation_raw)*ratio)]

    texts_validation = split_corpus(corpus_validation_raw, END_OF_TEXT)

    # Convert texts to input IDs
    logging.info("converting training texts to input ids")
    texts_ids_train = texts_to_input_ids(texts_train, encoder)
    texts_ids_train = [item + [token_to_int[END_OF_TEXT]] for item in texts_ids_train]

    logging.info("converting validation texts to input ids")
    texts_ids_validation = texts_to_input_ids(texts_validation, encoder)
    texts_ids_validation = [item + [token_to_int[END_OF_TEXT]] for item in texts_ids_validation]
    
    n_positions = max([len(text_ids) for text_ids in texts_ids_train])
    logging.info(f"Maxmial size of a text: {n_positions}")
    logging.info(
        f"Average length of stories: {np.mean([len(text_ids) for text_ids in texts_ids_train]):.1f}"
    )

    vocab_size = len(int_to_token)
    n_positions = max([len(text_ids) for text_ids in texts_ids_train])
    number_of_tokens = len([item for sublist in texts_ids_train for item in sublist])

    logging.info(f"Size of vocabulary: {vocab_size}")
    logging.info(f"Maxmial size of a text: {n_positions}")
    logging.info(f"Number of tokens in training set: {number_of_tokens}")

    dataset_info = {
        "vocab_size": vocab_size,
        "n_positions": n_positions,
        "n_tokens_training:": number_of_tokens
    }

    logging.info("writing results")

    write_to_json(token_to_int, path.joinpath("token_to_int.json"))
    write_to_json(int_to_token, path.joinpath("int_to_token.json"))
    write_to_json(texts_ids_train, path.joinpath("texts_ids_train.json"))
    write_to_json(texts_ids_validation, path.joinpath("texts_ids_validation.json"))
    write_to_json(dataset_info, path.joinpath("dataset_info.json"))
    tar_folder(path)

    end_time = time.perf_counter()
    logging.info(f"Time taken: {end_time - start_time:.6f} seconds")

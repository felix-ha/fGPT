import time

from tokenizer import (
    split_tokens_raw,
    clean_tokens,
    get_unique_tokens,
    create_token_to_int_dicts,
    create_encoder,
)
from data_prep import split_corpus, texts_to_input_ids, input_ids_to_tensor
from constants import *


def print_unique_characters(corpus: str):
    unique_chars = set(corpus.replace(END_OF_TEXT, ""))
    sorted_chars = sorted(unique_chars)
    print(sorted_chars)


def replace_characters(corpus, replacment_dict):
    for char, replacement in replacment_dict.items():
        corpus = corpus.replace(char, replacement)
    return corpus


def pipeline(file_path_train, file_path_validation):
    start_time = time.perf_counter()

    # Create vocabular, i. e. token <-> int mappings
    with open(file_path_train, "r", encoding="utf8") as file:
        corpus_train_raw = file.read()
    corpus_train_clean = replace_characters(corpus_train_raw, CHARACTER_REPLACEMENTS)
    tokens_raw = split_tokens_raw(corpus_train_clean, DELIMTERS)
    tokens_all = clean_tokens(tokens_raw, TOKEN_TO_REMOVE)
    tokens_unique = get_unique_tokens(tokens_all)
    token_to_int, int_to_token = create_token_to_int_dicts(tokens_unique)
    encoder = create_encoder(token_to_int, DELIMTERS, TOKEN_TO_REMOVE)

    print_unique_characters(corpus_train_raw)
    print(f"Size of vocabulary: {len(int_to_token)}")

    # Split whole corpus after character replacments in CHARACTER_REPLACEMENTS accoring to END_OF_TEXT token
    texts_train = split_corpus(corpus_train_clean, END_OF_TEXT)
    # TODO implement handling of unknown token in data_validation.txt
    with open(file_path_validation, "r", encoding="utf8") as file:
        corpus_validation_raw = file.read()
    texts_validation = split_corpus(corpus_validation_raw, END_OF_TEXT)

    # Convert texts to input IDs
    texts_ids_train = texts_to_input_ids(texts_train, encoder)
    texts_ids_validation = texts_to_input_ids(texts_validation, encoder)

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.6f} seconds")

    X_train = input_ids_to_tensor(texts_ids_train)
    X_validation = input_ids_to_tensor(texts_ids_validation)

    assert True

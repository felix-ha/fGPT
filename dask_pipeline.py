import os
import wget
import argparse
from pathlib import Path
import pandas as pd
import dask.dataframe as dd
import pyarrow as pa
from spacy.lang.en import English
from constants import *
from data_prep import write_to_json, read_from_json
from main import get_model

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def load_file(path, ratio=1.0):
    lines = []
    with open(path, "r", encoding="utf8") as file:
        for line in file:
            lines.append(line.strip() + "\n")
    if ratio < 1.0:
        lines = lines[:int(len(lines) * ratio)]
    logging.info(f"Loaded {len(lines)} lines from {path}.")
    return "".join(lines)


nlp = English()
tokenizer = nlp.tokenizer
def tokenize_text(text):
    try:
        return [t.text for t in tokenizer(text)]
    except:
        return []
    
    
def reverse_dict(dict):
    return {value: key for key, value in dict.items()}


def load_vocabulary(vocabulary_file):
    token_to_int = read_from_json(vocabulary_file)
    int_to_token = reverse_dict(token_to_int)
    return token_to_int, int_to_token


def get_texts_ids(dataset_file):
    df = dd.read_parquet(dataset_file, dtype_backend="pyarrow")
    texts_ids = list(df['ids'].compute())
    del df
    return texts_ids



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


def datapipeline(input_file, output_path, train, n_vocab, ratio=1.0, partition_size="100MB"):
    prefix = "train" if train else "valid"
    output_path.mkdir(parents=True, exist_ok=True)

    tokenized_file = 'tokenized_train.parquet' if train else 'tokenized_valid.parquet'
    tokenized_file = output_path.joinpath(tokenized_file)

    dataset_file = 'dataset_train.parquet' if train else 'dataset_valid.parquet'
    dataset_file = output_path.joinpath(dataset_file)
    vocabulary_file = output_path.joinpath('token_to_int.json')


    texts_string = load_file(input_file, ratio=ratio)
    texts = texts_string.split(END_OF_TEXT)
    data = [{"text_raw": text} for text in texts]
    logging.info(f'loaded {len(data)} stories')
    del texts
    df = dd.from_pandas(pd.DataFrame(data), npartitions=10).repartition(partition_size=partition_size)

    df['text_clean'] = df['text_raw'].str.strip()
    for char, replacement in CHARACTER_REPLACEMENTS.items():
        df['text_clean'] = df['text_clean'].str.replace(char, replacement, regex=False)

    df['tokens']= df['text_clean'].map_partitions(lambda df: df.map(tokenize_text), meta=(None, 'object'))

    schema = pa.schema([
                ('text_clean', pa.string()),
                ('tokens',pa.list_(pa.string()))
            ])

    df[['text_clean', 'tokens']].to_parquet(tokenized_file, schema=schema)


    if train: 
        df = dd.read_parquet(tokenized_file, dtype_backend="pyarrow")
        tokens = df['tokens'].explode()
        tokens = tokens.groupby(tokens).count().nlargest(n_vocab)
        token_to_int = {token: id for id, token in zip(range(n_vocab), tokens.index)}
        token_to_int[UNK] = len(token_to_int)
        if END_OF_TEXT not in token_to_int:
            token_to_int[END_OF_TEXT] = len(token_to_int)
        write_to_json(token_to_int, vocabulary_file)
        del tokens
        
    del df

    token_to_int, int_to_token = load_vocabulary(vocabulary_file)
    df_ids = dd.read_parquet(tokenized_file, dtype_backend="pyarrow").repartition(partition_size=partition_size)
    
    def create_ids(tokens):
        return [token_to_int.get(token, token_to_int[UNK]) for token in tokens] + [token_to_int[END_OF_TEXT]]

    df_ids['ids'] = df_ids['tokens'].map_partitions(lambda df: df.map(create_ids), meta=(None, 'object'))

    schema = pa.schema([
        ('text_clean', pa.string()),
        ('ids',pa.list_(pa.int32()))
    ])

    df_ids[['text_clean', 'ids']].to_parquet(dataset_file, schema=schema)
    del df_ids

       
def create_dataset_info(dataset_file, vocabulary_file, output_path):
        token_to_int, int_to_token = load_vocabulary(vocabulary_file)
        texts_ids_train = get_texts_ids(dataset_file)
        n_stories = len(texts_ids_train)
        vocab_size = len(int_to_token)
        n_positions = max([len(text_ids) for text_ids in texts_ids_train])
        number_of_tokens = len([item for sublist in texts_ids_train for item in sublist])

        logging.info(f"Number of stories: {n_stories}")
        logging.info(f"Size of vocabulary: {vocab_size}")
        logging.info(f"Maxmial size of a text: {n_positions}")
        logging.info(f"Number of tokens in training set: {number_of_tokens}")

        dataset_info = {
            "vocab_size": vocab_size,
            "n_positions": n_positions,
            "n_tokens_training:": number_of_tokens,
            "n_stories": n_stories
        }

        write_to_json(dataset_info, output_path.joinpath("dataset_info.json"))


def data_pipeline(data_path, ratio, full):
    n_vocab = 10_000
    partition_size="100MB"

    if full:
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
        path_train = Path("data/data_train.txt")
        path_validation = Path("data/data_validation.txt")

    dataset_file = data_path.joinpath('dataset_train.parquet')
    vocabulary_file = data_path.joinpath('token_to_int.json')

    datapipeline(path_train, data_path, train=True, n_vocab=n_vocab, ratio=ratio, partition_size=partition_size)
    create_dataset_info(dataset_file, vocabulary_file, data_path)
    datapipeline(path_validation, data_path, train=False, n_vocab=n_vocab, ratio=ratio, partition_size=partition_size)


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
        help="Ratio of the data to use for processing.",
    )
    args = parser.parse_args()

    data_path = Path('datapipeline')

    data_pipeline(data_path, args.ratio, args.full)

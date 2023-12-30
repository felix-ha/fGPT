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

from tqdm import tqdm
from transformers import GPT2TokenizerFast, AutoTokenizer

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def get_texts_ids(dataset_file):
    df = dd.read_parquet(dataset_file, dtype_backend="pyarrow")
    texts_ids = list(df["ids"].compute())
    del df
    return texts_ids


def load_file(input_file, output_path, n_texts_per_partition):
    lines = []
    n_texts_current = 0
    partition = 0

    schema = pa.schema([("text_clean", pa.string())])

    with open(input_file, "r", encoding="utf8") as file:
        for line in file:
            lines.append(line.strip() + "\n")
            if END_OF_TEXT in line:
                n_texts_current += 1
            if n_texts_current == n_texts_per_partition:
                write_partition(lines, schema, partition, output_path)
                lines = []
                n_texts_current = 0
                partition += 1
        # write last lines
        write_partition(lines, schema, partition, output_path)


def write_partition(lines, schema, partition, output_path):
    texts = "".join(lines).split(END_OF_TEXT)
    data = [{"text_raw": text} for text in texts]
    df = dd.from_pandas(pd.DataFrame(data), npartitions=4)
    df["text_clean"] = df["text_raw"].map_partitions(
        lambda df: df.map(clean_text), meta=(None, "object")
    )
    df[["text_clean"]].to_parquet(
        output_path.joinpath(f"{partition}.parquet"), schema=schema
    )


def clean_text(text):
    text = text.strip()
    for char, replacement in CHARACTER_REPLACEMENTS.items():
        text = text.replace(char, replacement)
    return text


def get_texts_of_partition(text_file, partition):
    df = dd.read_parquet(
        text_file.joinpath(f"{partition}.parquet"), columns=["text_clean"]
    )
    return list(df["text_clean"].compute())


def batch_iterator(text_file, n_partitions):
    for i in tqdm(range(0, n_partitions)):
        yield get_texts_of_partition(text_file, i)


def datapipeline(
    input_file,
    output_path,
    train,
    n_vocab,
    n_texts_per_partition,
    partition_size
):
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_path.joinpath("tokenizer")
    data_info_file = output_path.joinpath("dataset_info.json")
    text_file = "text_train.parquet" if train else "text_valid.parquet"
    text_file = output_path.joinpath(text_file)
    dataset_file = "dataset_train.parquet" if train else "dataset_valid.parquet"
    dataset_file = output_path.joinpath(dataset_file)
    hf_dir = "hf"

    load_file(input_file, text_file, n_texts_per_partition)
    logging.info("written raw data")

    if train:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir=hf_dir)
        tokenizer = tokenizer.train_new_from_iterator(
            text_iterator=batch_iterator(text_file, 1), vocab_size=n_vocab
        )
        tokenizer.save_pretrained(tokenizer_path, cache_dir=hf_dir)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    df = dd.read_parquet(text_file, dtype_backend="pyarrow").repartition(
        partition_size=partition_size
    )
    df["ids"] = df["text_clean"].map_partitions(
        lambda df: df.map(tokenizer.encode), meta=(None, "object")
    )

    schema = pa.schema([("text_clean", pa.string()), ("ids", pa.list_(pa.int32()))])

    df[["text_clean", "ids"]].to_parquet(dataset_file, schema=schema)

    if train:
        df = dd.read_parquet(dataset_file, dtype_backend="pyarrow")
        df["length"] = df["ids"].map_partitions(
            lambda df: df.map(len), meta=(None, "int")
        )
        n_stories = len(df)
        number_of_tokens = df["length"].sum().compute()
        n_positions = df["length"].max().compute()

        logging.info(f"Number of stories: {n_stories}")
        logging.info(f"Size of vocabulary: {n_vocab}")
        logging.info(f"Maxmial size of a text: {n_positions}")
        logging.info(f"Number of tokens in training set: {number_of_tokens}")

        dataset_info = {
            "vocab_size": n_vocab,
            "n_positions": int(n_positions),
            "n_tokens_training:": int(number_of_tokens),
            "n_stories": int(n_stories),
        }
        write_to_json(dataset_info, data_info_file)


def data_pipeline(data_path, full, n_vocab, n_texts_per_partition, partition_size):

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
        path_train = "data/data_train.txt"
        path_validation = "data/data_validation.txt"

    datapipeline(
        path_train,
        data_path,
        train=True,
        n_vocab=n_vocab,
        n_texts_per_partition=n_texts_per_partition,
        partition_size=partition_size,
    )
    datapipeline(
        path_validation,
        data_path,
        train=False,
        n_vocab=n_vocab,
        n_texts_per_partition=n_texts_per_partition,
        partition_size=partition_size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        default=False,
        action="store_true",
        help="Use full TinyStores dataset instead of the small one.",
    )
    args = parser.parse_args()

    data_path = Path("datapipeline")
    n_vocab = 10_000
    partition_size = "100MB"
    n_texts_per_partition = 100_000

    data_pipeline(data_path, args.full, n_vocab, n_texts_per_partition, partition_size)

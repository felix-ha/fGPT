from pathlib import Path

import dask.dataframe as dd
import dask
dask.config.set({"dataframe.convert-string": False})
import pyarrow as pa
import pandas as pd
from constants import *
from data_prep import load_file, split_corpus, write_to_json, read_from_json
from tokenizer import create_encoder, create_decoder
import torch
from torch.utils.data import DataLoader
from data_prep import collate_fn
from dionysus.training import TrainingConfig, train
from model import (
    LanguageModelDataset,
    simpleGPT,
    cross_entropy_language_model,
    generate,
)


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


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



def load_file_polars(path):
    lines = []
    with open(path, "r", encoding="utf8") as file:
        for line in file:
            lines.append(line.strip() + "\n")
    logging.info(f"Loaded {len(lines)} lines from {path}.")
    return lines


def replace_characters(text, replacment_dict):
    for char, replacement in replacment_dict.items():
        text = text.replace(char, replacement)
    return text

from spacy.lang.en import English

nlp = English()
tokenizer = nlp.tokenizer
def tokenize_text(text):
    try:
        return [t.text for t in tokenizer(text)]
    except:
        return []


def clean_tokens(tokens_raw: list[str], tokens_to_remove: list[str]=TOKEN_TO_REMOVE) -> list[str]:
    result = [token for token in tokens_raw if token not in tokens_to_remove]
    return result


def reverse_dict(dict):
    return {value: key for key, value in dict.items()}


def create_vocabulary(tokenized_file, vocabulary_file, n_vocab):
    if vocabulary_file.is_file():
        logging.info("vocabulary was already created")
    else:
        logging.info('creating vocabulary')
        df = dd.read_parquet(tokenized_file, dtype_backend="pyarrow")
        tokens = df['tokens'].explode()
        tokens = tokens.groupby(tokens).count().nlargest(n_vocab)
        token_to_int = {token: id for id, token in zip(range(n_vocab), tokens.index)}
        token_to_int[UNK] = len(token_to_int)
        if END_OF_TEXT not in token_to_int:
            token_to_int[END_OF_TEXT] = len(token_to_int)
        write_to_json(token_to_int, vocabulary_file)
        del df


def load_vocabulary(vocabulary_file):
    token_to_int = read_from_json(vocabulary_file)
    int_to_token = reverse_dict(token_to_int)
    return token_to_int, int_to_token


def get_texts_ids(dataset_file):
    df = dd.read_parquet(dataset_file, dtype_backend="pyarrow")
    texts_ids = list(df['ids'].compute())
    del df
    return texts_ids


def create_dataset(input_file, output_path, n_vocab, train):
    prefix = "train" if train else "valid"
    output_path.joinpath(prefix).mkdir(parents=True, exist_ok=True)
    output_file = output_path.joinpath(prefix).joinpath("data.parquet")
    replaced_file = output_path.joinpath(prefix).joinpath("replaced.parquet")
    tokenized_file = output_path.joinpath(prefix).joinpath("tokenzied.parquet")
    vocabulary_file = output_path.joinpath('token_to_int.json')
    dataset_train_file = 'dataset_train.parquet' if train else 'dataset_valid.parquet'
    dataset_train_file = output_path.joinpath(dataset_train_file)

    if output_file.is_file() or output_file.is_dir():
        logging.info('read raw texts')
        df = dd.read_parquet(output_file)
    else:
        logging.info("creating raw texts")
        corpus_train_raw = load_file_polars(input_file)
        result = "".join(corpus_train_raw)
        del corpus_train_raw
        result = result.split(END_OF_TEXT)
        data = [{"text": text + f" {END_OF_TEXT}"} for text in result]
        data = data[:250_000]
        del result
        logging.info("creating dataframe for raw texts")
        df = dd.from_pandas(pd.DataFrame(data), npartitions=100)
        df.to_parquet(output_file)

    if replaced_file.is_file() or replaced_file.is_dir():
        logging.info("read placed texts")
        df = dd.read_parquet(replaced_file)
    else:
        logging.info("creating replaced texts")
        df['text'] = df['text'].str.strip()
        for char, replacement in CHARACTER_REPLACEMENTS.items():
            df['text'] = df['text'].str.replace(char, replacement, regex=False)
        logging.info("writing parquet for replaced texts")
        df.to_parquet(replaced_file, overwrite=True)


    if tokenized_file.is_file() or tokenized_file.is_dir():
        logging.info("reading tokenized texts")
        df = dd.read_parquet(tokenized_file, dtype_backend="pyarrow")
    else:
        df['tokens']= df['text'].apply(lambda t: tokenize_text(t))
        logging.info("tokenize texts")
        schema = pa.schema([
            ('text', pa.string()),
            ('tokens',pa.list_(pa.string()))
        ])
        logging.info('writing parquet for tokenized texts')
        df.to_parquet(tokenized_file, schema=schema)
        df = dd.read_parquet(tokenized_file, dtype_backend="pyarrow")


    if train:
        create_vocabulary(tokenized_file, vocabulary_file, n_vocab)

    token_to_int, int_to_token = load_vocabulary(vocabulary_file)

    # Create IDs for all texts

    def create_ids(tokens):
        return [token_to_int.get(token, token_to_int[UNK]) for token in tokens]

    if dataset_train_file.is_file() or dataset_train_file.is_dir():
        logging.info("dataset was already created")
    else:
        logging.info("creating dataset")
        df_train = dd.read_parquet(tokenized_file).repartition(partition_size="100MB")
        df_train['ids'] = df_train['tokens'].map_partitions(lambda df: df.map(create_ids))
        schema = pa.schema([
            ('text', pa.string()),
            ('ids',pa.list_(pa.int32()))
        ])
        df_train.to_parquet(dataset_train_file, schema=schema)


    if train:
        texts_ids_train = get_texts_ids(dataset_train_file)

        vocab_size = len(int_to_token)
        n_positions = max([len(text_ids) for text_ids in texts_ids_train])
        number_of_tokens = len([item for sublist in texts_ids_train for item in sublist])

        logging.info(f"Size of vocabulary: {vocab_size}")
        logging.info(f"Maxmial size of a text: {n_positions}")
        logging.info(f"Number of tokens in training set: {number_of_tokens}")

        dataset_info = {
            "vocab_size": vocab_size,
            "n_positions": n_positions,
            "n_tokens_training:": number_of_tokens,
        }

        write_to_json(dataset_info, output_path.joinpath("dataset_info.json"))


def pipeline():
    n_vocab = 10_000

    path = Path('datapipeline')
    path.mkdir(parents=True, exist_ok=True)

    # TODO move paths to constants
    input_train_file = "data/TinyStoriesV2-GPT4-train.txt"
    input_valid_file = "data/TinyStoriesV2-GPT4-valid.txt"

    create_dataset(input_train_file, path, n_vocab, train=True)
    create_dataset(input_valid_file, path, n_vocab, train=False)

    # main----------------------

    dataset_train_file = path.joinpath('dataset_train.parquet')
    dataset_valid_file = path.joinpath('dataset_valid.parquet')
    vocabulary_file = path.joinpath('token_to_int.json')

    token_to_int, int_to_token = load_vocabulary(vocabulary_file)
    texts_ids_train = get_texts_ids(dataset_train_file)
    texts_ids_validation = get_texts_ids(dataset_valid_file)

    dataset_info = read_from_json(path.joinpath("dataset_info.json"))
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

    device = "gpu" if torch.cuda.is_available() else "cpu"

    model = get_model(vocab_size, n_positions, device)

    loss_func = cross_entropy_language_model

    quit()

    torch.manual_seed(0)
    train_config = TrainingConfig(
        model=model,
        epochs=1,
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
        checkpoint_step_batch=500,
        checkpoint_path = None
    )

    logging.info(f"maybe set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32")

    print("done")
    quit()
    train(train_config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of parameters: {num_params}")

if __name__ == "__main__":
    pipeline()

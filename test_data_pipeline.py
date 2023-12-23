from pathlib import Path
import tempfile
import pandas as pd

from dask_pipeline import data_pipeline
from data_prep import read_from_json
from constants import *

token_to_int_expected = {
    ".": 0,
    "the": 1,
    "was": 2,
    "in": 3,
    "Tom": 4,
    "Jenny": 5,
    "living": 6,
    "room": 7,
    "bathroom": 8,
    "bedroom": 9,
    "cleaning": 10,
    "cooking": 11,
    "kitchen": 12,
    "playing": 13,
    "running": 14,
    "sleeping": 15,
    "<|unk|>": 16,
    "<|endoftext|>": 17,
}
dataset_info_expected = {
    "vocab_size": 18,
    "n_positions": 9,
    "n_tokens_training:": 42,
    "n_stories": 6,
}

df_tokenized_train_expected = pd.DataFrame(
    {
        "text_clean": {
            0: "Jenny was playing in the living room.",
            1: "Jenny was running in the living room.",
            2: "Tom was sleeping in the bedroom.",
            3: "Tom was cooking in the kitchen.",
            4: "Tom was cleaning the bathroom.",
            5: "",
        },
        "tokens": {
            0: ["Jenny", "was", "playing", "in", "the", "living", "room", "."],
            1: ["Jenny", "was", "running", "in", "the", "living", "room", "."],
            2: ["Tom", "was", "sleeping", "in", "the", "bedroom", "."],
            3: ["Tom", "was", "cooking", "in", "the", "kitchen", "."],
            4: ["Tom", "was", "cleaning", "the", "bathroom", "."],
            5: [],
        },
    }
)

df_tokenized_valid_expected = pd.DataFrame(
    {
        "text_clean": {
            0: "Jenny was sleeping in the bedroom.",
            1: "Tom was playing in the living room.",
            2: "Tom was running in the living room.",
            3: "",
        },
        "tokens": {
            0: ["Jenny", "was", "sleeping", "in", "the", "bedroom", "."],
            1: ["Tom", "was", "playing", "in", "the", "living", "room", "."],
            2: ["Tom", "was", "running", "in", "the", "living", "room", "."],
            3: [],
        },
    }
)

df_train_expected = pd.DataFrame(
    {
        "text_clean": {
            0: "Jenny was playing in the living room.",
            1: "Jenny was running in the living room.",
            2: "Tom was sleeping in the bedroom.",
            3: "Tom was cooking in the kitchen.",
            4: "Tom was cleaning the bathroom.",
            5: "",
        },
        "ids": {
            0: [5, 2, 13, 3, 1, 6, 7, 0, 17],
            1: [5, 2, 14, 3, 1, 6, 7, 0, 17],
            2: [4, 2, 15, 3, 1, 9, 0, 17],
            3: [4, 2, 11, 3, 1, 12, 0, 17],
            4: [4, 2, 10, 1, 8, 0, 17],
            5: [17],
        },
    }
)

df_valid_expected = pd.DataFrame(
    {
        "text_clean": {
            0: "Jenny was sleeping in the bedroom.",
            1: "Tom was playing in the living room.",
            2: "Tom was running in the living room.",
            3: "",
        },
        "ids": {
            0: [5, 2, 15, 3, 1, 9, 0, 17],
            1: [4, 2, 13, 3, 1, 6, 7, 0, 17],
            2: [4, 2, 14, 3, 1, 6, 7, 0, 17],
            3: [17],
        },
    }
)


def test_training():
    with tempfile.TemporaryDirectory() as path_data:
        dataset_info_file = Path(path_data).joinpath("dataset_info.json")
        vocabulary_file = Path(path_data).joinpath("token_to_int.json")
        tokenized_train_file = Path(path_data).joinpath("tokenized_train.parquet")
        tokenized_valid_file = Path(path_data).joinpath("tokenized_valid.parquet")
        dataset_train_file = Path(path_data).joinpath("dataset_train.parquet")
        dataset_vaild_file = Path(path_data).joinpath("dataset_valid.parquet")

        data_pipeline(Path(path_data), ratio=1.0, full=False)

        assert read_from_json(vocabulary_file) == token_to_int_expected
        assert read_from_json(dataset_info_file) == dataset_info_expected

        df_expected = df_tokenized_train_expected.explode("tokens")
        df_actual = pd.read_parquet(tokenized_train_file).explode("tokens")
        assert df_expected.equals(
            df_actual
        ), "tokenized_train.parquet is not as expected"

        df_expected = df_tokenized_valid_expected.explode("tokens")
        df_actual = pd.read_parquet(tokenized_valid_file).explode("tokens")
        assert df_expected.equals(
            df_actual
        ), "tokenized_valid.parquet is not as expected"

        df_expected = df_train_expected.explode("ids")
        df_actual = pd.read_parquet(dataset_train_file).explode("ids")
        df_actual["text_clean"] = df_actual["text_clean"].astype(str)
        assert df_expected.equals(df_actual), "dataset_train.parquet is not as expected"

        df_expected = df_valid_expected.explode("ids")
        df_actual = pd.read_parquet(dataset_vaild_file).explode("ids")
        df_actual["text_clean"] = df_actual["text_clean"].astype(str)
        assert df_expected.equals(df_actual), "dataset_valid.parquet is not as expected"


if __name__ == "__main__":
    test_training()

from pathlib import Path
import tempfile
import pandas as pd

from fgpt.data import data_pipeline, get_texts_ids, load_vocabulary
from data_prep import read_from_json
from fgpt.constants import *

from dionysus.training import TrainingConfig, train
import torch
from torch.utils.data import DataLoader
from data_prep import collate_fn
from model import (
    LanguageModelDataset,
    simpleGPT,
    cross_entropy_language_model,
    generate
)
from main import get_model
from tokenizer import create_encoder, create_decoder
from inference import load_model

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
).convert_dtypes()

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
).convert_dtypes()

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

        data_pipeline(Path(path_data), full=False)

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


        data_path = Path(path_data)
        texts_ids_train = get_texts_ids(data_path.joinpath('dataset_train.parquet'))
        texts_ids_validation = get_texts_ids(data_path.joinpath('dataset_valid.parquet'))

        vocabulary_file = data_path.joinpath('token_to_int.json')
        token_to_int, int_to_token = load_vocabulary(vocabulary_file)
        dataset_info = read_from_json(data_path.joinpath("dataset_info.json"))

        vocab_size = dataset_info["vocab_size"]
        n_positions = dataset_info["n_positions"]

        dataset_train = LanguageModelDataset(texts_ids_train)
        dataset_validation = LanguageModelDataset(texts_ids_validation)

        dataloader_train = DataLoader(
            dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn
        )
        dataloader_validation = DataLoader(
            dataset_validation, batch_size=8, shuffle=False, collate_fn=collate_fn
        )

        stop_token_id = token_to_int[END_OF_TEXT]

        device = "cpu"

        model = get_model(vocab_size, n_positions, device)

        loss_func = cross_entropy_language_model


        with tempfile.TemporaryDirectory() as save_path:
            torch.manual_seed(0)
            model_name = "fGPT"
            checkpoint_path = Path(save_path).joinpath(model_name).joinpath('last/model.pt')
            train_config = TrainingConfig(
                model=model,
                epochs=2,
                loss_func=loss_func,
                training_loader=dataloader_train,
                validation_loader=dataloader_validation,
                optimizer="AdamW",
                device=device,
                force_write_logs=False,
                save_model=True,
                tar_result=True,
                save_path=save_path,
                model_name=model_name,
                progress_bar=True,
                checkpoint_step=1,
                checkpoint_step_batch=1,
                checkpoint_path = checkpoint_path if checkpoint_path.is_file() else None
            )

            train(train_config)


            model_dict_file = checkpoint_path
            dataset_info_path = dataset_info_file

            token_to_int, int_to_token = load_vocabulary(vocabulary_file)
            encoder = create_encoder(token_to_int, END_OF_TEXT, TOKEN_TO_REMOVE, UNK)
            decoder = create_decoder(int_to_token)
            stop_token_id = token_to_int[END_OF_TEXT]

            dataset_info = read_from_json(dataset_info_path)
            vocab_size = dataset_info["vocab_size"]
            n_positions = dataset_info["n_positions"]    

            model = load_model(model_dict_file, vocab_size, n_positions)
            
            with torch.no_grad():
                model.eval()
                prompt = "Tom was "
                output, _ = generate(
                    model,
                    prompt,
                    encoder,
                    decoder,
                    stop_token_id=stop_token_id,
                    max_n=5,
                    choices_per_step=1,
                    sample=True,
                    temperature=1,
                )

            assert len(output) > 0



if __name__ == "__main__":
    test_training()

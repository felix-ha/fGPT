"""
This module contains a unit tests for the complete data pipeline.
As a final test, the output will be fed into the transformer.
"""

import pytest
import os
import tempfile
from data_pipeline import pipeline
from dionysus.training import TrainingConfig, train
from model import (
    LanguageModelDataset,
    simpleGPT,
    cross_entropy_language_model,
    generate,
)
from data_prep import read_from_json, get_token_int_dicts
from constants import *
from torch.utils.data import DataLoader
from data_prep import collate_fn
from tokenizer import create_encoder, create_decoder


with tempfile.TemporaryDirectory() as path_data:

    @pytest.fixture
    def data():
        pipeline("data/data_train.txt", "data/data_validation.txt", path_data)

    def test_training(data):
        with tempfile.TemporaryDirectory() as temp_dir:
            token_to_int, int_to_token = get_token_int_dicts(path_data)
            texts_ids_train = read_from_json(
                os.path.join(path_data, "texts_ids_train.json")
            )
            texts_ids_validation = read_from_json(
                os.path.join(path_data, "texts_ids_validation.json")
            )

            dataset_info = read_from_json(os.path.join(path_data, "dataset_info.json"))
            vocab_size = dataset_info["vocab_size"]
            n_positions = dataset_info["n_positions"]

            assert len(token_to_int) == 25
            assert len(int_to_token) == 25
            assert len(texts_ids_train) == 5
            assert len(texts_ids_validation) == 3

            assert n_positions == 8
            assert vocab_size == 25

            device = "cpu"

            encoder = create_encoder(token_to_int, END_OF_TEXT, TOKEN_TO_REMOVE, UNK)
            decoder = create_decoder(int_to_token)

            dataset_train = LanguageModelDataset(texts_ids_train)
            dataset_validation = LanguageModelDataset(texts_ids_validation)

            dataloader_train = DataLoader(
                dataset_train, batch_size=2, shuffle=True, collate_fn=collate_fn
            )
            dataloader_validation = DataLoader(
                dataset_validation, batch_size=2, shuffle=False, collate_fn=collate_fn
            )

            assert len(dataloader_train) == 3
            assert len(dataloader_validation) == 2

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
                epochs=2,
                loss_func=loss_func,
                training_loader=dataloader_train,
                validation_loader=dataloader_validation,
                optimizer="AdamW",
                device=device,
                force_write_logs=False,
                save_model=True,
                tar_result=True,
                save_path=temp_dir,
                model_name="GPT-2",
                progress_bar=True,
            )

            train(train_config)

            prompt = "Tom was"
            output, choices = generate(
                model,
                prompt,
                encoder,
                decoder,
                stop_token_id=stop_token_id,
                max_n=3,
                choices_per_step=3,
            )

            assert choices.shape[1] == 4

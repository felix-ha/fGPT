"""
This module contains a unit tests for the complete data pipeline.
As a final test, the output will be fed into the transformer.
"""

import pytest
import tempfile
from data_pipeline import pipeline
from dionysus.training import TrainingConfig, train
from model import simpleGPT, cross_entropy_language_model


@pytest.fixture
def data():
    value = pipeline("data/data_train.txt", "data/data_validation.txt")
    yield value


@pytest.mark.dependency()
def test_data_pipeline(data):
    assert data.n_positions == 9
    assert len(data.dataloader_train) == 3
    assert len(data.dataloader_validation) == 2


@pytest.mark.dependency(depends=["test_data_pipeline"])
def test_training(data):
    with tempfile.TemporaryDirectory() as temp_dir:
        model = simpleGPT(
            data.vocab_size,
            n_embd=10,
            num_heads=2,
            block_size=data.n_positions,
            n_layer=2,
            dropout=0.1,
            device="cpu",
        )

        loss_func = cross_entropy_language_model

        train_config = TrainingConfig(
            model=model,
            epochs=5,
            loss_func=loss_func,
            training_loader=data.dataloader_train,
            validation_loader=data.dataloader_validation,
            optimizer="AdamW",
            device="gpu",
            save_model=True,
            tar_result=True,
            save_path=temp_dir,
            model_name="GPT-2",
            progress_bar=False,
        )

        train(train_config)

        assert True

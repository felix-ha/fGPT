"""
This module contains a unit tests for the complete data pipeline.
As a final test, the output will be fed into the transformer.
"""

import pytest
from data_pipeline import pipeline
from transformers import GPT2Config, GPT2LMHeadModel


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
    gpt2_config = GPT2Config(
        n_positions=data.n_positions,
        vocab_size=data.vocab_size,
        n_embd=5,
        n_inner=5,
        n_layer=1,
        n_head=1,
    )

    model = GPT2LMHeadModel(gpt2_config)

    # y_logits = model(x).logits
    # assert y_logits.shape == (batch_size, context, data.vocab_size)

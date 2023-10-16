import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from model import cross_entropy_language_model

# TODO fixt test
def test_cross_entropy_language_model():
    vocab_size = 5
    batch_size = 1
    context = 4
    x = torch.zeros((batch_size, context, vocab_size))
    y = torch.zeros((batch_size, context), dtype=torch.long)

    loss = cross_entropy_language_model(x, y)
    assert loss.item() == 1.6094379425048828


# TODO compare output of custom transformer with GPT2LMHeadModel
@pytest.mark.skip()
def test_lm_head_dims():
    vocab_size = 5
    batch_size = 1
    context = 4
    x = torch.zeros((batch_size, context), dtype=torch.long)

    gpt2_config = GPT2Config(
        n_positions=5,
        vocab_size=vocab_size,
        n_embd=5,
        n_inner=5,
        n_layer=1,
        n_head=1,
    )

    model = GPT2LMHeadModel(gpt2_config)

    y_logits = model(x).logits
    assert y_logits.shape == (batch_size, context, vocab_size)

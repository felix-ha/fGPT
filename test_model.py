import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from model import cross_entropy_language_model, generate
from tokenizer import create_encoder, create_decoder


END_OF_TEXT = "<|endoftext|>"
UNK = "<|unk|>"


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


class MockModel:
    def __init__(self):
        self.idx = 0

    def __call__(self, x):
        # TODO fix output, needs to be batched
        if self.idx == 0:
            self.idx += 1
            return torch.tensor([[[2, 4, 7, 1, 1]]], dtype=torch.float32)
        if self.idx == 1:
            self.idx += 1
            return torch.tensor([[[1, 1, 1, 8, 2]]], dtype=torch.float32)
        if self.idx == 2:
            self.idx += 1
            return torch.tensor([[[3, 1, 1, 1, 2]]], dtype=torch.float32)
        if self.idx == 3:
            self.idx += 1
            return torch.tensor([[[1, 1, 1, 1, 2]]], dtype=torch.float32)

        return x


def test_generate():
    token_to_index = {"a": 0, "b": 1, "c": 2, "d": 3, END_OF_TEXT: 4, UNK: 5}
    index_to_token = {v: k for k, v in token_to_index.items()}
    encoder = create_encoder(
        token_to_index, END_OF_TEXT, tokens_to_remove=[" "], unk=UNK
    )
    decoder = create_decoder(index_to_token)

    model = MockModel()

    prompt = "a b"

    output, _ = generate(
        model, prompt, encoder, decoder, stop_token_id=4, max_n=10, choices_per_step=3
    )

    assert output == f"c d a {END_OF_TEXT} "

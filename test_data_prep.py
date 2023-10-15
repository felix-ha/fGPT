import torch
from data_prep import (
    split_corpus,
    texts_to_input_ids,
    input_ids_to_tensor,
    LanguageModelDataset,
)
from tokenizer import create_encoder

END_OF_TEXT = "<|endoftext|>"


def test_split_corpus():
    with open("data/data_prep_1.txt", "r", encoding="utf8") as file:
        file_content = file.read()
        result = split_corpus(file_content, END_OF_TEXT)
        assert len(result) == 6
        for text in result:
            assert text == f"This is a test.{END_OF_TEXT}"


def test_texts_to_input_ids():
    token_to_int = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    encoder = create_encoder(token_to_int, delimiters=[" "], tokens_to_remove=[" "])
    input = ["A B C", "C B A", "A B C D E", "E D A B C"]
    text_ids_actual = texts_to_input_ids(input, encoder)
    text_ids_expected = [[0, 1, 2], [2, 1, 0], [0, 1, 2, 3, 4], [4, 3, 0, 1, 2]]
    assert text_ids_actual == text_ids_expected


def test_input_ids_to_tensor_one_dim():
    input_ids = [1, 2]
    X_actual = input_ids_to_tensor(input_ids)
    X_expected = torch.tensor([[1, 2]])
    assert torch.equal(X_actual, X_expected)


def test_input_ids_to_tensor():
    input_ids = [[1, 2], [1], [1, 2, 3]]
    X_actual = input_ids_to_tensor(input_ids)
    X_expected = torch.tensor([[1, 2, 0], [1, 0, 0], [1, 2, 3]])
    assert torch.equal(X_actual, X_expected)


def test_dataset():
    input_ids = [[1, 2], [1], [1, 2, 3]]
    dataset = LanguageModelDataset(input_ids)
    assert len(dataset) == 3
    assert torch.equal(dataset[0], torch.tensor([[1, 2]]))
    assert torch.equal(dataset[1], torch.tensor([[1]]))
    assert torch.equal(dataset[2], torch.tensor([[1, 2, 3]]))
    assert torch.equal(dataset[-1], torch.tensor([[1, 2, 3]]))
    X_batch_expected = torch.tensor([[1, 2, 0], [1, 0, 0], [1, 2, 3]])
    assert torch.equal(dataset[:], X_batch_expected)

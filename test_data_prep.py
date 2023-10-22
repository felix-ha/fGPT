import pytest
import torch
import tempfile
from pathlib import Path
from data_prep import (
    load_file,
    split_corpus,
    texts_to_input_ids,
    input_ids_to_tensor,
    collate_fn,
    write_to_json,
    read_from_json,
    get_token_int_dicts,
)
from torch.utils.data import DataLoader
from model import LanguageModelDataset
from tokenizer import create_encoder


END_OF_TEXT = "<|endoftext|>"
UNK = "<|unk|>"
END_OF_TEXT_ID = 99
PADDING_ID = 0


def test_serialize_deserialize_texts_id():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        texts_ids_train = [[1, 2, 3], [4, 5, 6]]
        texts_ids_validation = [[7, 8, 9], [10, 11, 12]]

        write_to_json(texts_ids_train, temp_dir.joinpath("texts_ids_train.json"))
        texts_ids_train_read = read_from_json(temp_dir.joinpath("texts_ids_train.json"))

        write_to_json(
            texts_ids_validation, temp_dir.joinpath("texts_ids_validation.json")
        )
        texts_ids_validation_read = read_from_json(
            temp_dir.joinpath("texts_ids_validation.json")
        )

        assert texts_ids_train == texts_ids_train_read
        assert texts_ids_validation == texts_ids_validation_read


def test_serialize_deserialize_token_int_dicts():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        token_to_int = {"A": 0, "B": 1}
        int_to_token = {0: "A", 1: "B"}

        write_to_json(token_to_int, temp_dir.joinpath("token_to_int.json"))
        write_to_json(int_to_token, temp_dir.joinpath("int_to_token.json"))

        token_to_int_read, int_to_token_read = get_token_int_dicts(temp_dir)

        assert token_to_int == token_to_int_read
        assert int_to_token == int_to_token_read


def test_load_data():
    path = "data/data_prep_1.txt"

    # join all lines into one string
    content_actual = load_file(path, ratio=1.0)
    print(content_actual)

    content_expected = """This is a test.
<|endoftext|>
This is a test.
<|endoftext|>

This is a test.
<|endoftext|> This is a test.
<|endoftext|>This is a test. <|endoftext|>This is a test.<|endoftext|>
"""
    assert content_actual == content_expected


def test_split_corpus():
    with open("data/data_prep_1.txt", "r", encoding="utf8") as file:
        file_content = file.read()
        result = split_corpus(file_content, END_OF_TEXT)
        assert len(result) == 6
        for text in result:
            assert text == f"This is a test.{END_OF_TEXT}"


def test_texts_to_input_ids():
    token_to_int = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, UNK: 5}
    encoder = create_encoder(
        token_to_int, delimiters=[" "], tokens_to_remove=[" "], unk=UNK
    )
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
    X_actual = input_ids_to_tensor(input_ids, PADDING_ID)
    X_expected = torch.tensor(
        [[1, 2, PADDING_ID], [1, PADDING_ID, PADDING_ID], [1, 2, 3]]
    )
    assert torch.equal(X_actual, X_expected)


def test_dataset():
    input_ids = [
        [1, 2, 3, END_OF_TEXT_ID],
        [4, 5, END_OF_TEXT_ID],
        [1, 2, 3, 4, 5, 6, END_OF_TEXT_ID],
    ]
    dataset = LanguageModelDataset(input_ids)
    assert len(dataset) == 3

    x_1, y_1 = dataset[0]
    assert x_1 == [1, 2, 3]
    assert y_1 == [2, 3, END_OF_TEXT_ID]

    x_2, y_2 = dataset[1]
    assert x_2 == [4, 5]
    assert y_2 == [5, END_OF_TEXT_ID]

    x_3, y_3 = dataset[2]
    assert x_3 == [1, 2, 3, 4, 5, 6]
    assert y_3 == [2, 3, 4, 5, 6, END_OF_TEXT_ID]


# slicing is not implemented
def test_dataset_slicing():
    input_ids = [
        [1, 2, 3, END_OF_TEXT_ID],
        [4, 5, END_OF_TEXT_ID],
        [1, 2, 3, 4, 5, 6, END_OF_TEXT_ID],
    ]
    dataset = LanguageModelDataset(input_ids)

    x_batch, y_batch = dataset[0:2]
    with pytest.raises(AssertionError):
        assert x_batch == [[1, 2, 3], [4, 5]]
        assert y_batch != [[2, 3, END_OF_TEXT_ID], [5, END_OF_TEXT_ID]]


def test_dataloader_single_batch():
    input_ids = [
        [1, 2, 3, END_OF_TEXT_ID],
        [4, 5, END_OF_TEXT_ID],
        [1, 2, 3, 4, 5, 6, END_OF_TEXT_ID],
    ]
    dataset = LanguageModelDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    for batch in dataloader:
        x, y = batch
        assert torch.equal(x, torch.tensor([[1, 2, 3], [4, 5, PADDING_ID]]))
        assert torch.equal(
            y, torch.tensor([[2, 3, END_OF_TEXT_ID], [5, END_OF_TEXT_ID, PADDING_ID]])
        )
        break


def test_dataloader_last_batch():
    input_ids = [
        [1, 2, 3, END_OF_TEXT_ID],
        [4, 5, END_OF_TEXT_ID],
        [1, 2, 3, 4, 5, 6, END_OF_TEXT_ID],
    ]
    dataset = LanguageModelDataset(input_ids)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    for batch in dataloader:
        x, y = batch
    assert torch.equal(x, torch.tensor([[1, 2, 3, 4, 5, 6]]))
    assert torch.equal(y, torch.tensor([[2, 3, 4, 5, 6, END_OF_TEXT_ID]]))


@pytest.mark.skip()
def test_load_text_file():
    file_path_train = "./data/data_train.txt"
    with open(file_path_train, "r", encoding="utf8") as file:
        corpus_train_raw = file.read()

    corpus_train_raw_lines = []
    with open(file_path_train, "r", encoding="utf8") as file:
        for line in file:
            corpus_train_raw_lines.append(line)
    corpus_train_raw_lines_full = " ".join(corpus_train_raw_lines)

    assert corpus_train_raw == corpus_train_raw_lines_full

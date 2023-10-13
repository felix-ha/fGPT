from tokenizer import (
    split_tokens_raw,
    clean_tokens,
    get_unique_tokens,
    create_token_to_int_dicts,
)

END_OF_TEXT = "<|endoftext|>"
TOKEN_TO_REMOVE = ["", " "]
DELIMTERS = [" ", END_OF_TEXT, "\n", ".", ",", ";", ":", "!", "?", '"']


def test_split_tokens_raw_1():
    with open("data/token_1.txt", "r", encoding="utf8") as file:
        file_content = file.read()
        tokens_raw_actual = split_tokens_raw(file_content, DELIMTERS)
        tokens_raw_expected = [
            "This",
            " ",
            "is",
            " ",
            "a",
            " ",
            "test",
            ".",
            "",
            "\n",
            "",
            END_OF_TEXT,
            "",
            "\n",
            "",
            "\n",
            "Start",
        ]
        assert tokens_raw_actual == tokens_raw_expected


def test_split_tokens_raw_2():
    with open("data/token_2.txt", "r") as file:
        file_content = file.read()
        tokens_raw_actual = split_tokens_raw(file_content, DELIMTERS)
        tokens_raw_expected = [
            "Tim",
            ":",
            "",
            " ",
            "",
            '"',
            "Let's",
            " ",
            "go",
            ",",
            "",
            " ",
            "John",
            "!",
            "",
            " ",
            "Ok",
            "?",
            "",
            '"',
            "",
        ]
        assert tokens_raw_actual == tokens_raw_expected


def test_clean_tokens():
    tokens_raw = [
        "This",
        " ",
        "is",
        " ",
        "a",
        " ",
        "test",
        ".",
        "",
        "\n",
        "",
        END_OF_TEXT,
        "",
        "\n",
        "",
        "\n",
        "Start",
    ]
    tokens_expected = [
        "This",
        "is",
        "a",
        "test",
        ".",
        "\n",
        END_OF_TEXT,
        "\n",
        "\n",
        "Start",
    ]
    tokens_actual = clean_tokens(tokens_raw, TOKEN_TO_REMOVE)
    assert tokens_actual == tokens_expected


def test_unique_tokens():
    tokens = [
        "This",
        "is",
        "a",
        "test",
        ".",
        "\n",
        END_OF_TEXT,
        "\n",
        "\n",
        "Start",
        END_OF_TEXT,
        "is",
    ]
    tokens_expected = ["\n", ".", "<|endoftext|>", "Start", "This", "a", "is", "test"]
    tokens_actual = get_unique_tokens(tokens)
    assert tokens_actual == tokens_expected


def test_token_dicts():
    tokens = ["\n", ".", END_OF_TEXT, "Start", "This", "a", "is", "test"]
    token_to_int_actual, int_to_token_actual = create_token_to_int_dicts(tokens)
    token_to_int_excpect = {
        "\n": 0,
        ".": 1,
        END_OF_TEXT: 2,
        "Start": 3,
        "This": 4,
        "a": 5,
        "is": 6,
        "test": 7,
    }
    int_to_token_expected = {
        0: "\n",
        1: ".",
        2: END_OF_TEXT,
        3: "Start",
        4: "This",
        5: "a",
        6: "is",
        7: "test",
    }
    assert token_to_int_actual == token_to_int_excpect
    assert int_to_token_actual == int_to_token_expected

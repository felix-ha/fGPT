from data_prep import split_corpus, texts_to_input_ids
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

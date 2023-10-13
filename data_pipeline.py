import time
from tokenizer import (
    split_tokens_raw,
    clean_tokens,
    get_unique_tokens,
    create_token_to_int_dicts,
)


CHARACTER_REPLACEMENTS = {"“": '"', "”": '"', "‘": '"', "’": '"', "-": "–", "—": "–"}
END_OF_TEXT = "<|endoftext|>"
TOKEN_TO_REMOVE = ["", " "]
DELIMTERS = [" ", END_OF_TEXT, "\n", ".", ",", ";", ":", "!", "?", '"']


def print_unique_characters(corpus: str):
    unique_chars = set(corpus.replace(END_OF_TEXT, ""))
    sorted_chars = sorted(unique_chars)
    print(sorted_chars)


def replace_characters(corpus, replacment_dict):
    for char, replacement in replacment_dict.items():
        corpus = corpus.replace(char, replacement)
    return corpus


def pipeline(file_path):
    start_time = time.perf_counter()

    with open(file_path, "r", encoding="utf8") as file:
        corpus_raw = file.read()
        corpus_clean = replace_characters(corpus_raw, CHARACTER_REPLACEMENTS)
        tokens_raw = split_tokens_raw(corpus_clean, DELIMTERS)
        tokens_all = clean_tokens(tokens_raw, TOKEN_TO_REMOVE)
        tokens_unique = get_unique_tokens(tokens_all)
        token_to_int_actual, int_to_token_actual = create_token_to_int_dicts(
            tokens_unique
        )

    print_unique_characters(corpus_raw)
    print(f"Size of vocabulary: {len(token_to_int_actual)}")

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.6f} seconds")


if __name__ == "__main__":
    pipeline(file_path="data.txt")

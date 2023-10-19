import re
import pandas as pd
from constants import TOKENS_NOT_TO_FILTER
import logging


def split_tokens_raw(corpus: str, delimiters: list[str]) -> list[str]:
    pattern = "|".join(map(re.escape, delimiters))
    pattern = f"({pattern})"
    return re.split(pattern, corpus)


def clean_tokens(tokens_raw: list[str], tokens_to_remove: list[str]) -> list[str]:
    return [token for token in tokens_raw if token not in tokens_to_remove]


def string_to_tokens(
    string: str, delimiters: list[str], tokens_to_remove: list[str]
) -> list[str]:
    """
    Combination of split_tokens_raw and clean_tokens.
    Util func for create_encoder
    """
    tokens_raw = split_tokens_raw(string, delimiters)
    tokens = clean_tokens(tokens_raw, tokens_to_remove)
    return tokens


def get_unique_tokens(
    tokens: list[str],
    vocab_size: int,
    tokens_not_to_filter: list[str] = TOKENS_NOT_TO_FILTER,
) -> list[str]:
    """
    Get unique tokens from a list of tokens.
    Only keeps the most frequent tokens up to the vocab_size.
    tokens_not_to_filter will be excluded from the filtering.
    """
    token_all_to_filter = [
        token for token in tokens if token not in tokens_not_to_filter
    ]
    df = pd.DataFrame(token_all_to_filter, columns=["token"])
    df = df.groupby("token").size().reset_index(name="token_count")
    df = df.sort_values(by=["token_count"], ascending=False)
    df_final = df.head(vocab_size)
    tokens_unique = df_final["token"].values.tolist() + tokens_not_to_filter
    logging.info(
        f"Number of unique tokens before filtering: {len(df) + len(tokens_not_to_filter)}"
    )
    logging.info(
        f"Number of unique tokens after filtering: {len(tokens_unique)} (= target_vocab_size + len(TOKENS_NOT_TO_FILTER))"
    )
    return sorted(tokens_unique)


def create_token_to_int_dicts(
    tokens: list[str], unk
) -> tuple[dict[str, int], dict[int, str]]:
    token_to_int = {token: i for i, token in enumerate(tokens)}
    token_to_int[unk] = len(token_to_int)
    int_to_token = {i: token for token, i in token_to_int.items()}
    return token_to_int, int_to_token


def create_encoder(token_to_index, delimiters, tokens_to_remove, unk):
    return lambda string: [
        token_to_index.get(char, token_to_index[unk])
        for char in string_to_tokens(string, delimiters, tokens_to_remove)
    ]


def create_decoder(index_to_token):
    return lambda idxs: "".join([index_to_token[index] + " " for index in idxs])

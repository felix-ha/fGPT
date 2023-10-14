import re


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


def get_unique_tokens(tokens: list[str]) -> list[str]:
    unique_tokens = list(set(tokens))
    return sorted(unique_tokens)


def create_token_to_int_dicts(
    tokens: list[str],
) -> tuple[dict[str, int], dict[int, str]]:
    token_to_int = {token: i for i, token in enumerate(tokens)}
    int_to_token = {i: token for i, token in enumerate(tokens)}
    return token_to_int, int_to_token


def create_encoder(token_to_index, delimiters, tokens_to_remove):
    return lambda string: [
        token_to_index[char]
        for char in string_to_tokens(string, delimiters, tokens_to_remove)
    ]


def create_decoder(index_to_token):
    return lambda idxs: "".join([index_to_token[index] + " " for index in idxs])

import re


def split_tokens_raw(corpus: str, delimiters: list[str]) -> list[str]:
    pattern = "|".join(map(re.escape, delimiters))
    pattern = f"({pattern})"
    return re.split(pattern, corpus)


def clean_tokens(tokens_raw: list[str], tokens_to_remove: list[str]) -> list[str]:
    return [token for token in tokens_raw if token not in tokens_to_remove]


def get_unique_tokens(tokens: list[str]) -> list[str]:
    unique_tokens = list(set(tokens))
    return sorted(unique_tokens)


def create_token_to_int_dicts(
    tokens: list[str],
) -> tuple[dict[str, int], dict[int, str]]:
    token_to_int = {token: i for i, token in enumerate(tokens)}
    int_to_token = {i: token for i, token in enumerate(tokens)}
    return token_to_int, int_to_token

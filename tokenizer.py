import pandas as pd
from constants import TOKENS_NOT_TO_FILTER, END_OF_TEXT
import logging
import re
from tqdm import tqdm
from spacy.lang.en import English


def split_tokens_raw(
    corpus: str, end_of_text_token=END_OF_TEXT, number_splits_for_sub_corpus: int = 10
) -> list[str]:
    logging.debug("start split_tokens_raw")

    nlp = English()
    tokenizer = nlp.tokenizer

    corpus = corpus.replace(end_of_text_token, "")
    len_corpus = len(corpus)
    step = len_corpus // number_splits_for_sub_corpus

    if len_corpus < 1_000_000:
        logging.debug(
            f"{len_corpus=} is smaller than 1_000_000, processing croupus at once"
        )
        return [t.text for t in tokenizer(corpus)]
    else:
        logging.warn(
            f"{len_corpus=} is greater than 1_000_000, processing croupus in steps"
        )
        tokens = []
        for i in tqdm(range(0, len(corpus), step)):
            start = i
            end = i + step
            if end > len_corpus:
                corpus_current = corpus[start:]
            else:
                corpus_current = corpus[start:end]

            logging.debug(f"Split step {start}:{end}")
            tokens_current = [t.text for t in tokenizer(corpus_current)]
            logging.debug(f"append to result")
            tokens.extend(tokens_current)
        logging.debug("end split_tokens_raw")
        return tokens


def clean_tokens(tokens_raw: list[str], tokens_to_remove: list[str]) -> list[str]:
    logging.debug("start clean_tokens")
    result = [token for token in tokens_raw if token not in tokens_to_remove]
    logging.debug("end clean_tokens")
    return result


def string_to_tokens(
    string: str, end_of_text_token: str, tokens_to_remove: list[str]
) -> list[str]:
    """
    Combination of split_tokens_raw and clean_tokens.
    Util func for create_encoder
    """
    tokens_raw = split_tokens_raw(string, end_of_text_token)
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
    logging.info("start get_unique_tokens")
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
    tokens: list[str], unk, end_of_text_token
) -> tuple[dict[str, int], dict[int, str]]:
    token_to_int = {token: i for i, token in enumerate(tokens)}
    token_to_int[unk] = len(token_to_int)
    token_to_int[end_of_text_token] = len(token_to_int)
    int_to_token = {i: token for token, i in token_to_int.items()}
    return token_to_int, int_to_token


def create_encoder(token_to_index, end_of_text_token, tokens_to_remove, unk):
    return lambda string: [
        token_to_index.get(char, token_to_index[unk])
        for char in string_to_tokens(string, end_of_text_token, tokens_to_remove)
    ]


def create_decoder(index_to_token):
    return lambda idxs: "".join([index_to_token[index] + " " for index in idxs])

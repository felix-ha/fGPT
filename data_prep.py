import torch
from constants import PADDING_ID, END_OF_TEXT
import json
from tqdm import tqdm
from pathlib import Path
import logging


def write_to_json(data, path):
    json_string = json.dumps(data)
    with open(path, "w") as f:
        f.write(json_string)


def read_from_json(path):
    with open(path, "r") as f:
        json_string = f.read()
    return json.loads(json_string)


def get_token_int_dicts(path):
    token_to_int = read_from_json(Path(path).joinpath("token_to_int.json"))
    int_to_token = read_from_json(Path(path).joinpath("int_to_token.json"))
    int_to_token = {int(k): v for k, v in int_to_token.items()}
    return token_to_int, int_to_token


def load_file(path, ratio):
    lines = []
    with open(path, "r", encoding="utf8") as file:
        for line in file:
            lines.append(line.strip() + "\n")
    logging.info(f"Loaded {len(lines)} lines from {path}.")
    if ratio < 1.0:
        logging.info(f"Only using {ratio * 100}% of the data.")
        lines = lines[0 : int(len(lines) * ratio)]
    result = "".join(lines)
    return result


def split_corpus(corpus: str, end_of_text_token: str) -> list:
    """
    Split corpus into sentences according to end_of_text_token.
    Trims all leading and trailing whitespace and linebreask from each sentence.
    The end_of_text_token is not removed from each sentence.
    """
    sentences = corpus.split(end_of_text_token)
    sentences = [sentence.strip() + end_of_text_token for sentence in sentences]
    return sentences[:-1]


def texts_to_input_ids(
    texts: list[str], encoder: callable, root_path, save_directory, token_to_int
) -> list[list[int]]:
    """
    Convert a list of texts to a list of lists of input IDs.
    """
    path = Path(root_path) / save_directory
    path.mkdir(parents=True, exist_ok=True)
    N_texts = len(texts)

    for i in tqdm(range(N_texts)):
        file_name = f"{i}.json"
        file_path = path / file_name

        if file_path.exists():
            continue
        else:
            text_tokens = encoder(texts[i]) + [token_to_int[END_OF_TEXT]]
            write_to_json(text_tokens, file_path)


def load_input_ids(root_path, save_directory):
    path = Path(root_path) / save_directory
    result = []
    for file_path in path.iterdir():
        if file_path.is_file():
            ids = read_from_json(file_path)
            result += [ids]
    return result


def input_ids_to_tensor(
    input_ids: list[list[int]], pad: int = PADDING_ID
) -> torch.tensor:
    """
    Convert a list of lists of input IDs to a tensor.
    If the inner lists have different lengths, they will be padded with pad.
    """
    if not isinstance(input_ids[0], list):
        return torch.tensor([input_ids])

    max_len = max([len(input_id) for input_id in input_ids])
    input_ids_padded = [
        input_id + [pad] * (max_len - len(input_id)) for input_id in input_ids
    ]
    return torch.tensor(input_ids_padded)


def collate_fn(batch):
    x, y = zip(*batch)
    return input_ids_to_tensor(x), input_ids_to_tensor(y)

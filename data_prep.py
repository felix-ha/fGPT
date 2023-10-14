import torch 
def split_corpus(corpus: str, end_of_text_token: str) -> list:
    """
    Split corpus into sentences according to end_of_text_token.
    Trims all leading and trailing whitespace and linebreask from each sentence.
    The end_of_text_token is not removed from each sentence.
    """
    sentences = corpus.split(end_of_text_token)
    sentences = [sentence.strip() + end_of_text_token for sentence in sentences]
    return sentences[:-1]


def texts_to_input_ids(texts: list[str], encoder: callable) -> list[list[int]]:
    """
    Convert a list of texts to a list of lists of input IDs.
    """
    input_ids = [encoder(text) for text in texts]
    return input_ids


def input_ids_to_tensor(input_ids: list[list[int]], pad: int = 0) -> torch.tensor:
    """
    Convert a list of lists of input IDs to a tensor.
    If the inner lists have different lengths, they will be padded with pad.
    """
    max_len = max([len(input_id) for input_id in input_ids])
    input_ids_padded = [input_id + [pad] * (max_len - len(input_id)) for input_id in input_ids]
    return torch.tensor(input_ids_padded)

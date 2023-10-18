CHARACTER_REPLACEMENTS = {"“": '"', "”": '"', "‘": '"', "’": '"', "-": "–", "—": "–"}
END_OF_TEXT = "<|endoftext|>"
UNK = "<|unk|>"
TOKEN_TO_REMOVE = ["", " "]
DELIMTERS = [" ", END_OF_TEXT, "\n", ".", ",", ";", ":", "!", "?", '"']
PADDING_ID = 0  # TODO fix global varibale. used in collate_fn

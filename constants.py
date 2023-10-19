CHARACTER_REPLACEMENTS = {"“": '"', "”": '"', "‘": '"', "’": '"', "-": "–", "—": "–"}
END_OF_TEXT = "<|endoftext|>"
UNK = "<|unk|>"
TOKEN_TO_REMOVE = ["", " "]
DELIMTERS = [" ", END_OF_TEXT, "\n", ".", ",", ";", ":", "!", "?", '"']
TOKENS_NOT_TO_FILTER = [
    END_OF_TEXT,
    "\n",
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    '"',
]  # Tokens that should not be filtered out when limiting the vocabulary size
PADDING_ID = 0  # TODO fix global varibale. used in collate_fn

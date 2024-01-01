import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s %(message)s")
file_handler = logging.FileHandler("log.txt")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

import argparse

from data_pipeline import pipeline
from constants import *
from model import simpleGPT


def get_model(vocab_size, n_positions, device):
    return simpleGPT(
        vocab_size=vocab_size,
        n_embd=768,
        num_heads=4,
        block_size=n_positions,
        n_layer=4,
        dropout=0.1,
        device=device,
    )

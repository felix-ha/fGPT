from pathlib import Path
import dask.dataframe as dd
from fgpt.constants import *
from data_prep import read_from_json
from fgpt.data import get_texts_ids
from main import get_model

from dionysus.training import TrainingConfig, train
import torch
from torch.utils.data import DataLoader
from data_prep import collate_fn
from model import (
    LanguageModelDataset,
    simpleGPT,
    cross_entropy_language_model
)


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


data_path = Path('datapipeline')
data_info_file = data_path.joinpath("dataset_info.json")
dataset_train_file = data_path.joinpath("dataset_train.parquet")
dataset_valid_file = data_path.joinpath("dataset_valid.parquet")

texts_ids_train = get_texts_ids(dataset_train_file)
texts_ids_validation = get_texts_ids(dataset_valid_file)
dataset_info = read_from_json(data_info_file)

vocab_size = dataset_info["vocab_size"]
n_positions = dataset_info["n_positions"]

dataset_train = LanguageModelDataset(texts_ids_train)
dataset_validation = LanguageModelDataset(texts_ids_validation)

dataloader_train = DataLoader(
    dataset_train, batch_size=8, shuffle=True, collate_fn=collate_fn
)
dataloader_validation = DataLoader(
    dataset_validation, batch_size=8, shuffle=False, collate_fn=collate_fn
)

device = "gpu" if torch.cuda.is_available() else "cpu"
model = get_model(vocab_size, n_positions, device)
loss_func = cross_entropy_language_model

torch.manual_seed(0)
checkpoint_path = Path('/notebooks/fGPT/runs/fGPT/last/model.pt')
train_config = TrainingConfig(
    model=model,
    epochs=2,
    loss_func=loss_func,
    training_loader=dataloader_train,
    validation_loader=dataloader_validation,
    optimizer="AdamW",
    device=device,
    force_write_logs=True,
    save_model=True,
    tar_result=True,
    save_path="runs",
    model_name="fGPT",
    progress_bar=True,
    checkpoint_step=1,
    checkpoint_step_batch=1_000,
    checkpoint_path = checkpoint_path if checkpoint_path.is_file() else None
)

train(train_config)

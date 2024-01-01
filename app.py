from pathlib import Path
import streamlit as st
import gdown
from fgpt.constants import *
from data_prep import read_from_json
from model import generate
import torch
from main import get_model
import logging
import time
from transformers import AutoTokenizer


@st.cache_resource
def load_model(model_dict_file, vocab_size, n_positions):
    model = get_model(vocab_size, n_positions, device="cpu")

    training_result_dict = torch.load(
        model_dict_file, map_location=torch.device("cpu")
    )
    model_state_dict = training_result_dict["model_state_dict"]
    model.load_state_dict(model_state_dict)

    return model


folder_downloads = Path("model")

if not folder_downloads.exists():
    folder_downloads.mkdir(parents=True, exist_ok=True)
    id = "12tKDt3vEHz4uKqiDWOBM2XaeAhaTOoNt"
    gdown.download_folder(id=id, quiet=False, use_cookies=False)

model_dict_file = folder_downloads.joinpath('model.pt')
dataset_info_path = folder_downloads.joinpath("dataset_info.json")
tokenizer_path = folder_downloads.joinpath("tokenizer")

dataset_info = read_from_json(dataset_info_path)
vocab_size = dataset_info["vocab_size"]
n_positions = dataset_info["n_positions"]    

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model = load_model(model_dict_file, vocab_size, n_positions)

st.write(
    """
# fGPT 

A language model trained from scratch on [tiny stories](https://arxiv.org/abs/2305.07759)


"""
)

settings = st.checkbox('Settings')

sample = False
max_n=300
top_k = None
top_p = None
n_beams = None
temperature = 1.0
strategy = "gready"

if settings:
    strategy = st.radio(
    "sampling strategy",
    ["gready", "top-k", "top-p" ],
            captions = ["gready sampling",
                         "top-k sampling", 
                         "top-p sampling"]
                         )
    
    max_n = st.slider('max token to generate', min_value=1, max_value=1000, value=300)
    temperature = st.slider('temperature', min_value=0.0, max_value=5.0, value=1.0)

    if strategy == 'gready':
        sample = st.checkbox('sample')
    elif strategy == 'top-k':
        top_k = st.slider('top-k', min_value=1, max_value=20, value=5)
    elif strategy == 'top-p':
        top_p = st.slider('top-p', min_value=0.01, max_value=1.0, value=0.8)


prompt = st.text_input("Enter the beginning of a story...")

if st.button("Generate"):

    start = time.perf_counter()

    output, _ = generate(
        model,
        tokenizer,
        prompt,
        max_n=max_n,
        choices_per_step=3,
        sample=sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )

    end = time.perf_counter()

    st.text_area("continued story by model", output, height=350)

    st.write(f"Time to generate: {end - start:0.1f} seconds")
    logging.info(f"""
    promt {prompt}
    output {output}
    """)

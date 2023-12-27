import os
import streamlit as st
import gdown
from constants import *
from data_prep import read_from_json
from dask_pipeline import load_vocabulary
from tokenizer import create_encoder, create_decoder
from model import generate
import torch
from main import get_model
import logging
import time


@st.cache_resource
def load_model(vocab_size, n_positions):
    model = get_model(vocab_size, n_positions, device="cpu")

    training_result_dict = torch.load(
        os.path.join(folder_downloads, "model.pt"), map_location=torch.device("cpu")
    )
    model_state_dict = training_result_dict["model_state_dict"]
    model.load_state_dict(model_state_dict)

    return model


folder_downloads = "downloads"

if not os.path.exists(folder_downloads):
    os.makedirs(folder_downloads)


urls = [
    os.getenv("URL_DATASET_INFO"),
    os.getenv("URL_TOKEN_TO_INT"),
    os.getenv("URL_MODEL"),
]

outputs = ["dataset_info.json", "token_to_int.json", "model.pt"]
for url, output in zip(urls, outputs):
    file = os.path.join(folder_downloads, output)
    if not os.path.isfile(file):
        gdown.download(url, file, quiet=False)


token_to_int, int_to_token = load_vocabulary(os.path.join(folder_downloads, "token_to_int.json"))
dataset_info = read_from_json(os.path.join(folder_downloads, "dataset_info.json"))
vocab_size = dataset_info["vocab_size"]
n_positions = dataset_info["n_positions"]

encoder = create_encoder(token_to_int, END_OF_TEXT, TOKEN_TO_REMOVE, UNK)
decoder = create_decoder(int_to_token)

stop_token_id = token_to_int[END_OF_TEXT]

model = load_model(vocab_size, n_positions)

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
    ["gready", "top-k", "top-p", "beam search"],
            captions = ["gready sampling",
                         "top-k sampling", 
                         "top-p sampling",
                         "beam search - warning: experimental feature"])
    
    max_n = st.slider('max token to generate', min_value=1, max_value=1000, value=300)
    temperature = st.slider('temperature', min_value=0.0, max_value=5.0, value=1.0)

    if strategy == 'gready':
        sample = st.checkbox('sample')
    elif strategy == 'top-k':
        top_k = st.slider('top-k', min_value=1, max_value=20, value=5)
    elif strategy == 'top-p':
        top_p = st.slider('top-p', min_value=0.01, max_value=1.0, value=0.8)
    elif strategy == 'beam search':
        n_beams = st.slider('number of beams', min_value=1, max_value=10, value=2)

    

prompt = st.text_input("Enter the beginning of a story...")

if st.button("Generate"):

    # start mesuaring time with perf_counter
    start = time.perf_counter()

    output, _ = generate(
        model,
        prompt,
        encoder,
        decoder,
        stop_token_id=stop_token_id,
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

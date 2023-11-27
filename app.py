import os
import streamlit as st
import gdown
import gc

from constants import *
from data_prep import read_from_json, get_token_int_dicts
from tokenizer import create_encoder, create_decoder
from model import simpleGPT, generate
import torch
from main import get_model


folder_downloads = 'downloads'

if not os.path.exists(folder_downloads):
    os.makedirs(folder_downloads)
	
urls = ['https://drive.google.com/uc?id=1USRoXjarH5-07AF50RXMm4FpCj9Qg20M',
	'https://drive.google.com/uc?id=1qNRjnN4W94rRxn5TDjcXbC-F0uyR8VjW',
	'https://drive.google.com/uc?id=1mkK1ME-5hfGRdkYS9jUBaz21Y1AMcRYv',
	'https://drive.google.com/uc?id=1KXCfgk6LHtgt934FwsizejQyWTKIMaId' ]

outputs = ["dataset_info.json", "token_to_int.json", "int_to_token.json", "model.pt"]
for url, output in zip(urls, outputs):
	file = os.path.join(folder_downloads, output)
	if not os.path.isfile(file):
		gdown.download(url, file, quiet=False)
	
token_to_int, int_to_token = get_token_int_dicts(folder_downloads)
dataset_info = read_from_json(os.path.join(folder_downloads, "dataset_info.json"))
vocab_size = dataset_info["vocab_size"]
n_positions = dataset_info["n_positions"]

encoder = create_encoder(token_to_int, END_OF_TEXT, TOKEN_TO_REMOVE, UNK)
decoder = create_decoder(int_to_token)

stop_token_id = token_to_int[END_OF_TEXT]

model = get_model(vocab_size, n_positions, device='cpu')

training_result_dict = torch.load(os.path.join(folder_downloads, "model.pt"), map_location=torch.device('cpu'))
model_state_dict = training_result_dict["model_state_dict"]
model.load_state_dict(model_state_dict)

st.write(
    """
# fGPT 

A language model trained from scratch on [tiny stories](https://arxiv.org/abs/2305.07759)


"""
)

prompt = st.text_input('Enter the beginning of a story...')

if st.button('Generate'):
    output, _ = generate(
            model,
            prompt,
            encoder,
            decoder,
            stop_token_id=stop_token_id,
            max_n=100,
            choices_per_step=3,
        )
    
    del model
    gc.collect()

    st.text_area("continued story by model", output)

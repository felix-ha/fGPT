from main import get_model

import os
from pathlib import Path
import torch
import pandas as pd

from constants import *
from data_prep import read_from_json, get_token_int_dicts
from tokenizer import create_encoder, create_decoder
from model import generate
from main import get_model
from dask_pipeline import load_vocabulary


def load_model(model_dict_file, vocab_size, n_positions):  
    model = get_model(vocab_size, n_positions, device="cpu")

    training_result_dict = torch.load(
        model_dict_file,
        map_location=torch.device("cpu"),
    )
    model_state_dict = training_result_dict["model_state_dict"]
    model.load_state_dict(model_state_dict)
    return model


model_dict_file = Path('/notebooks/fGPT/runs/fGPT/last/model.pt')
vocabulary_file = Path('datapipeline').joinpath('token_to_int.json')
dataset_info_path = Path('datapipeline').joinpath("dataset_info.json")

token_to_int, int_to_token = load_vocabulary(vocabulary_file)
encoder = create_encoder(token_to_int, END_OF_TEXT, TOKEN_TO_REMOVE, UNK)
decoder = create_decoder(int_to_token)
stop_token_id = token_to_int[END_OF_TEXT]

dataset_info = read_from_json(dataset_info_path)
vocab_size = dataset_info["vocab_size"]
n_positions = dataset_info["n_positions"]    

model = load_model(model_dict_file, vocab_size, n_positions)


prompts = [
    "Alice was so tired when she got back home so she went",
    'Jack and Lily saw a rainbow after a rainy day. They were amazed by the colors. Jack said, "Look, Lily. A rainbow has',
    "Jack and Lily liked to watch the moon at night. They noticed that the moon changed its shape every night. Sometimes the moon was big and round, and sometimes it was",
    "Jack wanted to read a book, so he went to",
    '"Can cows fly?", Alice asked her mother.',
    '"What do birds like to eat?", Tom asked his mother.',
    '"What language do they speak in France?", Tom asked his mother',
    "If I throw a ball up in the air, eventually it will",
    'It was winter and cold outside so his mother told him, "You should',
    "Lily likes cats and dogs. She asked her mom for a dog and her mom said no, so instead she asked",
    'Jack told Mary, "If you give me your banana, I’ll give you my apple". Mary gave Jack her Banana so',
    "On weekends Jack went to visit his grandmother whereas on weekdays he would go to school. Last weekend, when Jack was on his way to",
    "Lily and Ben were having an argument. Ben said that cake is much better than ice cream and Lily said that",
    'Lily and Ben are having an argument. They are trying to decide between the park and the swimming pool. Ben says, "I want to go to the park". Lily says',
    "Jack's mother was not home, and his father was at home. When Jack came home, he said hello to",
    "Lily doesn't like swimming. When her father wants to take her to the swimming pool, she says",
    "Both Ben and Lily wanted cake. Father said that there was only one piece of cake left. They",
    "Ben went to visit Lily in her house, but she was not at home. Ben knocked on the door,",
    '"Hi Jane, have you seen Alice? I cannot find her anywhere", said Jack.',
    'Max had two dogs. One was white and the other was black. Max walked up the street and saw a kid with a dog. He told the kid, "I see you have a Brown dog. I also have',
    'Anne had a piece of candy in her left pocket and a piece of chocolate in her right pocket. Annes mom asked her, "Anne, what is that you have in your left pocket?"',
    "Alice had both an apple and a carrot in her bag. She took the apple out of the bag and gave it to Jack. She reached into the bag again and took",
    'Alice and Jack walked up the street and met a girl in a red dress. The girl said to them, "Hi, I am Jane. What are your names?"',
    'Diva was hungry, and wanted to bake a cake, but she did not have any sugar at home, so she decided to go ask around. She started walking and met a squirrel. She asked the squirrel, "Would you happen',
]

responses = []

with torch.no_grad():
    model.eval()
    for prompt in prompts:
        output, _ = generate(
            model,
            prompt,
            encoder,
            decoder,
            stop_token_id=stop_token_id,
            max_n=500,
            choices_per_step=1,
            sample=True,
            temperature=0.5,
        )
        
        responses.append(output)
        

        print(prompt)
        print(output)
        print('----------------------------')


result = pd.DataFrame({'prompt': prompts, 'response': responses})
result.to_csv('evaluation.csv', sep=";")

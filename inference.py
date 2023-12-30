from main import get_model

import os
from pathlib import Path
import torch
import pandas as pd
import math
import numpy as np

from constants import *
from data_prep import read_from_json 
from model import generate
from main import get_model
from pydantic import BaseModel
from typing import List
import functools 
from transformers import AutoTokenizer
from tokenizers.processors import TemplateProcessing


class Beam(BaseModel):
    ids:  List[int]
    probabilities: List[float]
    
    def __len__(self):
        return len(self.ids)
    
    def product_proba(self):
        return functools.reduce(lambda a, b: a * b, self.probabilities)
    
    def sum_log_prob(self, alpha):
        if len(self.probabilities) == 0:
            raise ValueError('No probablities available')
        return  (1 / self.__len__() ** alpha) *sum(map(math.log, self.probabilities))


def generate_beam(
    model,
    prompt,
    encoder,
    decoder,
    stop_token_id,
    max_n,
    beam_size=2,
    temperature=1.0,
):

    x_input = torch.tensor([encoder(prompt)])
    response_idx = x_input.shape[1]

    model.eval()
    with torch.no_grad():
        for i in range(max_n):
            if i == 0:
                beams = []
                y_output = model(x_input)
                logits_last = y_output[:, -1, :]
                logits_last /= temperature
                probabilities_next_token = torch.softmax(logits_last, dim=-1).squeeze()
                sorted_token_ids = torch.argsort(
                    probabilities_next_token, dim=-1, descending=True
                )

                for choice_idx in range(beam_size):
                    token_id = sorted_token_ids[choice_idx].item()
                    token_prob = probabilities_next_token[token_id].cpu().numpy().item()
                    beams.append(Beam(ids=[token_id], probabilities=[token_prob]))
                    
            else:
                beams_to_predict = beams[-beam_size:].copy()
                n_vocab = logits_last.shape[1]
        
                cond_beam_probabilities = torch.empty(n_vocab * beam_size)
                beam_probabilities = []
                for idx in range(beam_size):
                    beam_current = beams_to_predict[idx]
                    tensor_to_append = torch.tensor(beam_current.ids).view(1, -1)
                    x_current = torch.cat((x_input, tensor_to_append), dim=-1)
                    y_output = model(x_current)
                    logits_last = y_output[:, -1, :]
                    logits_last /= temperature
                    probabilities_next_token = torch.softmax(logits_last, dim=-1).squeeze() 
                    beam_probabilities.append(probabilities_next_token)
                    cond_beam_probabilities[(idx * n_vocab):((idx+1) * n_vocab)] = probabilities_next_token * beam_current.product_proba()               
                    
                sorted_token_ids = torch.argsort(
                    cond_beam_probabilities, dim=-1, descending=True
                )
                
                beam_ids = sorted_token_ids[0:beam_size] // n_vocab
                token_ids_for_next_beam = sorted_token_ids % n_vocab
                
                for token_id, beam_id in zip(token_ids_for_next_beam.tolist(), beam_ids.tolist()):
                    beam_current = beams_to_predict[beam_id]
                
                    token_prob = beam_probabilities[beam_id][token_id].cpu().numpy().item()
                    new_ids = beam_current.ids.copy()
                    new_ids.append(token_id)
                    new_probabilities = beam_current.probabilities.copy()
                    new_probabilities.append(token_prob)

                    beams.append(Beam(ids=new_ids,
                                      probabilities=new_probabilities
                                     )
                                )

        len_final_candidate = max(map(lambda beam: len(beam), beams))
        alpha = 3
        
        final_candidates = beams.copy()

        sums = list(map(lambda beam: beam.sum_log_prob(alpha), final_candidates))
        id_choosen = np.argmax(sums)
        choosen_beam = final_candidates[id_choosen]
        result =  decoder(choosen_beam.ids)
        
        return result



def load_model(model_dict_file, vocab_size, n_positions):  
    model = get_model(vocab_size, n_positions, device="cpu")

    training_result_dict = torch.load(
        model_dict_file,
        map_location=torch.device("cpu"),
    )
    model_state_dict = training_result_dict["model_state_dict"]
    model.load_state_dict(model_state_dict)
    return model


if __name__ == "__main__":
    model_dict_file = Path('runs/fGPT/last/model.pt')
    data_path = Path('datapipeline')
    dataset_info_path = data_path.joinpath("dataset_info.json")
    tokenizer_path = data_path.joinpath("tokenizer")

    dataset_info = read_from_json(dataset_info_path)
    vocab_size = dataset_info["vocab_size"]
    n_positions = dataset_info["n_positions"]    

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
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
                tokenizer,
                prompt,
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

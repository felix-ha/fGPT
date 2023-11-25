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
import wget
import os
from torch.utils.data import DataLoader
from data_prep import collate_fn
from tokenizer import create_encoder, create_decoder, split_tokens_raw
from data_prep import read_from_json, get_token_int_dicts

from dionysus.training import TrainingConfig, train
from model import (
    LanguageModelDataset,
    simpleGPT,
    cross_entropy_language_model,
    generate,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        default=False,
        action="store_true",
        help="Use full TinyStores dataset instead of the small one.",
    )
    parser.add_argument(
        "--ratio",
        default=1.0,
        type=float,
        help="Ratio of the data to use for training.",
    )
    parser.add_argument(
        "--splits",
        default=2,
        type=int,
        help="Ratio of the data to use for training.",
    )

    args = parser.parse_args()

    if args.full:
        logging.info("Using full TinyStores dataset.")
        path_train = "data/TinyStoriesV2-GPT4-train.txt"
        path_validation = "data/TinyStoriesV2-GPT4-valid.txt"
        if not os.path.exists(path_train):
            logging.info("Downloading full TinyStores training dataset.")
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
            wget.download(url, path_train)
        if not os.path.exists(path_validation):
            logging.info("Downloading full TinyStores validation dataset.")
            url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
            wget.download(url, path_validation)
    else:
        logging.info("Using small dev dataset.")
        path_train = "data/data_train.txt"
        path_validation = "data/data_validation.txt"

    path = "datapipeline"
    pipeline(path_train, path_validation, path, args.ratio, args.splits)

    device = "cpu"

    token_to_int, int_to_token = get_token_int_dicts(path)
    texts_ids_train = read_from_json(os.path.join(path, "texts_ids_train.json"))
    texts_ids_validation = read_from_json(
        os.path.join(path, "texts_ids_validation.json")
    )

    vocab_size = len(int_to_token)
    n_positions = max([len(text_ids) for text_ids in texts_ids_train])

    encoder = create_encoder(token_to_int, END_OF_TEXT, TOKEN_TO_REMOVE, UNK)
    decoder = create_decoder(int_to_token)

    dataset_train = LanguageModelDataset(texts_ids_train)
    dataset_validation = LanguageModelDataset(texts_ids_validation)

    dataloader_train = DataLoader(
        dataset_train, batch_size=4, shuffle=True, collate_fn=collate_fn
    )
    dataloader_validation = DataLoader(
        dataset_validation, batch_size=4, shuffle=False, collate_fn=collate_fn
    )

    stop_token_id = token_to_int[END_OF_TEXT]

    model = simpleGPT(
        vocab_size=vocab_size,
        n_embd=768,
        num_heads=4,
        block_size=n_positions,
        n_layer=4,
        dropout=0.1,
        device=device,
    )

    loss_func = cross_entropy_language_model

    train_config = TrainingConfig(
        model=model,
        epochs=5,
        loss_func=loss_func,
        training_loader=dataloader_train,
        validation_loader=dataloader_validation,
        optimizer="AdamW",
        device=device,
        force_write_logs=False,
        save_model=True,
        tar_result=True,
        save_path="runs",
        model_name="GPT-2",
        progress_bar=True,
    )

    train(train_config)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of parameters: {num_params}")

prompts = [
    "Alice was so tired when she got back home so she went",
    'Jack and Lily saw a rainbow after a rainy day. They were amazed by the colors. Jack said, "Look, Lily. A rainbow has',
    "Jack and Lily liked to watch the moon at night. They noticed that the moon changed its shape every night. Sometimes the moon was big and round, and sometimes it was",
    "Jack wanted to read a book, so he went to",
    '"Can cows fly?", Alice asked her mother.',
    '"What do birds like to eat?", Tom asked his mother.',
    '"What language do they speak in France?", Tom asked his mother'
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

prompts = ['Jack was', 'Jack was not ']

for prompt in prompts:
    output, choices = generate(
        model,
        prompt,
        encoder,
        decoder,
        stop_token_id=stop_token_id,
        max_n=5,
        choices_per_step=3,
    )

    logging.info(f"\n{choices}")
    logging.info(f"Promt: {prompt}")
    logging.info(f"Model: {output}")

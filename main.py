import logging
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

from dionysus.training import TrainingConfig, train
from model import simpleGPT, cross_entropy_language_model, generate
from data_pipeline import pipeline
from constants import *

if __name__ == "__main__":
    data = pipeline("data/data_train.txt", "data/data_validation.txt")

    stop_token_id = data.token_to_int[END_OF_TEXT]

    model = simpleGPT(
        data.vocab_size,
        n_embd=8,
        num_heads=4,
        block_size=data.n_positions,
        n_layer=1,
        dropout=0.1,
        device="cpu",
    )

    loss_func = cross_entropy_language_model

    train_config = TrainingConfig(
        model=model,
        epochs=2,
        loss_func=loss_func,
        training_loader=data.dataloader_train,
        validation_loader=data.dataloader_validation,
        optimizer="AdamW",
        device="gpu",
        colab=True, 
        save_model=True,
        tar_result=True,
        save_path="runs",
        model_name="GPT-2",
        progress_bar=True,
    )

    train(train_config)

    prompt = "Tom was"
    output, choices = generate(
        model,
        prompt,
        data.encoder,
        data.decoder,
        stop_token_id=stop_token_id,
        max_n=5,
        choices_per_step=3,
    )

    logging.info(f"\n{choices}")
    logging.info(f"Promt: {prompt}")
    logging.info(f"Model: {output}")

import torch
from transformers import GPT2Config, AutoTokenizer, GPT2Model, GPT2LMHeadModel
from dionysus.utils import compute_size


def print_model_stats(model):
    print(
        f"{model.__class__.__name__} has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters"
    )
    print(f"and is {compute_size(model)} mb large")


tokenizer = AutoTokenizer.from_pretrained("gpt2")
model_bare = GPT2Model(GPT2Config())
model = GPT2LMHeadModel(GPT2Config())

print_model_stats(model_bare)
print_model_stats(model)

batch_size = 1
context = 4
x = torch.zeros((batch_size, context), dtype=torch.long)

inputs = tokenizer("Hello my name is", return_tensors="pt")
assert x.shape == inputs.input_ids.shape

y_features = model_bare(x)
assert y_features.last_hidden_state.shape == (
    batch_size,
    context,
    model_bare.config.n_embd,
)

y_logits = model(x)
assert y_logits.logits.shape == (batch_size, context, model.config.vocab_size)

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset
from constants import END_OF_TEXT


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        T = x.shape[-2]
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class FeedFoward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size, dropout) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, block_size, dropout=0.0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedFoward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# TODO add model config as for huggingface GT2Config


class simpleGPT(nn.Module):
    def __init__(
        self, vocab_size, n_embd, num_heads, block_size, n_layer, dropout, device
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head=num_heads, block_size=block_size, dropout=dropout)
                for _ in range(n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        if self.device != "cpu":
            device = "cuda"
        else:
            device = "cpu"

        positinal_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + positinal_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        return logits


def cross_entropy_language_model(logits, targets):
    """
    Removes the time dimension for logits and targets and computes the cross entropy loss
    For the F.cross_entropy function, the inputs are predicted unnormalized logits and output are ground truth class indices or class probabilities
    """
    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = targets.view(B * T)
    loss = F.cross_entropy(logits, targets)
    return loss


def generate(
    model,
    tokenizer,
    prompt,
    max_n,
    choices_per_step,
    sample=False,
    temperature=1.0,
    top_k=None,
    top_p=None
):
    if top_k and top_p:
        raise ValueError("top_k and top_p cannot be used together")
    
    stop_token_id = tokenizer.convert_tokens_to_ids(END_OF_TEXT)
    response_ids = []
    x_input = tokenizer.encode(prompt, return_tensors="pt")  
    response_idx = x_input.shape[1]
    iterations = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_n):
            iteration = dict()
            iteration["Input"] = tokenizer.decode(x_input.squeeze().tolist())
            y_output = model(x_input)
            logits_last = y_output[:, -1, :]
            if temperature > 0:
                logits_last /= temperature
            probabilities_next_token = torch.softmax(logits_last, dim=-1).squeeze()

            sorted_token_ids = torch.argsort(
                probabilities_next_token, dim=-1, descending=True
            )
            for choice_idx in range(choices_per_step):
                token_id = sorted_token_ids[choice_idx].item()
                token_prob = probabilities_next_token[token_id].cpu().numpy()
                token_choice = f"{tokenizer.decode([token_id])} ({100 * token_prob:.2f}%)"
                iteration[f"Choice {choice_idx+1}"] = token_choice

            if top_k:
                probabilities_next_token[probabilities_next_token < torch.topk(probabilities_next_token, top_k)[0][..., -1, None]] = 0
                probabilities_next_token /= probabilities_next_token.sum()
            elif top_p:
                sorted_probs, sorted_indices = torch.sort(probabilities_next_token, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                sorted_indices_to_remove = sorted_indices_to_remove.to(torch.bool)
                sorted_probs[sorted_indices_to_remove] = 0
                probabilities_next_token /= probabilities_next_token.sum()

            if sample or top_k or top_p:
                token_id = torch.multinomial(probabilities_next_token, 1)
            else:
                token_id = torch.argmax(probabilities_next_token)

            x_input = torch.cat((x_input, token_id.reshape(1, -1)), dim=1)
            iterations.append(iteration)
            if token_id == stop_token_id:
                break

    response_ids = x_input[:, response_idx:]
    result = tokenizer.decode(response_ids.squeeze().tolist()).replace(END_OF_TEXT, "")
    if token_id != stop_token_id:
        result = result + f"(reached maximium tokens to generate: {max_n})"
    return result, pd.DataFrame(iterations)


class LanguageModelDataset(Dataset):
    def __init__(self, input_ids: list[list[int]]):
        super(LanguageModelDataset, self).__init__()
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    # slicing is not implemented
    def __getitem__(self, idx):
        return self.input_ids[idx][:-1], self.input_ids[idx][1:]

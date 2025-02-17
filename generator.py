import torch
import torch.nn as nn
from datasets import load_dataset
import os
import torch.nn.functional as F
from tqdm import tqdm
from data_utils import CharTokenizer, TinyStoriesDataset
from torch.utils.data import Dataset, DataLoader
from  transformer import Transformer
from trainer import save_model, load_model

# File path for saving the model
MODEL_PATH = "tiny_transformer"

# Network Parameters
num_epochs = 60
batch_size = 128
learning_rate = 3e-4
dropout_rate = 0.2
context_size = 512
embedding_dim = 384
num_heads = 6
num_blocks = 6

# Get vocabulary size
tokenizer = CharTokenizer.read_from_file("vocabulary.txt")
vocabulary = tokenizer.vocabulary
vocab_size = len(vocabulary)

# Function to generate text from the model
def generate(model, start_text, tokenizer, device):
    model.eval()  # Set model to evaluation mode
    print(start_text, end="")

    # Convert input text to token indices
    tokens = [tokenizer.start_token] + tokenizer.encode(start_text)
    chars = torch.tensor(tokens).to(device)
    chars = chars.view(1, -1)  # Reshape to (batch_size=1, sequence_length)

    while True:
        output = model(chars)  # Forward pass
        prob = torch.nn.functional.softmax(output[0, -1], dim=0)  # Get last token prediction
        idx = torch.multinomial(prob, num_samples=1)  # Sample from distribution
        new_token = idx.item()
        if new_token == tokenizer.end_token:
            break
        char = tokenizer.decode([new_token])
        print(char, end="", flush=True)  # Print generated character

        # Append new token and keep context_size limit
        chars = torch.cat([chars, idx.view(1, 1)], dim=1)
        chars = chars[:, -context_size:]  # Keep only last `context_size` tokens

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

with torch.no_grad():
    device = torch.device("mps")
    tokenizer = CharTokenizer.read_from_file("vocabulary.txt")
    vocab_size = len(tokenizer.vocabulary)
    model = Transformer(vocab_size, context_size, embedding_dim, num_blocks, num_heads, dropout_rate, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model, optimizer, start_epoch, _ = load_model(model, optimizer)
    model = model.to(device)

    os.system('clear')

    while True:
        input_text = input("Enter starting text: ")
        generate(model, input_text, tokenizer, device)
        print("\n\n")


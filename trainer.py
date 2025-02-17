import torch
import torch.nn as nn
from datasets import load_dataset
import os
import torch.nn.functional as F
from tqdm import tqdm
from data_utils import CharTokenizer, TinyStoriesDataset, TinyStoriesDatasetRandomisedChunks
from torch.utils.data import Dataset, DataLoader
from transformer import Transformer

# File path for saving the model
MODEL_PATH = "tiny_transformer"

#Config
device = torch.device("mps")
torch.set_default_dtype(torch.bfloat16)
#
# Network Parameters
num_epochs = 40
batch_size = 128
learning_rate = 3e-4
dropout_rate = 0.2
context_size = 512
embedding_dim = 384
num_heads = 6
num_blocks = 6

# Function to save the model
def save_model(model, optimizer, epoch, loss):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    torch.save(checkpoint, MODEL_PATH + f" - {epoch}" + ".pth")
    torch.save(checkpoint, MODEL_PATH)
    print(f"Model saved at epoch {epoch} with loss {loss:.4f}")

# Function to load the model
def load_model(model, optimizer):
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Model loaded from epoch {start_epoch} with loss {loss:.4f}")
        return model, optimizer, start_epoch, loss
    else:
        print("No saved model found. Training from scratch.")
        return model, optimizer, 0, None

# Function to train the model
def train(model, train_loader,tokenizer, learning_rate):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Load model if saved
    model, optimizer, start_epoch, _ = load_model(model, optimizer)

    if start_epoch < num_epochs:
        print("Training started")
        for epoch in range(start_epoch, num_epochs + 1):  # Continue training from last saved epoch
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", unit="batch")  # tqdm for batch progress

            for batch_idx, (x, y) in enumerate(progress_bar):
                x, y = x.to(device), y.to(device)

                output = model(x)
                loss = F.cross_entropy(output.view(-1, tokenizer.vocab_size), y.view(-1))

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

                # Update tqdm progress bar with batch loss
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(train_loader)

            # Print epoch loss
            print(f"Epoch {epoch}: Loss = {avg_loss:.4f}")

            # Save model every 1000 epochs
            save_model(model, optimizer, epoch, avg_loss)

# Function to validate the model
def validate(model, val_loader, tokenizer):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = F.cross_entropy(output.view(-1, tokenizer.vocab_size), y.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    model.train()  # Set back to train mode after validation

# Function to generate text from the model
def generate(model, start_text, num_chars):
    model.eval()  # Set model to evaluation mode
    print(start_text, end="")

    # Convert input text to token indices
    tokens = [tokenizer.start_token] + tokenizer.encode(start_text)
    chars = torch.tensor(tokens).to(device)
    chars = chars.view(1, -1)  # Reshape to (batch_size=1, sequence_length)

    for _ in range(num_chars):
        output = model(chars)  # Forward pass
        prob = torch.nn.functional.softmax(output[0, -1], dim=0)  # Get last token prediction
        idx = torch.multinomial(prob, num_samples=1)  # Sample from distribution
        new_token = idx.item()
        if new_token == tokenizer.end_token:
            break
        char = tokenizer.decode([new_token])
        print(char, end="")  # Print generated character

        # Append new token and keep context_size limit
        chars = torch.cat([chars, idx.view(1, 1)], dim=1)
        chars = chars[:, -context_size:]  # Keep only last `context_size` tokens

def main():
    # Load dataset
    dataset = load_dataset("roneneldan/TinyStories")
    
    # Create tokenizer from training split
    all_text = "".join(sample["text"] for sample in dataset["train"])
    tokenizer = CharTokenizer.from_data(all_text)
    tokenizer.save_to_file("vocabulary.txt")
    vocabulary = tokenizer.vocabulary
    vocab_size = len(vocabulary)

    # Create Train & Validation datasets 
    train_dataset = TinyStoriesDatasetRandomisedChunks(dataset["train"], tokenizer, context_size)
    val_dataset = TinyStoriesDatasetRandomisedChunks(dataset["validation"], tokenizer, context_size)
    # Create DataLoader for Train & Validation datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Transformer(vocab_size, context_size, embedding_dim, num_blocks, num_heads, dropout_rate, device)
    model = model.to(device)
    model.train()
    train(model, train_loader, tokenizer, learning_rate)
    with torch.no_grad():
        validate(model, val_loader, tokenizer)

if __name__ == "__main__":
    main()

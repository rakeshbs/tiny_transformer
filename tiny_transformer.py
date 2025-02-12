import torch
import torch.nn as nn
from datasets import load_dataset
import os
import torch.nn.functional as F
from tqdm import tqdm
from data_utils import CharTokenizer, TinyStoriesDataset
from torch.utils.data import Dataset, DataLoader

# File path for saving the model
MODEL_PATH = "tiny_transformer"

#Config
device = torch.device("mps")
torch.set_default_dtype(torch.bfloat16)
#
# Network Parameters
num_epochs = 20
batch_size = 64
learning_rate = 3e-4
dropout_rate = 0.2
context_size = 512
embedding_dim = 384
num_heads = 6
num_blocks = 6

# Load dataset
dataset = load_dataset("roneneldan/TinyStories")
# Create tokenizer from training split
tokenizer = CharTokenizer(dataset["train"])
# Create Train & Validation datasets 
train_dataset = TinyStoriesDataset(dataset["train"], tokenizer, context_size)
val_dataset = TinyStoriesDataset(dataset["validation"], tokenizer, context_size)
# Create DataLoader for Train & Validation datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Get vocabulary size
vocabulary = tokenizer.vocabulary
vocab_size = len(vocabulary)

# Single Attention Head to process the context of input data
class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_size)
        self.key = nn.Linear(embedding_dim, head_size)
        self.value = nn.Linear(embedding_dim, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size).to(device)))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        _, T, E = x.shape
        # This is the attention mechanism.
        # Each token produces a query, key, and value vector.
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # The query vector is multiplied with the key vector to get the attention weights.
        attention = torch.matmul(q, k.transpose(-2, -1)) * E ** -0.5
        # The attention weights are masked to prevent the model from looking into the future.
        # A lower triangular matrix is used to mask the attention weights.
        attention = attention.masked_fill(self.tril[:T, :T]== 0, float("-inf"))
        # softmax is applied to the attention weights to get the final attention weights.
        attention = torch.nn.functional.softmax(attention, dim=-1)
        # Dropout is applied to the attention weights
        attention = self.dropout(attention)
        # The attention weights are multiplied with the value vector to get the final output.
        output = torch.matmul(attention, v)
        return output

# Multi Attention Head to process the context of input data. 
# Each head processes the context differently and the outputs are concatenated to get the final output.
# Each head outputs a vector of size embedding_dim // num_heads
class MultiAttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()        

        # Multiple Self Attention Heads
        self.attention_heads = nn.ModuleList([SelfAttention(embedding_dim // num_heads) for _ in range(num_heads)])
        # Projection layer for additional processing of the output of the attention heads.
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = torch.cat([head(x) for head in self.attention_heads], dim=2)
        x = self.dropout(self.projection(x))
        return x

# Feed Forward Network to process the output of the attention heads.
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffwd = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.ffwd(x)

# Each Transformer Block consists of a Multi Attention Head and a Feed Forward Network.
# It also has a Layer Normalization layer to normalize the output of the Multi Attention Head and the Feed Forward Network
# Residual connections are used to add the output of the Multi Attention Head and the Feed Forward Network to the input.
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_heads = MultiAttentionHead(embedding_dim // num_heads)
        self.feed_forward = FeedForward()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        # Residual connection is added to the output of the Multi Attention Head and the Feed Forward Network.
        x = x + self.attention_heads(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x

# The Transformer model consists of an Embedding layer, Positional Embedding layer, Transformer Blocks, and a Linear layer.
class Transformer(nn.Module):
    def __init__(self, vocab_size, context_size):
        super().__init__()
        # Embedding Layer to convert the tokens to vectors.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Poistional Embedding Layer to add the position of the tokens to the vectors.
        self.positional_embedding = nn.Embedding(context_size, embedding_dim)
        # Transformer Blocks to process the context of the input data.
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, x):
        _, T = x.shape
        token_embed = self.embedding(x)
        # Calculate the positionial embedding for the input data.
        position_emb = self.positional_embedding(torch.arange(T, device=device))
        x = token_embed + position_emb
        x = self.blocks(x)
        x = self.linear(self.layer_norm(x))
        return x


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
def train(model):
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
def validate(model):
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



model = Transformer(len(vocabulary), context_size)
model = model.to(device)
model.train()
train(model)
with torch.no_grad():
    validate(model)
    for i in range(5):
        generate(model, "Once", 512)



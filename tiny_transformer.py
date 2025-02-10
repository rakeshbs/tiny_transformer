import torch
import torch.nn as nn
from datasets import load_dataset

device = torch.device("mps")
# Network Parameters
batch_size = 64
learning_rate = 1e-3
context_size = 128
num_epochs = 10000
embedding_dim = 128
num_heads = 4
dropout_rate = 0.1
num_blocks = 4

ds = load_dataset("minnbanya/nlp-a2-sherlock")
# Concatenate all the text in the train and validation sets
train_data = "".join([ds["train"][i]["text"] for i in range(len(ds["train"]))])
validation_data = "".join([ds["validation"][i]["text"] for i in range(len(ds["validation"]))])

# Extract the vocabulary for making the tokenizer
vocabulary = sorted(set(list(train_data + validation_data)))
print(f"Vocabulary size: {len(vocabulary)}")
print(f"Vocabulary : {vocabulary}")
vocab_size = len(vocabulary)


# Create the character to token mapping.
char_to_token = {char: idx for idx, char in enumerate(vocabulary)}
# Create the token to character mapping.
token_to_char = {idx: char for char, idx in char_to_token.items()}

# Convert the text to a list of tokens
def encode(text):
    return [char_to_token[char] for char in text]

# Convert the list of tokens to text
def decode(encoded_text):
    return "".join([token_to_char[idx] for idx in encoded_text])

# Encode the train and validation data
encoded_train_data = encode(train_data)
encoded_validation_data = encode(validation_data)

# Create batches of encoded data based on the context size and batch size by randomly sampling the encoded data.
def get_batch_data(data_type="train"):
    data = encoded_train_data
    if data_type == "validation":
        data = encoded_validation_data

    x, y = [], []
    for _ in range(0, batch_size):
        idx = torch.randint(0, len(data) - context_size, (1,))
        x.append(torch.tensor(data[idx:idx + context_size]))
        y.append(torch.tensor(data[idx + 1:idx + context_size + 1]))

    x, y = torch.stack(x), torch.stack(y)
    return x, y

# Single Attention Head to process the context of input data
class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(embedding_dim, head_size)
        self.key = nn.Linear(embedding_dim, head_size)
        self.value = nn.Linear(embedding_dim, head_size)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size).to(device)))

    def forward(self, x):
        _, T, _ = x.shape
        # This is the attention mechanism.
        # Each token produces a query, key, and value vector.
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # The query vector is multiplied with the key vector to get the attention weights.
        attention = torch.matmul(q, k.transpose(-2, -1)) * context_size ** -0.5
        # The attention weights are masked to prevent the model from looking into the future.
        # A lower triangular matrix is used to mask the attention weights.
        attention = attention.masked_fill(self.tril[:T, :T]== 0, float("-inf"))
        # softmax is applied to the attention weights to get the final attention weights.
        attention = torch.nn.functional.softmax(attention, dim=-1)
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
        x = self.norm1(x + self.attention_heads(x))
        x = self.norm2(x + self.feed_forward(x))
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
        self.block = nn.Sequential(*[TransformerBlock() for _ in range(num_blocks)])
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        _, T = x.shape
        token_embed = self.embedding(x)
        # Calculate the positionial embedding for the input data.
        position_emb = self.positional_embedding(torch.arange(T, device=device))
        x = token_embed + position_emb
        x = self.block(x)
        x = self.linear(self.layer_norm(x))
        return x


# Function to train the model
def train(model):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    print("Training started")
    for epoch in range(num_epochs):
        x, y = get_batch_data()
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = loss_fn(output.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Function to validate the model
def validate(model):
    loss_fn = nn.CrossEntropyLoss()
    x, y = get_batch_data("validation")
    x = x.to(device)
    y = y.to(device)
    output = model(x)
    loss = loss_fn(output.view(-1, vocab_size), y.view(-1))
    print(f"Validation Loss: {loss.item()}")

#Function to generate text from the model
def generate(model, start_text, num_chars):
    chars = torch.tensor(encode(start_text)).to(device)
    chars = chars.view(1, len(chars))
    for i in range(num_chars):
        output = model(chars)
        prob = torch.nn.functional.softmax(output[0, -1], dim=0)
        idx = torch.multinomial(prob, num_samples=1)
        char = decode(idx.cpu().numpy())
        print(char, end="")
        chars = torch.cat([chars, idx.view(1, 1)], dim=1)
        chars = chars[:, -context_size:]


model = Transformer(len(vocabulary), context_size)
model = model.to(device)
model.train()
train(model)
with torch.no_grad():
    validate(model)
    generate(model, "Sherlock Holmes", 100)

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

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))

    def forward(self, x):
        _, T, _ = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attention = torch.matmul(q, k.transpose(-2, -1)) * context_size ** -0.5
        attention = attention.masked_fill(self.tril[:T, :T]== 0, float("-inf"))
        attention = torch.nn.functional.softmax(attention, dim=-1)
        output = torch.matmul(attention, v)
        return output

class Transformer(nn.Module):

    def __init__(self, vocab_size, context_size):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention_head = SelfAttention(embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.attention_head(x)
        x = self.linear(x)
        return x


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

def validate(model):
    loss_fn = nn.CrossEntropyLoss()
    x, y = get_batch_data("validation")
    x = x.to(device)
    y = y.to(device)
    output = model(x)
    loss = loss_fn(output.view(-1, vocab_size), y.view(-1))
    print(f"Validation Loss: {loss.item()}")

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

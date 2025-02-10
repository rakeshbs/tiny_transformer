import torch
import torch.nn as nn
from datasets import load_dataset


# Network Parameters
batch_size = 1
learning_rate = 1e-3
context_size = 128

ds = load_dataset("minnbanya/nlp-a2-sherlock")
# Concatenate all the text in the train and validation sets
train_data = "".join([ds["train"][i]["text"] for i in range(len(ds["train"]))])
validation_data = "".join([ds["validation"][i]["text"] for i in range(len(ds["validation"]))])

# Extract the vocabulary for making the tokenizer
vocabulary = sorted(set(list(train_data + validation_data)))
print(f"Vocabulary size: {len(vocabulary)}")
print(f"Vocabulary : {vocabulary}")


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

x,y = get_batch_data()

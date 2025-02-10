import torch
import torch.nn as nn
from datasets import load_dataset


ds = load_dataset("minnbanya/nlp-a2-sherlock")
# Concatenate all the text in the train and validation sets
train_data = "".join([ds["train"][i]["text"] for i in range(len(ds["train"]))])
validation_data = "".join([ds["validation"][i]["text"] for i in range(len(ds["validation"]))])

# Extract the vocabulary for making the tokenizer
vocabulary = sorted(set(list(train_data + validation_data)))
print(f"Vocabulary size: {len(vocabulary)}")
print(f"Vocabulary : {vocabulary}")

encoded_train_data = tokenizer.encode(train_data)
encoded_validation_data = tiny_tokenizer.encode(validation_data)

# Create the character to token mapping.
self.char_to_idx = {char: idx for idx, char in enumerate(vocabulary)}
# Create the token to character mapping.
self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

# Convert the text to a list of token indices
def encode(self, text):
    return [self.char_to_idx[char] for char in text]

# Convert the list of token indices to text
def decode(self, encoded_text):
    return "".join([self.idx_to_char[idx] for idx in encoded_text])

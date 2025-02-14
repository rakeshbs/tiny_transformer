import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Special Tokens
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"

class CharTokenizer:

    @classmethod
    def from_data(cls, text):
        """Create a character-based tokenizer from a dataset split."""
        vocabulary = sorted(set(list(text)) | {SOS_TOKEN, EOS_TOKEN, PAD_TOKEN})
        return cls(vocabulary)

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.vocab_size = len(self.vocabulary)

        print(f"Vocabulary size: {self.vocab_size}")
        # Token mapping
        self.char_to_token = {char: idx for idx, char in enumerate(self.vocabulary)}
        self.token_to_char = {idx: char for char, idx in self.char_to_token.items()}

        self.start_token = self.char_to_token[SOS_TOKEN]
        self.end_token = self.char_to_token[EOS_TOKEN]
        self.pad_token = self.char_to_token[PAD_TOKEN]

    def encode(self, text):
        """Convert text into token indices."""
        return [self.char_to_token[char] for char in text]

    def decode(self, tokens):
        """Convert token indices back into text."""
        return "".join([self.token_to_char[idx] for idx in tokens])

    def save_to_file(self, path):
        """Save tokenizer to a file."""
        with open(path, "w") as f:
            f.write("\n".join(self.vocabulary))

    @classmethod
    def read_from_file(cls, path):
        with open(path, "r") as f:
            string = sorted(set(f.read().splitlines()))
            return cls(string)

class TinyStoriesDataset(Dataset):
    def __init__(self, dataset_split, tokenizer, context_size):
        """
        dataset_split: List of dictionary samples (already loaded dataset split)
        tokenizer: Shared CharTokenizer instance
        context_size: Number of tokens per window
        """
        self.tokenizer = tokenizer
        self.context_size = context_size

        # Get stories with valid length
        offset = 5
        self.stories = [sample["text"] for sample in dataset_split if len(sample["text"]) < context_size - offset]

        self.encoded_stories = []
        start_token = tokenizer.start_token
        end_token = tokenizer.end_token
        padding_token = tokenizer.pad_token

        for story in self.stories:
            encoded_story = self.tokenizer.encode(story)
            encoded_story = [start_token] + encoded_story + [end_token]

            # Adjust padding
            padding = context_size - len(encoded_story)
            if padding > 0:
                encoded_story += [padding_token] * padding

            self.encoded_stories.append(encoded_story)

    def __len__(self):
        return len(self.encoded_stories)

    def __getitem__(self, idx):
        """Returns a single story as (input, target)."""
        x = self.encoded_stories[idx][:-1]
        y = self.encoded_stories[idx][1:]
        return torch.tensor(x), torch.tensor(y)


def test():
    context_size = 512
    batch_size = 128

    dataset = load_dataset("roneneldan/TinyStories")
    tokenizer = CharTokenizer(dataset["train"])
    train_dataset = TinyStoriesDataset(dataset["train"], tokenizer, context_size)
    val_dataset = TinyStoriesDataset(dataset["validation"], tokenizer, context_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Total training stories: {len(train_dataset)}")
    print(f"Total validation stories: {len(val_dataset)}")

    example_train_batch = next(iter(train_loader))
    example_val_batch = next(iter(val_loader))
    print(f"Train batch shape: {example_train_batch[0].shape}")  # (batch_size, context_size)
    print(f"Validation batch shape: {example_val_batch[0].shape}")  # (batch_size, context_size)

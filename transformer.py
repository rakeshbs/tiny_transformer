import torch
import torch.nn as nn

# Single Attention Head to process the context of input data.
# This module computes query, key, and value vectors for each token, performs the scaled dot-product attention,
# applies a causal mask to ensure the model doesn't look ahead, and then produces the attended output.
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, head_size, context_size, dropout_rate, device):
        """
        Args:
            embedding_dim (int): The dimensionality of the input embeddings.
            head_size (int): The dimensionality for the query, key, and value vectors.
            context_size (int): Maximum length of the sequence (used for masking).
            dropout_rate (float): Dropout rate for attention probabilities.
            device (torch.device): The device to allocate tensors on.
        """
        super().__init__()
        # Linear layers to project the input into query, key, and value spaces.
        self.query = nn.Linear(embedding_dim, head_size)
        self.key = nn.Linear(embedding_dim, head_size)
        self.value = nn.Linear(embedding_dim, head_size)
        # Precompute the lower-triangular matrix for causal masking. This prevents tokens from attending to future tokens.
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size, device=device)))
        # Dropout layer for regularizing attention weights.
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim)
        Returns:
            torch.Tensor: Output tensor after applying self-attention.
        """
        batch_size, T, _ = x.shape
        
        # Compute the query, key, and value matrices using learned linear projections.
        q = self.query(x)  # Shape: (batch_size, T, head_size)
        k = self.key(x)    # Shape: (batch_size, T, head_size)
        v = self.value(x)  # Shape: (batch_size, T, head_size)
        
        # Calculate the scaling factor based on the head dimension.
        scale = q.size(-1) ** -0.5
        
        # Compute the dot products between queries and keys to obtain raw attention scores.
        # The operation produces a tensor of shape (batch_size, T, T)
        attention = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply the precomputed lower triangular mask to prevent tokens from attending to future tokens.
        # Positions where the mask is 0 are set to negative infinity.
        attention = attention.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        
        # Normalize the attention scores using softmax to obtain the attention weights.
        attention = torch.nn.functional.softmax(attention, dim=-1)
        
        # Apply dropout to the attention weights for regularization.
        attention = self.dropout(attention)
        
        # Multiply the attention weights with the value vectors to get the final output.
        # Resulting shape: (batch_size, T, head_size)
        output = torch.matmul(attention, v)
        return output

# Multi Attention Head to process the context of input data.
# This module consists of multiple SelfAttention heads operating in parallel.
# The outputs from all heads are concatenated and then passed through a linear projection and dropout layer.
class MultiAttentionHead(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate, context_size, device):
        """
        Args:
            embedding_dim (int): Dimensionality of the input embeddings.
            num_heads (int): Number of parallel attention heads.
            dropout_rate (float): Dropout rate for the attention and projection layers.
            context_size (int): Maximum sequence length for masking.
            device (torch.device): Device to allocate tensors on.
        """
        super().__init__()
        # Compute the dimension for each attention head.
        head_size = embedding_dim // num_heads
        # Create a list of SelfAttention heads.
        self.attention_heads = nn.ModuleList([
            SelfAttention(embedding_dim, head_size, context_size, dropout_rate, device)
            for _ in range(num_heads)
        ])
        # A linear projection layer to combine the outputs of the attention heads.
        self.projection = nn.Linear(embedding_dim, embedding_dim)
        # Dropout layer for additional regularization.
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim)
        Returns:
            torch.Tensor: Output tensor after processing through multiple attention heads.
        """
        # Process the input with each attention head and concatenate the results along the embedding dimension.
        x = torch.cat([head(x) for head in self.attention_heads], dim=2)
        # Apply a linear projection and dropout to fuse the multiple heads.
        x = self.dropout(self.projection(x))
        return x

# Feed Forward Network to process the output of the attention mechanism.
# Consists of a two-layer MLP with a ReLU activation in between and dropout for regularization.
class FeedForward(nn.Module):
    def __init__(self, embedding_dim, dropout_rate):
        """
        Args:
            embedding_dim (int): Dimensionality of the input and output.
            dropout_rate (float): Dropout rate for the feed-forward network.
        """
        super().__init__()
        self.ffwd = nn.Sequential(
            # Expand the embedding dimension to 4x before applying non-linearity.
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            # Project back to the original embedding dimension.
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim)
        Returns:
            torch.Tensor: Output tensor after the feed-forward network.
        """
        return self.ffwd(x)

# Transformer Block combining Multi-Attention Head and Feed Forward Network with Layer Normalization and Residual Connections.
class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout_rate, context_size, device):
        """
        Args:
            embedding_dim (int): Dimensionality of the model.
            num_heads (int): Number of attention heads.
            dropout_rate (float): Dropout rate for the block.
            context_size (int): Maximum sequence length for masking.
            device (torch.device): Device for tensor allocations.
        """
        super().__init__()
        # Multi-head self-attention component.
        self.attention_heads = MultiAttentionHead(embedding_dim, num_heads, dropout_rate, context_size, device)
        # Feed forward network component.
        self.feed_forward = FeedForward(embedding_dim, dropout_rate)
        # Layer normalization for stabilizing training.
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim)
        Returns:
            torch.Tensor: Output tensor after applying self-attention and feed-forward networks with residual connections.
        """
        # Apply layer normalization, followed by multi-head attention, then add the residual connection.
        x = x + self.attention_heads(self.norm1(x))
        # Apply layer normalization, followed by the feed-forward network, then add the residual connection.
        x = x + self.feed_forward(self.norm2(x))
        return x

# The overall Transformer model.
# It consists of an embedding layer, positional embeddings, a stack of Transformer blocks, and a final linear layer.
class Transformer(nn.Module):
    def __init__(self, vocab_size, context_size, embedding_dim, num_blocks, num_heads, dropout_rate, device):
        """
        Args:
            vocab_size (int): Size of the vocabulary (number of tokens).
            context_size (int): Maximum sequence length.
            embedding_dim (int): Dimensionality of the token embeddings.
            num_blocks (int): Number of stacked Transformer blocks.
            num_heads (int): Number of attention heads per block.
            dropout_rate (float): Dropout rate used throughout the model.
            device (torch.device): Device to run the model on.
        """
        super().__init__()
        # Embedding layer converts input tokens to vector representations.
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Positional embedding layer adds positional information to each token's embedding.
        self.positional_embedding = nn.Embedding(context_size, embedding_dim)
        # Stack multiple Transformer blocks for deep processing of the input context.
        self.blocks = nn.Sequential(*[
            TransformerBlock(embedding_dim, num_heads, dropout_rate, context_size, device)
            for _ in range(num_blocks)
        ])
        # A final layer normalization for the output of the Transformer blocks.
        self.layer_norm = nn.LayerNorm(embedding_dim)
        # Linear layer projects the normalized embeddings to the vocabulary size for final token predictions.
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.device = device

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length) containing token indices.
        Returns:
            torch.Tensor: Output logits for each token position of shape (batch_size, sequence_length, vocab_size).
        """
        batch_size, T = x.shape
        
        # Convert token indices into embeddings.
        token_embed = self.embedding(x)
        
        # Create a tensor of positions and convert them into positional embeddings.
        position_ids = torch.arange(T, device=self.device)
        position_emb = self.positional_embedding(position_ids)
        
        # Combine token embeddings with positional embeddings.
        x = token_embed + position_emb
        
        # Pass the combined embeddings through the stack of Transformer blocks.
        x = self.blocks(x)
        
        # Normalize the output, then project it to the vocabulary size for token prediction.
        x = self.linear(self.layer_norm(x))
        return x

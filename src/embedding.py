# import torch
# import torch.nn as nn

# class TokenPositionalEmbedding(nn.Module):
#     """
#     PyTorch module that handles both token and positional embeddings.
#     """
#     def __init__(self, vocab_size: int, output_dim: int, context_length: int):
#         super().__init__()
#         self.token_embedding = nn.Embedding(vocab_size, output_dim)
#         self.positional_embedding = nn.Embedding(context_length, output_dim)

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#         """
#         Inputs shape: (batch_size, seq_len)
#         Returns shape: (batch_size, seq_len, output_dim)
#         """
#         # Get token embeddings
#         token_emb = self.token_embedding(inputs)
        
#         # Get positional embeddings
#         pos_indices = torch.arange(inputs.shape[1])
#         pos_emb = self.positional_embedding(pos_emb)
        
#         # PyTorch automatically broadcasts the addition: 
#         # (batch, seq, dim) + (seq, dim) -> (batch, seq, dim)
#         return token_emb + pos_emb

import random

def get_shape(data):
    """Helper to get the shape of a nested list (mimics tensor.shape)."""
    shape = []
    d = data
    while isinstance(d, list):
        shape.append(len(d))
        if len(d) == 0:
            break
        d = d[0]
    return tuple(shape)

def add_embeddings(token_emb, pos_emb):
    """
    Adds token embeddings (3D list) and positional embeddings (2D list).
    Mimics PyTorch's broadcasting: (batch, seq, dim) + (seq, dim) -> (batch, seq, dim)
    """
    result = []
    for batch in token_emb:
        new_batch = []
        for i, token_vec in enumerate(batch):
            new_vec = [t + p for t, p in zip(token_vec, pos_emb[i])]
            new_batch.append(new_vec)
        result.append(new_batch)
    return result

class Embedding:
    """Pure Python implementation of an Embedding layer."""
    def __init__(self, vocab_size: int, output_dim: int):
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.weight = [
            [random.gauss(0.0, 1.0) for _ in range(output_dim)]
            for _ in range(vocab_size)
        ]

    def __call__(self, indices):
        if isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], int):
            return [self.weight[i] for i in indices]
        elif isinstance(indices, list) and len(indices) > 0 and isinstance(indices[0], list):
            return [
                [self.weight[token_id] for token_id in seq]
                for seq in indices
            ]
        else:
            raise ValueError("Indices must be a 1D or 2D list of integers.")

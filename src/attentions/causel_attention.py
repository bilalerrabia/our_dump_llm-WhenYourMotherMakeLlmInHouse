import torch
from torch import nn

inputs = torch.tensor(
    [[0.43, 0.15, 0.89, 0.12], # Your     (x^1)
    [0.55, 0.87, 0.66, 0.12], # journey  (x^2)
    [0.57, 0.85, 0.64, 0.12], # starts   (x^3)
    [0.22, 0.58, 0.33, 0.12], # with     (x^4)
    [0.77, 0.25, 0.10, 0.12], # one      (x^5)
    [0.05, 0.80, 0.55, 0.12]] # step     (x^6)
)
class CausalAttention(nn.Module):

    def __init__(self, d_in, d_out, context_length,
                dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) # New
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1)) # New

    def forward(self, x):
        b, num_tokens, d_in = x.shape # New batch dimension b
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # Changed transposefo
        attn_scores.masked_fill_(  # New, _ ops are in-place
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)  # `:num_tokens` to account for cases where the number of tokens in the batch is smaller than the supported context_size
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights) # New

        context_vec = attn_weights @ values
        return context_vec

# batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape) 
# torch.manual_seed(123)
# context_length = batch.shape[1]
# ca = CausalAttention(d_in, d_out, context_length, 0.0)
# context_vecs = ca(batch)
# print("context_vecs.shape:", context_vecs.shape)
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

# x_2 = inputs[1] #A
d_in = inputs.shape[1] #B
d_out = 4 #C

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

keys = inputs @ W_key
values = inputs @ W_value
queries = inputs @ W_query
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
print("queries.shape:", queries.shape)

# 2-the attention scores matrices : Q * KT = [6 * 6]
attn_scores = queries @ keys.T
# print(attn_scores)

# 3-the attention weights: softmax(Q * KT / sqrt(dk))
d_k = keys.shape[-1]
attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
print(attn_weights)

# 4-context vectore: attention weights * V.
context_vec = attn_weights @ values
print(context_vec)


class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec

attention = SelfAttention(4, 4)

attention.forward(inputs)

# import os
# import sys
# import torch

# # Add the src directory to the system path so we can import our modules
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# from src.tokenizer import Tokenizer
# from src.data_loader import DatasetPreparation, create_dataloader
# from src.embedding import TokenPositionalEmbedding

# def main():
#     # 1. Read the data
#     data_file = "data_test.txt"
#     try:
#         with open(data_file, "r", encoding="utf-8") as f:
#             data_str = f.read()
#     except IOError as e:
#         print(f"Error reading file {data_file}: {e}")
#         return

#     print(f"Original text length: {len(data_str)} characters")

#     # 2. Initialize and TRAIN the tokenizer
#     tokenizer = Tokenizer()
#     num_merges = 100
#     print(f"Training tokenizer with {num_merges} merges...")
#     tokenizer.train(data_str, num_merges=num_merges)

#     # 3. Encode the text
#     data_tokens = tokenizer.encode(data_str)
#     print(f"Total tokens after encoding: {len(data_tokens)}")

#     # 4. Create Dataset and DataLoader
#     context_window = 4
#     stride = 4
#     batch_size = 8
    
#     dataset = DatasetPreparation(data_tokens, context_window=context_window, stride=stride)
#     dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

#     # 5. Fetch the first batch
#     print("\n--- Fetching First Batch ---")
#     data_iter = iter(dataloader)
#     inputs, targets = next(data_iter)

#     # Convert pure Python lists to PyTorch Tensors
#     # Shape will be (batch_size, context_window)
#     inputs_tensor = torch.tensor(inputs, dtype=torch.long)
#     targets_tensor = torch.tensor(targets, dtype=torch.long)

#     print("Inputs (Tensor):")
#     print(inputs_tensor)
#     print("Inputs shape:", inputs_tensor.shape)

#     # 6. Embeddings
#     print("\n--- Running Embeddings ---")
#     # Vocab size is 256 (bytes) + num_merges we trained
#     vocab_size = 256 + num_merges 
#     output_dim = 256

#     # Initialize the PyTorch embedding layer
#     embedding_layer = TokenPositionalEmbedding(vocab_size, output_dim, context_window)

#     # Pass the tensor through the layer
#     final_embeddings = embedding_layer(inputs_tensor)
    
#     print("Final input embeddings shape:", final_embeddings.shape)

# if __name__ == "__main__":
#     main()

import os
import sys
import random

# Add the src directory to the system path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.tokenizer import Tokenizer
from src.data_loader import DatasetPreparation, create_dataloader
from src.embedding import Embedding, get_shape, add_embeddings

def main():
    # 1. Read the data
    data_file = "data_test.txt"
    vocab_cache_file = "tokenizer_vocab.json" # Cache file path

    try:
        with open(data_file, "r", encoding="utf-8") as f:
            data_str = f.read()
    except IOError as e:
        print(f"Error reading file {data_file}: {e}")
        return
    num_merges = len(data_str) // 34 # 3% of the training data
    print(f"Original text length: {len(data_str)} characters")

    # 2. Initialize Tokenizer
    tokenizer = Tokenizer()

    # Check if we have a cached vocabulary
    if os.path.exists(vocab_cache_file):
        tokenizer.load_vocab(vocab_cache_file)
    else:
        print(f"No cache found. Training tokenizer with {num_merges} merges...")
        tokenizer.train(data_str, num_merges=num_merges)
        tokenizer.save_vocab(vocab_cache_file)

    # Calculate and print the vocabulary size based on loaded merges
    # 256 base bytes + length of map_replace
    vocab_size = 256 + len(tokenizer.map_replace) 
    print(f"Vocabulary size: {vocab_size}")

    # 3. Encode the text
    data_tokens = tokenizer.encode(data_str)
    print(f"Total tokens after encoding: {len(data_tokens)}")

    # 4. Create Dataset and DataLoader
    context_window = 4
    stride = 4
    batch_size = 8
    
    dataset = DatasetPreparation(data_tokens, context_window=context_window, stride=stride)
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # 5. Fetch the first batch
    print("\n--- Fetching First Batch ---")
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    print("Inputs (Token IDs):")
    for seq in inputs:
        print(seq)

    # 6. Embeddings (Pure Python)
    print("\n--- Running Embeddings ---")
    output_dim = 256
    random.seed(123) # Set seed for reproducible random weights

    # Initialize the token embedding layer
    token_embedding_layer = Embedding(vocab_size, output_dim)
    token_embeddings = token_embedding_layer(inputs)
    print("Token embeddings shape:", get_shape(token_embeddings))

    # Initialize the positional embedding layer
    pos_embedding_layer = Embedding(context_window, output_dim)
    
    # Generate positional indices [0, 1, 2, 3]
    pos_indices = list(range(context_window))
    pos_embeddings = pos_embedding_layer(pos_indices)
    print("Positional embeddings shape:", get_shape(pos_embeddings))

    # Add them together using our custom pure Python broadcasting function
    final_embeddings = add_embeddings(token_embeddings, pos_embeddings)
    print("Final input embeddings shape:", get_shape(final_embeddings))

if __name__ == "__main__":
    main()
import random

class DatasetPreparation:
    """
    Pure Python Dataset class to create input-target pairs.
    """
    def __init__(self, ids: list[int], context_window: int, stride: int):
        self.input_ids: list[list[int]] = []
        self.target_ids: list[list[int]] = []
        
        for i in range(0, len(ids) - context_window, stride):
            self.input_ids.append(ids[i : i + context_window])
            self.target_ids.append(ids[i + 1 : i + context_window + 1])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(dataset, batch_size=4, shuffle=True, drop_last=True):
    """
    Pure Python generator to mimic PyTorch's DataLoader.
    Yields batches of (inputs, targets).
    """
    indices = list(range(len(dataset)))
    
    if shuffle:
        random.shuffle(indices)
        
    if drop_last:
        num_batches = len(indices) // batch_size
    else:
        num_batches = (len(indices) + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch_indices = indices[i * batch_size : (i + 1) * batch_size]
        
        batch_inputs = []
        batch_targets = []
        
        for idx in batch_indices:
            inp, tgt = dataset[idx]
            batch_inputs.append(inp)
            batch_targets.append(tgt)
            
        yield batch_inputs, batch_targets
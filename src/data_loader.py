"""Pure-Python dataset and dataloader (mimicking torch.utils.data)."""

import random
from collections.abc import Iterator

Sample = tuple[list[int], list[int]]
Batch = tuple[list[list[int]], list[list[int]]]


class DatasetPreparation:
    """Create input/target pairs with a sliding window over token ids.

    For every window start ``i`` (stepping by ``stride``):
      input  = ids[i : i + context_window]
      target = ids[i + 1 : i + context_window + 1]
    """

    def __init__(self, ids: list[int], context_window: int, stride: int) -> None:
        self.input_ids: list[list[int]] = []
        self.target_ids: list[list[int]] = []

        for i in range(0, len(ids) - context_window, stride):
            self.input_ids.append(ids[i : i + context_window])
            self.target_ids.append(ids[i + 1 : i + context_window + 1])

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> Sample:
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    dataset: DatasetPreparation,
    batch_size: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
) -> Iterator[Batch]:
    """Yield batches of (inputs, targets) as a generator."""
    indices = list(range(len(dataset)))

    if shuffle:
        random.shuffle(indices)

    if drop_last:
        num_batches = len(indices) // batch_size
    else:
        num_batches = -(-len(indices) // batch_size)  # ceil division

    for i in range(num_batches):
        batch_indices = indices[i * batch_size : (i + 1) * batch_size]
        samples = [dataset[idx] for idx in batch_indices]

        batch_inputs = [inp for inp, _ in samples]
        batch_targets = [tgt for _, tgt in samples]
        yield batch_inputs, batch_targets
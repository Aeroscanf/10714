import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(self, dataset: Dataset, batch_size: Optional[int] = 1, shuffle: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.ordering = np.arange(len(dataset))
        self.current_index = 0

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.ordering)
        return self

    def __next__(self):
        if self.current_index >= len(self.dataset):
            raise StopIteration
        
        batch_indices = self.ordering[self.current_index:self.current_index + self.batch_size]
        
        # Get a batch of data
        batch = [self.dataset[int(i)] for i in batch_indices]
    
        self.current_index += self.batch_size

        # Convert batch to Tensors
        if isinstance(batch[0], tuple):
            # If dataset returns tuples (e.g., data and labels)
            return tuple(Tensor(np.array([item[i] for item in batch])) for i in range(len(batch[0])))

        else:
            # If dataset returns single items
            return Tensor(np.array(batch))

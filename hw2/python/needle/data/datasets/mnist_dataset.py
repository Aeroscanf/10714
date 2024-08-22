from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.images, self.labels = self.parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def parse_mnist(self, image_filename, label_filename):
        with gzip.open(image_filename, 'rb') as f:
            magic, num, row, col = np.frombuffer(f.read(16), dtype = '>i4')
            X = np.frombuffer(f.read(), dtype = np.uint8).reshape(num, row*col)

            X = X.astype(np.float32)
            X -= np.min(X)
            X /= np.max(X)
    
        with gzip.open(label_filename, 'rb') as f:
            magic, num = np.frombuffer(f.read(8), dtype = '>i4')
            y = np.frombuffer(f.read(), dtype = np.uint8)
    
        return X, y

#    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
#        imgs = self.images[index]
#        label = self.labels[index]

#        if len(imgs.shape) > 1:
#            imgs = np.vstack([self.apply_transforms(img.reshape(28, 28, 1)) for img in imgs])
#        else:
#            imgs = self.apply_transforms(imgs.reshape(28, 28, 1))
#        return (imgs, label)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        imgs = self.images[index]
        labels = self.labels[index]

        # Check if we're dealing with a single image or multiple images
        if imgs.ndim == 1:
            # Single image
            img = imgs.reshape(28, 28, 1)
            if self.transforms:
                img = self.apply_transforms(img)
            return (img.flatten(), labels)
        else:
            # Multiple images
            imgs = imgs.reshape(-1, 28, 28, 1)
            if self.transforms:
                imgs = np.stack([self.apply_transforms(img) for img in imgs])
            return (imgs.reshape(imgs.shape[0], -1), labels)

#    def __getitem__(self, index) -> object:
#            img = self.images[index]
#            label = self.labels[index]

            # Correct reshape to match MNIST image format
#            img = img.reshape((28, 28))  # Reshape to 28x28 
#            img = img.reshape(28, 28, 1)  # Add a channel dimension (1 for grayscale)

            # Apply transforms (if any)
#            img = self.apply_transforms(img)

#            return img, label

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION
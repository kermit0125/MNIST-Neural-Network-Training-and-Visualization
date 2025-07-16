import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


#PyTorch data loading tool
    #Load MNIST from npz file
    #Convert to float32 / [0,1] Normalize
    #Increase channel dimension (1, 28, 28)
class NpzMNIST(Dataset):
    def __init__(self, npz_path, train=True, transform=None):
        # Load the npz file
        data = np.load(npz_path)
        if train:
            self.images = data["x_train"]
            self.labels = data["y_train"]
        else:
            self.images = data["x_test"]
            self.labels = data["y_test"]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]          # shape (28,28)
        label = int(self.labels[idx])

        # Convert to float32 and normalize to [0,1]
        image = image.astype(np.float32) / 255.0

        # Add channel dimension (1,28,28)
        image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(image)

        # Convert to torch tensor
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label


def get_dataloaders(batch_size=64, npz_path=None):
    """
    Returns train_loader and test_loader using custom NpzMNIST dataset.
    """
    transform = None  # Not strictly needed here since we pre-process manually

    train_dataset = NpzMNIST(
        npz_path=npz_path,
        train=True,
        transform=transform
    )
    test_dataset = NpzMNIST(
        npz_path=npz_path,
        train=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader

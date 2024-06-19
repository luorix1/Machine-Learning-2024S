import random
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import json

class MLDataset(Dataset):
    def __init__(self, img_path, label_path, augmentations=[]):
        self.path = img_path
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        # NOTE: (added) whether to use data augmentation
        self.augmentations = augmentations

         # Define individual augmentation transformations
        self.available_transforms = {
            "random_translate": transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.2),
            "color_jitter": transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.5),
            "random_rotation": transforms.RandomApply([transforms.RandomAffine(degrees=8)], p=0.2),
            "gaussian_blur": transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.2),
        }

        # Create the transform list based on provided augmentations
        self.transform = transforms.Compose(
            [self.available_transforms[aug] for aug in self.augmentations if aug in self.available_transforms]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgs = np.load(f"{self.path}/{idx}.npy")
        
        seq_length = imgs.shape[0]
        label = self.labels[str(idx)]

        # NOTE: changed when numpy array is converted to tensor
        imgs_tensor = torch.tensor(imgs, dtype=torch.float32)
        
        # NOTE: (added) data augmentation
        if self.augmentations:
            # switch to channel first
            imgs_tensor = imgs_tensor.permute(0, 3, 1, 2)
            imgs_tensor = self.transform(imgs_tensor)
            # switch back to channel last
            imgs_tensor = imgs_tensor.permute(0, 2, 3, 1)

        # NOTE: changed when numpy array is converted to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        # Apply random character swap
        if "random_swap" in self.augmentations and seq_length > 1:
            if random.random() < 0.1:  # Swap with a low probability
                i, j = random.sample(range(seq_length), 2)
                imgs_tensor[[i, j]] = imgs_tensor[[j, i]]
                label_tensor[[i, j]] = label_tensor[[j, i]]
        
        return imgs_tensor, label_tensor, seq_length


def collate_fn(batch):
    sequences, targets, lengths = zip(*batch)

    padded_sequences = pad_sequence([seq for seq in sequences], batch_first=True)
    padded_targets = pad_sequence([tar for tar in targets], batch_first=True)
    
    return padded_sequences, padded_targets, torch.tensor(lengths, dtype=torch.float32)

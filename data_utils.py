import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import json

class MLDataset(Dataset):
    def __init__(self, img_path, label_path, use_aug=False):
        self.path = img_path
        with open(label_path, 'r') as f:
            self.labels = json.load(f)

        # NOTE: (added) whether to use data augmentation
        self.use_aug = use_aug

        # FIXME: random grayscale, swapping out images corresponding to identical letters -> must implement
        self.transform = transforms.Compose([
            # NOTE: (added) data augmentation - Random Translation
            # translate randomly between -0.1 and 0.1 with a probability of 0.2
            transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.2),
            # NOTE: (added) data augmentation - Color Jitter
            # randomly change the brightness, contrast, saturation, and hue with a probability of 0.2
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)], p=0.5),
            # NOTE: (added) data augmentation - Random Rotation
            # rotate randomly between -8 and 8 degrees with a probability of 0.2
            transforms.RandomApply([transforms.RandomAffine(degrees=8)], p=0.2),
            # NOTE: (added) data augmentation - Gaussian Blur
            # apply gaussian blur with a probability of 0.2
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3))], p=0.2)
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        imgs = np.load(f"{self.path}/{idx}.npy")
        
        seq_length = imgs.shape[0]
        label = self.labels[str(idx)]

        # NOTE: changed when numpy array is converted to tensor
        imgs_tensor = torch.tensor(imgs, dtype=torch.float32)
        
        # NOTE: (added) data augmentation
        if self.use_aug:
            # switch to channel first
            imgs_tensor = imgs_tensor.permute(0, 3, 1, 2)
            imgs_tensor = self.transform(imgs_tensor)
            # switch back to channel last
            imgs_tensor = imgs_tensor.permute(0, 2, 3, 1)

        # NOTE: changed when numpy array is converted to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return imgs_tensor, label_tensor, seq_length


def collate_fn(batch):
    sequences, targets, lengths = zip(*batch)

    padded_sequences = pad_sequence([seq for seq in sequences], batch_first=True)
    padded_targets = pad_sequence([tar for tar in targets], batch_first=True)
    
    return padded_sequences, padded_targets, torch.tensor(lengths, dtype=torch.float32)

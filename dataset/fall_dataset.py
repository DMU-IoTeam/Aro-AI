import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob, os

class FallDataset(Dataset):
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        label = self.labels[idx]
        return torch.tensor(data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

def load_dataset(fall_dir, normal_dir):
    fall_files = glob.glob(os.path.join(fall_dir, "*.npy"))
    normal_files = glob.glob(os.path.join(normal_dir, "*.npy"))
    files = fall_files + normal_files
    labels = [1]*len(fall_files) + [0]*len(normal_files)
    return files, labels

def get_dataloaders(CFG):
    train_files, train_labels = load_dataset(CFG.train_fall_dir, CFG.train_normal_dir)
    val_files, val_labels = load_dataset(CFG.val_fall_dir, CFG.val_normal_dir)

    train_loader = DataLoader(FallDataset(train_files, train_labels), batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(FallDataset(val_files, val_labels), batch_size=CFG.batch_size, shuffle=False)
    return train_loader, val_loader

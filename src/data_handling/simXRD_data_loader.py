import torch
from torch.utils.data import Dataset, DataLoader
from ase.db import connect
import numpy as np

class simXRDDataset(Dataset):
    def __init__(self, db_path):
        self.db = connect(db_path)
        self.length = self.db.count()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        row = self.db.get(idx + 1)  # ASE db indexing starts at 1
        
        # Extract features (Note that it is normalised to 100)
        intensity = np.array(eval(row.intensity), dtype=np.float32)
        
        # Extract labels
        space_group = eval(row.tager)[0]
        
        # Convert to tensor
        intensity_tensor = torch.from_numpy(intensity)
        space_group_tensor = torch.tensor(space_group, dtype=torch.long)
        
        return intensity_tensor, space_group_tensor

def create_data_loaders(train_path, val_path, test_path, batch_size=32, num_workers=3):
    train_dataset = simXRDDataset(train_path)
    val_dataset = simXRDDataset(val_path)
    test_dataset = simXRDDataset(test_path)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader
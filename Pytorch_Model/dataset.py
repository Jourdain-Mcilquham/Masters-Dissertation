import os
os.environ["TORCH_USE_CUDA_DSA"] = "1"
from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd

class SpectralDataset(Dataset):
    def __init__(self, batch_size, data_location="....h5"):
        super().__init__()
        self.batch_size = batch_size 
        self.data_location = data_location
        with pd.HDFStore(self.data_location, mode='r') as newstore:
            self.num_batches = newstore.get("data").shape[0]//batch_size
            # self.num_batches = 1000
            print(f"Number of batches: {self.num_batches}")

    def __getitem__(self, index):
        # Load a batch of data
        with pd.HDFStore(self.data_location, mode='r') as newstore:
            df_restored = newstore.select('data',
                                            start=index * self.batch_size,
                                            stop=(index+1) * self.batch_size)
            batch = np.asarray(df_restored)
            return torch.from_numpy(batch).float()
        
    
    def __len__(self):
        return self.num_batches    

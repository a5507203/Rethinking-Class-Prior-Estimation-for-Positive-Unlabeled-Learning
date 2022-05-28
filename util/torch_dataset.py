import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms



class DatasetArray(Dataset):
    
    def __init__(self, data, labels=None, transform=None):
        if labels != None:
            self.data_arr = np.asarray(data).astype(np.float32)
            self.label_arr = np.asarray(labels).astype(np.long)
        else:
            tmp_arr = np.asarray(data)
            self.data_arr = tmp_arr[:,:-1].astype(np.float32)
            self.label_arr = tmp_arr[:,-1].astype(np.long)
        self.transform = transform

        
    def __len__(self):
        return len(self.data_arr)
    
    def __getitem__(self, index):
     
        data = self.data_arr[index]
        label = self.label_arr[index]
        
        if self.transform is not None:
            data = self.transform(data)
            
        return (data, label)
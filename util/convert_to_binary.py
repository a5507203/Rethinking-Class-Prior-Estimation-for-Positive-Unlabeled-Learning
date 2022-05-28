from csv_rw_pd import *
import pandas as pd 
import numpy as np
import random
from util.torch_dataset import DatasetArray
from torch.utils.data import Subset
from data_split import to_binary_dataset, getData



def write_csv_pd(df, file_path, header=None):
    df.to_csv(file_path, index = None, header=None)


df = getData("./dataset/shuttle_binary.data", header=None)
df = to_binary_dataset(df)
print(sample_summary(df.values.tolist()))
write_csv_pd(df,"./dataset/Statlog_binary.data")
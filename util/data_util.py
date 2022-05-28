import pandas as pd 
import numpy as np
import random

from util.torch_dataset import DatasetArray
from torch.utils.data import Subset

from sklearn.preprocessing import LabelEncoder,StandardScaler
from numbers import Number
import numpy as np



def convert_format(sample):
    sample = str(sample)
    sample = sample.replace("], [","; ").replace(",","").replace("[[","").replace("]]","")
    return sample
    

def to_binary_dataset(df):
    label_index = df.columns[-1]
    df.loc[df.loc[:,label_index]>1,label_index]=1
    return df



def relabel(data, new_label = 0):
    arr = np.array(data)
    arr[:,-1]=new_label
    return arr.tolist()



def subsampling(samples, sample_lengths=[400,400]):
    subsamples = []
    unusedsamples = []
    for i in range(len(samples)):
        sample = samples[i]
        n = len(sample)
        idxs = [j for j in range(n)]
        print(int(sample_lengths[i]))
        print(n)
        subsample_idxs = np.random.choice(n, int(sample_lengths[i]), replace = True)
        unusedsamples_idxs = list(set(idxs)-set(subsample_idxs))

        subsample = [sample[idx] for idx in subsample_idxs]
        unusedsample = [sample[idx] for idx in unusedsamples_idxs]

        subsamples.append(subsample)
        unusedsamples.append(unusedsample)
   
    return subsamples, unusedsamples



def random_pu_split(df, positive_label, positive_frac = 0.25):
    
    label_index = df.columns[-1]
    idxs_of_the_label = list(df.loc[df.loc[:,label_index]==positive_label,label_index].index)
    number_of_positive_examples = int(len(idxs_of_the_label)*positive_frac)
    positive_idxs = random.sample(idxs_of_the_label,number_of_positive_examples)
    mixture_idxs = list(set(df.index)-set(positive_idxs))

    return df.iloc[mixture_idxs,:].values.tolist(), df.iloc[positive_idxs,:].values.tolist()




def getData( file_path ,header = None ):
    df = pd.read_csv(file_path, header = None)
    #change the value of the attribute to numberical value if is not
    enc = LabelEncoder()
    df = df.dropna(how='any',axis=0) 
    for i in range (len(df.columns)):

        col = df.columns[i]

        if(isinstance(df[col].iloc[0], Number)):
            continue
        enc.fit(df[col])
        df[col] = enc.transform(df[col])
    scaler = StandardScaler()
    scaled_df = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_df)
    scaled_df.iloc[:,-1] = df.iloc[:,-1]
    return scaled_df

import numpy as np
import random
import torch
import torch.utils.data
from torch.utils.data import Subset
import torchvision
import torchvision.models
import torchvision.transforms

from util.torch_dataset import DatasetArray
from util.data_util import *
from util.dataset_util import *



def get_loader(positive_label = 1, batch_size =100, num_workers = 0, dataset_path="./dataset/shuttle_binary.data", total_sample_length=800,positive_frac=0.25,val_set_frac=0.2):

    df = getData(dataset_path)

    # build samples i.i.d. drawn from F and H respectively
    # AS MENTIONED IN OUR PAPER: the mixture and component samples have the same size as did in (Ramaswamy et al., 2016)
    mix_comp_sample_length = total_sample_length/2
    val_set_length = mix_comp_sample_length*val_set_frac
    mixture_sample, component_sample = random_pu_split(df, positive_label=positive_label, positive_frac=positive_frac)

    # sample the training data
    [tr_unlabeled_sample, tr_positive_sample],[unused_mixture_sample, unused_component_sample] = subsampling(
        samples=[mixture_sample, component_sample],
        sample_lengths=[mix_comp_sample_length,mix_comp_sample_length]
    )
    
    # sample some validation data from unused data set
    [val_unlabeled_sample, val_positive_sample],_ = subsampling(
        samples=[unused_mixture_sample, unused_component_sample],
        sample_lengths=[min(val_set_length,len(unused_mixture_sample)), min(val_set_length,len(unused_component_sample))]
    )

    # if the dataset does not have enough validation data, we also sample some validation data from the training set 
    [temp_val_mixture_sample, temp_val_component_sample],_ = subsampling(
        samples=[tr_unlabeled_sample, tr_positive_sample],
        sample_lengths=[max(val_set_length-len(val_unlabeled_sample),0), max(val_set_length-len(unused_component_sample),0)]
    )

    val_unlabeled_sample += temp_val_mixture_sample
    val_positive_sample += temp_val_component_sample

    class_prior = sample_summary(tr_unlabeled_sample=tr_unlabeled_sample, positive_label=positive_label)
    kp_star_val = sample_summary(tr_unlabeled_sample=val_unlabeled_sample, positive_label=positive_label)

    tr_unlabeled_sample = relabel(data = tr_unlabeled_sample, new_label = 0)
    tr_positive_sample = relabel(data = tr_positive_sample, new_label = 1)

    val_unlabeled_sample = relabel(data = val_unlabeled_sample, new_label = 0)
    val_positive_sample = relabel(data = val_positive_sample, new_label = 1)


    tr_sample = tr_unlabeled_sample + tr_positive_sample
    val_sample = val_unlabeled_sample + val_positive_sample

    # build pytorch dataset and loader
    train_dataset = DatasetArray(data = tr_sample)
    validation_dataset = DatasetArray(data = val_sample)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=100,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=100,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, tr_unlabeled_sample, tr_positive_sample, val_unlabeled_sample, val_positive_sample, class_prior, kp_star_val


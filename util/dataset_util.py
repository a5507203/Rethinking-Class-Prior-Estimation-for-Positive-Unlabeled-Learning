import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Subset
from util.torch_dataset import DatasetArray


def count_class(dataset):
    loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            num_workers=1,
        )
    _,labels = iter(loader).next()
    labels = labels.numpy()
    unique_elements, counts_elements = np.unique(labels,return_counts=True)
    return unique_elements, counts_elements


def sample_summary(tr_unlabeled_sample, positive_label=1):

    unique_elements, counts_elements =count_class(DatasetArray(tr_unlabeled_sample))
    class_prior = counts_elements[positive_label]/len(tr_unlabeled_sample)

    return class_prior


import numpy as np
from scipy.stats import multivariate_normal



# def reducible_dataset(means =[0,1], variances=[1,1], negative_class_prior = 0.5, dim = 10, sample_size = 600000, remove_anchor= True):
#     data, targets, posterior = gaussian_generator_ind(means = means, variances=variances, negative_class_prior = negative_class_prior, dim = dim, sample_size = sample_size)
#     if(remove_anchor):
#         idxs = np.where((posterior < 0.998) & (posterior > 0.002))
#         posterior = posterior[idxs]
#         data = data[idxs]
#         targets = targets[idxs]
#     return data, targets

def reducible_dataset(means =[0,1], variances=[1,1], negative_class_prior = 0.5, dim = 10, sample_size = 600000, remove_anchor= True):
    data, targets, posterior = gaussian_generator_ind(means = means, variances=variances, negative_class_prior = negative_class_prior, dim = dim, sample_size = sample_size)
    if(remove_anchor):
        s = cus_sort(posterior)
        idxs = get_anchor_index(s) 
    
        all_idxs = [ i for i in range(len(data))]
        idxs = list(set(all_idxs)-set(idxs))


        posterior = posterior[idxs]
        data = data[idxs]
        targets = targets[idxs]
    return data, targets


def cus_sort(l):
    d = {i:l[i] for i in range(len(l))}
    s = [(k,d[k]) for k in sorted(d, key=d.get, reverse=False)]
    return s

def get_anchor_index(index_p_list,delete_frac=0.02):

    n = len(index_p_list)
    num_anchors = int(n*delete_frac)
    min_f_list = []
    max_f_list = []
    for (idx, p) in index_p_list:
        if(len(min_f_list)<num_anchors):
            min_f_list.append(idx)
        else:
            break
    for (idx, p) in reversed(index_p_list):
        if(len(max_f_list)<num_anchors):
            max_f_list.append(idx)
        else:
            break
    return min_f_list + max_f_list



def gaussian_generator_ind(means =[0,1], variances=[1,1], negative_class_prior = 0.5, dim = 10, sample_size = 600000):

    data = []
    labels = []
    neg_sample_size = int(sample_size*negative_class_prior)
    pos_sample_size = sample_size - neg_sample_size
    mn_neg = multivariate_normal(mean=[means[0]]*dim, cov=np.eye(dim)*variances[0])
    mn_pos = multivariate_normal(mean=[means[1]]*dim, cov=np.eye(dim)*variances[1])
    negative_labels = np.array([-1]*neg_sample_size)
    postive_labels = np.array([1]*pos_sample_size)
    neg_data = mn_neg.rvs(size=neg_sample_size)
    pos_data = mn_pos.rvs(size=pos_sample_size)
    data = np.concatenate((neg_data, pos_data))
    targets = np.concatenate((negative_labels,postive_labels))
    posterior = get_posterior(data, mn_neg, mn_pos)

    return data, targets, posterior




def get_posterior(x,mn_neg,mn_pos):

    neg_density = mn_neg.pdf(x)
    pos_density = mn_pos.pdf(x)
    x_density = neg_density+pos_density
    neg_post = neg_density/x_density
    pos_post = pos_density/x_density
    dist = neg_post
    return dist


reducible_dataset()
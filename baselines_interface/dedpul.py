from baselines_python.dedpul import *
import numpy as np

import warnings
warnings.filterwarnings('ignore')

def run_dedpul(x,x1):
    target_pu = [1]*len(x)+[0]*len(x1)
    target_pu = np.array(target_pu)
    data = x+x1
    data = np.array(data)
    test_alpha = 2
    try:
        test_alpha, poster = estimate_poster_cv(data, target_pu, estimator='dedpul', alpha=None,
                                                estimate_poster_options={'disp': False, 'alpha_as_mean_poster': True},
                                                estimate_diff_options={
                                                    'MT': False, 'MT_coef': 0.25, 'decay_MT_coef': False, 'tune': False,
                                                    'bw_mix': 0.05, 'bw_pos': 0.1, 'threshold': 'mid', 
                                                    'n_gauss_mix': 20, 'n_gauss_pos': 10,
                                                    'bins_mix': 20, 'bins_pos': 20, 'k_neighbours': None,},
                                                estimate_preds_cv_options={'bn': True, 'l2': 1e-4,
                                                    'cv': 5, 'n_networks': 1, 'lr': 1e-3, 'hid_dim': 128, 'n_hid_layers': 1,
                                                },
                                                train_nn_options={
                                                    'n_epochs': 250, 'batch_size': 64, 'loss_function': 'log',
                                                    'n_batches': None, 'n_early_stop': 7, 'disp': False,
                                                }
                                            )
    except:
        print('fail to converge')


    return 1-test_alpha

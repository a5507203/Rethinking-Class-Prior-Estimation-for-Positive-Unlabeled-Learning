import numpy as np
from baselines_python.rankpruning import RankPruning
from sklearn.linear_model import LogisticRegression
import warnings


def run_rankpruning(x,x1):
    target_pu = [1]*len(x)+[0]*len(x1)
    target_pu = np.array(target_pu)
    data = x+x1
    data = np.array(data)
    rp = RankPruning(clf = LogisticRegression())
    rp.fit(data, target_pu)
    return rp.pi1

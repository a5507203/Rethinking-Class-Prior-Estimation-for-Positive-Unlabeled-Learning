from baselines_python.kmpm import *
import numpy as np

def run_kmpms(X_mixture, X_component):
    
    X_component = np.array(X_component).astype(np.double)
    X_mixture = np.array(X_mixture).astype(np.double)

    (KM1,KM2)=wrapper(X_mixture,X_component)

    return KM1, KM2


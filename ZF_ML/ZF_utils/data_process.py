#  __        __  _____  _____ 
#  \ \      / / |__  / |  ___|
#   \ \ /\ / /    / /  | |_   
#    \ V  V /    / /_  |  _|  
#     \_/\_/    /____| |_|          2023-12-1
#                                    
#   To preprocess or postprocess the data 
#    

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc
from numpy import interp
import itertools
import numpy as np
import os
import sys
import torch.distributed as dist
import time
try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib import colors
    matplotlib.use('Agg')
except:
    print('please install matploltib in order to make plots')
    plt = False

###########################################################################################################################################################################
    
PID2FLOAT_MAP = {22: 0,
                 211: .1, -211: .2, 
                 321: .3, -321: .4, 
                 130: .5, 
                 2112: .6, -2112: .7, 
                 2212: .8, -2212: .9, 
                 11: 1.0, -11: 1.1,
                 13: 1.2, -13: 1.3}

def remap_pids(events, pid_i=0, error_on_unknown=True):
    """Remaps PDG id numbers to small floats for use in a neural network.
    `events` are modified in place and nothing is returned.

    **Arguments**

    - **events** : _numpy.ndarray_
        - The events as an array of arrays of particles.
    - **pid_i** : _int_
        - The column index corresponding to pid information in an event.
    - **error_on_unknown** : _bool_
        - Controls whether a `KeyError` is raised if an unknown PDG ID is
        encountered. If `False`, unknown PDG IDs will map to zero.
    """

    if events.ndim == 3:
        pids = events[:,pid_i,:].astype(int).reshape((events.shape[0]*events.shape[2]))
        if error_on_unknown:
            events[:,pid_i,:] = np.asarray([PID2FLOAT_MAP[pid]
                                            for pid in pids]).reshape(events.shape[0],events.shape[2])
        else:
            events[:,pid_i,:] = np.asarray([PID2FLOAT_MAP.get(pid, 0)
                                            for pid in pids]).reshape(events.shape[0],events.shape[2])
    else:
        print('no remap_pid')

###########################################################################################################################################################################
















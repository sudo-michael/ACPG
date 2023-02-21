import numpy as np
import scipy.stats
import argparse

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = a.shape[1]
    m = np.mean(a, axis=0)
    if a.shape[0] > 1:
        se = scipy.stats.sem(a, axis=0)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return m, h
    else:
        return m, [0] * m.shape[0]

def dir_builder(args, parent_dir):
    print(args)
    d = vars(args)
    for key, val in d.items():
        parent_dir += key + "_" + str(val)
    return parent_dir


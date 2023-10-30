import numpy as np
import scipy.stats

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
    for key, val in args.items():
        if key in ["num_iterations", "run", "d", "eta", "init_c", "critic_max_num_iterations", "critic_max_lr", "critic_stop_trs", "actor_max_num_iterations", "actor_max_lr", "actor_stop_trs", 'c_in_stepsize']:
            parent_dir += key + "_" + str(val)
    return parent_dir

def fill_array(lst, num_iterations, t):
    last_val = lst[-1]
    desired_length =  num_iterations - t
    lst += [last_val] * (desired_length)
    return lst


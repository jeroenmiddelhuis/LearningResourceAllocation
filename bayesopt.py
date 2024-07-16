import time
# from bayesian_optimization import BayesianOptimization
# Supress NaN warnings
import warnings
warnings.filterwarnings("ignore", category =RuntimeWarning)
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'C:\Users\user\workspace\scikit-optimize')
sys.path.append(r'C:\Users\user\workspace\scikit-optimize\skopt')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/scikit-optimize/skopt')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/scikit-optimize')
import pickle as pkl

import os
import pandas as pd

from train_SVFA import aggregate_sims, simulate_competition
from skopt.space import Real, Integer
from skopt.utils import use_named_args

import numpy as np

import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize


def main():

    # Bounded region of parameter space

    cur_time = int(time.time())
    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed + 2)

    arrival_rate =  0.45

    A_vals = [1.209298, 4.898732, 0.496618, 2.658753, 2.234656, 12.339795, 2000]


    space = [Real(A_vals[0]-0.5, A_vals[0]+0.5, name='a1'),
             Real(A_vals[1]-0.5, A_vals[1]+0.5, name='a2'),
             Real(A_vals[2]-0.5, A_vals[2]+0.5, name='a3'),
             Real(A_vals[3]-0.5, A_vals[3]+0.5, name='a4'),
             Real(A_vals[4]-0.5, A_vals[4]+0.5, name='a5'),
             Real(A_vals[5]-0.5, A_vals[5]+0.5, name='a6'),
             Real(A_vals[6]-0.5, A_vals[6]+0.5, name='a7')]

    res = gp_minimize(aggregate_sims,  # the function to minimize
                      space,  # the bounds on each dimension of x
                      acq_func="EI",  # the acquisition function
                      n_calls=15,  # the number of evaluations of f
                      n_random_starts=3,  # the number of random initialization points
                      noise=0.1 ** 2,  # the noise level (optional)
                      random_state=seed)


    model_num = np.random.randint(0, 100000)

    pkl.dump((res.func_vals, res.x_iters), open('./eliran_results/' + str(model_num) + 'arrival_rate_' +str(str(arrival_rate)) +'_low_utilization.pkl', 'wb'))



if __name__ == "__main__":


    main()

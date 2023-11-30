import time
import sys
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/scikit-optimize')
sys.path.append('/home/eliransc/projects/def-dkrass/eliransc/scikit-optimize\skopt')
# from bayesian_optimization import BayesianOptimization
# Supress NaN warnings
import warnings
warnings.filterwarnings("ignore", category =RuntimeWarning)
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(r'C:\Users\user\workspace\scikit-optimize')
sys.path.append(r'C:\Users\user\workspace\scikit-optimize\skopt')

import pickle as pkl

import os
import pandas as pd

from train_SVFA import aggregate_sims, simulate_competition
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from tqdm import tqdm
import numpy as np

import matplotlib.pyplot as plt
from skopt.plots import plot_gaussian_process
from skopt import gp_minimize
# from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
def main():

    # Bounded region of parameter space
    systems = ["low_utilization", "high_utilization", "slow_server", "down_stream", "n_system", "parallel", "complete", "complete_reversed",  "complete_parallel" ]

    a8 = np.random.randint(len(systems))

    pbounds = {'a1': (0, 20),
               'a2': (0, 20),
               'a3': (0, 20),
               'a4': (0, 20),
               'a5': (0, 20),
               'a6': (0, 20),
               'a7': (50, 550),
               'a8': (a8, a8+0.0001)}

    optimizer = BayesianOptimization(
        f=aggregate_sims,
        pbounds=pbounds,
        verbose=2,
        random_state=10,
        allow_duplicate_points=True,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    optimizer.maximize(
                init_points=5,
                n_iter=120,
            )

    pkl.dump(optimizer, open('optimizier_complete.pkl', 'wb'))




if __name__ == "__main__":


    main()

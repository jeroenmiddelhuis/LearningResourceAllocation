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
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt import BayesianOptimization
from bayes_opt.util import load_logs
def main():

    # Bounded region of parameter space

    pbounds = {'a1': (0, 20),
               'a2': (0, 20),
               'a3': (0, 20),
               'a4': (0, 20),
               'a5': (0, 20),
               'a6': (0, 20),
               'a7': (0, 20)}

    optimizer = BayesianOptimization(
        f=aggregate_sims,
        pbounds=pbounds,
        verbose=2,
        random_state=2,
        allow_duplicate_points=True,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
    )

    for ind in tqdm(range(2)):


        if os.path.exists('./logs.json'):

            load_logs(optimizer, logs=["./logs.json"]);
            vals = [res for i, res in enumerate(optimizer.res)]
            print(len(vals))
            print('num_ites is 1')


            print('Start optimizing')
            optimizer.maximize(
                init_points=0,
                n_iter=1,
            )

            logger = JSONLogger(path="./logs.json")
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

            load_logs(optimizer, logs=["./logs.json"]);
            vals = [res for i, res in enumerate(optimizer.res)]
            print(len(vals))

        else:
            print('num_ites is 7')
            logger = JSONLogger(path="./logs.json")
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

            print('Start optimizing')
            optimizer.maximize(
                init_points=2,
                n_iter=5,
            )


        print('Finish')
        vals = [res for i, res in enumerate(optimizer.res)]

        pkl.dump(vals, open('res_complete_all.pkl', 'wb'))





    # space = [Real(0, 20, name='a1'),
    #          Real(0, 20, name='a2'),
    #          Real(0, 20, name='a3'),
    #          Real(0, 20, name='a4'),
    #          Real(0, 20, name='a5'),
    #          Real(0, 20, name='a6'),
    #          Real(0, 20, name='a7')]
    #
    # res = gp_minimize(aggregate_sims,  # the function to minimize
    #                   space,  # the bounds on each dimension of x
    #                   acq_func="EI",  # the acquisition function
    #                   n_calls=1,  # the number of evaluations of f
    #                   n_random_starts=1,  # the number of random initialization points
    #                   noise=0.1 ** 2,  # the noise level (optional)
    #                   random_state=1234)


    # model_num = np.random.randint(0, 100000)





if __name__ == "__main__":


    main()

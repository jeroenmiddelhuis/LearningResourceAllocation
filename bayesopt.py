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


running_time = 5000
import numpy as np
import pickle as pkl
import os
from simulator import Simulator
from planners import  ShortestProcessingTime, PPOPlanner, Bayes_planner # GreedyPlanner, ShortestProcessingTime, DedicatedResourcePlanner,
import pandas as pd


#'complete_parallel', 'complete_reversed', 'complete',


listis = [ 'parallel', 'n_system', 'down_stream', 'slow_server', 'high_utilization', 'low_utilization']
configtype = np.random.choice(listis)
# configtype = 'low_utilization'

arrival_list = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
arrival_rate = np.random.choice(listis)
arrival_rate = 0.5

print(arrival_rate, configtype)
# You can build your bayesian optimization model around this framework:
# -Determine parameters for the planner
# -Run the simulation with the planner
# -Get the total_reward
def simulate_competition1(A):
    print(configtype)
    simulator_fake = Simulator(running_time, ShortestProcessingTime(), config_type=configtype, reward_function='AUC')
    a1 = A[0]
    a2 = A[1]
    a3 = A[2]
    a4 = A[3]
    a5 = A[4]
    a6 = A[5]
    a7 = A[6]
    # print(a1, a2, a3, a4, a5, a6, a7)
    planner = Bayes_planner(a1, a2, a3, a4, a5, a6,a7,simulator_fake)  # ShortestProcessingTime() # Insert your planner here, input can be the parameters of your model
    log_dir = os.getcwd() + '\\results_test'
    # The config types dictates the system
    simulator = Simulator(running_time, planner, config_type=configtype, reward_function='AUC', arrival_rate=arrival_rate, write_to=log_dir)
    # You can access some proporties from the simulation:
    # simulator.resource_pools: for each tasks 1) the resources that can process it and 2) the mean and variance of the processing time of that assignment
    # simulator.mean_interarrival_time
    # simulator.task_types
    # simulator.resources
    # simulator.initial_task_dist
    # simulator.resource_pools
    # simulator.transitions

    # You should want to optimize the total_reward, this is related to the mean cycle time, however the total reward also includes uncompleted casese
    # Total reward = total cycle time
    nr_uncompleted_cases, total_reward, CT_mean, CT_std, utilisation = simulator.run()
    # print('CT_mean: ', CT_mean)

    return CT_mean


def aggregate_sims1(A):


    import time
    cur_time = int(time.time())
    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed+2)
    model_num = np.random.randint(0, 1000)
    tot_res = []
    print(A)
    for ind in range(20):
        res = simulate_competition1(A)
        print(res)
        # print(res)
        tot_res.append(res)

    print(np.array(tot_res).mean())
    pkl.dump((tot_res,A), open('./eliran_results/'+configtype + '_arrival_rate_' +str(arrival_rate)+'_model_num_' + str(model_num) + '.pkl', 'wb'))
    return np.array(tot_res).mean()  #tot_res #


def main():

    # Bounded region of parameter space

    cur_time = int(time.time())
    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed + 2)

    print(arrival_rate)

    if configtype == 'low_utilization':
        A_vals = [1.209298, 4.898732, 0.496618, 2.658753, 2.234656, 12.339795, 2000]
    elif configtype == 'high_utilization':
        A_vals =  [1.321091, 4.556206,  0.790609,0.000000, 3.051695,    15.739693, 2000]
    elif configtype == 'down_stream':
        A_vals = [1.655077, 4.013198,0.491260, 0.279967, 2.213313, 7.287871, 20000]
    elif configtype == 'n_system':
        A_vals = [0.000000,	5.000000,	1.584558,	0.000000,	0.000000	, 7.0, 2000]
    elif configtype == 'slow':
        A_vals = [5.000000, 3.767883, 0.0, 1.645102, 0.000000, 20.000000, 20000]
    elif configtype == 'complete_all':
        A_vals= [0.337376 ,   4.110818,    1.167821,    0.862430 ,   0.474204 ,   19.111311, 20000]


    space = [Real(A_vals[0]-0.5, A_vals[0]+0.5, name='a1'),
             Real(A_vals[1]-0.5, A_vals[1]+0.5, name='a2'),
             Real(A_vals[2]-0.5, A_vals[2]+0.5, name='a3'),
             Real(A_vals[3]-0.5, A_vals[3]+0.5, name='a4'),
             Real(A_vals[4]-0.5, A_vals[4]+0.5, name='a5'),
             Real(A_vals[5]-0.5, A_vals[5]+0.5, name='a6'),
             Real(A_vals[6]-0.5, A_vals[6]+0.5, name='a7')]

    res = gp_minimize(aggregate_sims1,  # the function to minimize
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

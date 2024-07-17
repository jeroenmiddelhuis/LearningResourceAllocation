from simulator import Simulator
from planners import  ShortestProcessingTime, PPOPlanner, Bayes_planner # GreedyPlanner, ShortestProcessingTime, DedicatedResourcePlanner,
import pandas as pd

running_time = 5000
import numpy as np
import pickle as pkl
import os

# You can build your bayesian optimization model around this framework:
# -Determine parameters for the planner
# -Run the simulation with the planner
# -Get the total_reward
def simulate_competition(A):

    simulator_fake = Simulator(running_time, ShortestProcessingTime(), config_type='low_utilization', reward_function='AUC')
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
    simulator = Simulator(running_time, planner, config_type='low_utilization', reward_function='AUC', arrival_rate=0.45, write_to=log_dir)
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


def aggregate_sims(A, bb):

    print(bb)
    import time
    cur_time = int(time.time())
    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed+2)
    model_num = np.random.randint(0, 1000)
    tot_res = []
    print(A)
    for ind in range(20):
        res = simulate_competition(A)
        print(res)
        # print(res)
        tot_res.append(res)

    print(np.array(tot_res).mean())
    pkl.dump((tot_res, A), open('./eliran_results/low_utilization' + '_arrival_rate_' +str(0.45)+'_model_num_' + str(model_num) + '.pkl', 'wb'))
    return np.array(tot_res).mean()  #tot_res #


def main():

    # high_utilization = [1.321091, 4.556206,  0.790609,0.000000, 3.051695,    15.739693]

    low_utilization = [1.209298, 4.898732,	0.496618, 2.658753,	2.234656,12.339795, 20]

    # n_system = [0.000000,	5.000000,	1.584558,	0.000000,	0.000000	, 7.0]
    #slow = [5.000000,	3.767883,	0.0,	1.645102,	0.000000,	20.000000]

    #down_stream = [1.655077, 4.013198,0.491260, 0.279967, 2.213313, 7.287871]

    # complete_all = [0.337376	4.110818	1.167821	0.862430	0.474204	19.111311]


    # simulate_competition(A)

    get_results = aggregate_sims([1.209298, 4.898732,	0.496618, 2.658753,	2.234656,12.339795, 2000])

    import time
    cur_time = int(time.time())
    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed)
    model_num = np.random.randint(0, 1000)


    pkl.dump(get_results, open(str(model_num) + '_final_complete.pkl', 'wb'))


if __name__ == "__main__":
    main()
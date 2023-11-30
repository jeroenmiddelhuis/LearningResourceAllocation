from simulator import Simulator
from planners import GreedyPlanner, ShortestProcessingTime, DedicatedResourcePlanner, PPOPlanner, Bayes_planner
import pandas as pd

running_time = 5000
import numpy as np
import pickle as pkl
import os

# You can build your bayesian optimization model around this framework:
# -Determine parameters for the planner
# -Run the simulation with the planner
# -Get the total_reward
def simulate_competition(A, system):

    simulator_fake = Simulator(running_time, ShortestProcessingTime(), config_type=system, reward_function='AUC')
    a1 = A[0]
    a2 = A[1]
    a3 = A[2]
    a4 = A[3]
    a5 = A[4]
    a6 = A[5]
    a7 = A[6]


    # a1 = 0.0
    # a2 = 0.0
    # a3 = 20.0
    # a4 = 0.0
    # a5 = 0.0
    # a6 = 20.0
    # a7 = 144.6


    # print(a1, a2, a3, a4, a5, a6, a7)
    planner = Bayes_planner(a1, a2, a3, a4, a5, a6,a7, simulator_fake)  # ShortestProcessingTime() # Insert your planner here, input can be the parameters of your model
    # planner1 = ShortestProcessingTime()

    # The config types dictates the system
    simulator = Simulator(running_time, planner, config_type=system, reward_function='AUC')
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
    nr_uncompleted_cases, total_reward, CT_mean, CT_std, = simulator.run()
    # print('stop')

    return CT_mean


def aggregate_sims(a1, a2, a3, a4, a5, a6, a7, a8):
    import time

    system_ind = int(a8)
    systems = ["low_utilization", "high_utilization", "slow_server", "down_stream", "n_system", "parallel", "complete", "complete_reversed",  "complete_parallel" ]

    system = systems[system_ind]
    print(system)

    A = np.array([a1, a2, a3, a4, a5, a6, a7])
    cur_time = int(time.time())
    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed)
    model_num = np.random.randint(0, 1000)
    tot_res = []

    # tot_res = [12.699543724273608, 13.163023643890151,  14.746113842371038, 11.139423899073817,  19.815261349809344,
    #           15.706213398983703,  11.491194604229205,  19.23838095500618,  16.316676161670987,  13.462364343565945,
    #           12.85587584268216, 15.06492276522547,  15.000369983396416,   12.28529415912139,  14.23442144776856,
    #           11.399387500794504,  19.649579691039293,  17.004019925853694,  12.952197298510864, 12.752720812319906]

    for ind in range(100):
        res = simulate_competition(A, system)
        print(res)
        tot_res.append(res)

    file_num = np.random.randint(1, 10000)
    file_name = system + '_' +str(file_num) + '_' +str(np.array(tot_res).mean()) +  'with_weights.pkl'

    pkl.dump((tot_res, (a1, a2, a3, a4, a5, a6, a7)), open(os.path.join('./results/', file_name), 'wb'))
        # pkl.dump(tot_res, open('run_500_res_simple_linear_high_utilisation3_' + str(model_num) + '.pkl', 'wb'))

    # print(res)
    # return (-np.array(tot_res).mean(),tot_res)

    return -np.array(tot_res).mean()


def main():

    # high_utilization = [1.321091, 4.556206,  0.790609,0.000000, 3.051695,    15.739693]

    #low_utilization = [1.209298, 4.898732,	0.496618, 2.658753,	2.234656,12.339795]

    # n_system = [0.000000,	5.000000,	1.584558,	0.000000,	0.000000	, 7.0]
    #slow = [5.000000,	3.767883,	0.0,	1.645102,	0.000000,	20.000000]

    #down_stream = [1.655077, 4.013198,0.491260, 0.279967, 2.213313, 7.287871]

    # complete_all = [0.337376	4.110818	1.167821	0.862430	0.474204	19.111311]


    # simulate_competition(A)

    a1 = 0.0
    a2 = 0.0
    a3 = 20.0
    a4 = 0.0
    a5 = 0.0
    a6 = 20.0
    a7 = 144.6


    get_results, tot_res = aggregate_sims(a1, a2, a3, a4, a5, a6, a7)

    import time
    cur_time = int(time.time())
    seed = cur_time + np.random.randint(1, 1000)  # + len(os.listdir(data_path)) +
    np.random.seed(seed)
    model_num = np.random.randint(0, 1000)


    pkl.dump(tot_res, open(str(model_num) + '_final_complete.pkl', 'wb'))


if __name__ == "__main__":
    main()
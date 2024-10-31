from simulator import Simulator
from planners import PPOPlanner
from planners_benchmark import ShortestProcessingTime, FIFO, Random#, DDQNPlanner, ParkSong, GreedyPlanner
from time import time
import numpy as np
import os
import sys


running_time = 5000
write = False
# Original main
def simulate_competition(config_type, planner, arrival_rate):
    results = []
    times = []
    #log_dir=os.getcwd() + f'/results_{planner}/'
    log_dir = f'./results_revision/results_{planner}/'
    os.makedirs(log_dir, exist_ok=True)
    #log_dir=None
    for i in range(100):
        if i % 5 == 0:
            print(i)

        # To change the planner, uncomment one of the following lines. The RLRAM planner has its own file: __main__RLRAM.py
        #planner = DedicatedResourcePlanner()
        if planner == 'SPT': planner = ShortestProcessingTime()
        if planner == 'FIFO': planner = FIFO()
        if planner == 'random': planner = Random()
        #planner = ParkSong()
        #planner = DDQNPlanner(config_type)
        if planner == 'PPO': planner = PPOPlanner(os.getcwd() + '/tmp/' + f"{config_type}_{arrival_rate}" + "/best_model.zip")

        if write == False:
            log_dir = None
        simulator = Simulator(running_time, planner, config_type=f'{config_type}', reward_function='cycle_time', write_to=log_dir, arrival_rate=arrival_rate)

        if type(planner) == PPOPlanner:# or type(planner) == ParkSong or type(planner) == DDQNPlanner:
            planner.linkSimulator(simulator)
            # if type(planner) == DDQNPlanner:
            #     planner.create_model()

        if write == True and i == 0:
            resource_str = ''
            for resource in simulator.resources:
                resource_str += resource + ','
            #with open(os.path.join(sys.path[0], f'{simulator.write_to}{planner}_results_{simulator.config_type}.txt'), "w") as file:
            with open(simulator.write_to + f'{planner}_{simulator.config_type}_{arrival_rate}.txt', "w") as file:
                # Writing data to a file
                file.write(f"uncompleted_cases,{resource_str}total_reward,mean_cycle_time,std_cycle_time\n")

        result = simulator.run()
        results.append(result)  


def main():
    config_type = 'low_utilization'# sys.argv[1]
    arrival_rate = 'pattern'#float(sys.argv[2]) if sys.argv[2] != 'pattern' else sys.argv[2]
    planner = 'SPT'#sys.argv[3]
    #for model_name in [config_type]:#['n_system', 'down_stream', 'high_utilization', 'low_utilization', 'slow_server', 'parallel']:#,
    simulate_competition(config_type, planner, arrival_rate)#['n_system', 'down_stream', 'high_utilization', 'low_utilization', 'slow_server', 'parallel', 
    print('\n')

if __name__ == "__main__":
    main()



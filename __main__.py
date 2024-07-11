from simulator import Simulator
from planners import PPOPlanner
from planners_benchmark import ParkSong, GreedyPlanner, ShortestProcessingTime, FIFO, Random, DDQNPlanner
from time import time
import numpy as np
import os
import sys


running_time = 5000
arrival_rate = 'pattern'
write = False
# Original main
def simulate_competition(model_name):
    results = []
    times = []
    log_dir=os.getcwd() +'\\results_test'
    os.makedirs(log_dir, exist_ok=True)
    #log_dir=None
    for i in range(100):
        if i % 5 == 0:
            print(i)

        # To change the planner, uncomment one of the following lines. The RLRAM planner has its own file: __main__RLRAM.py
        #planner = DedicatedResourcePlanner()
        planner = ShortestProcessingTime()
        #planner = FIFO()
        #planner = Random()       
        #planner = ParkSong()
        #planner = DDQNPlanner(model_name)
        #planner = PPOPlanner(os.getcwd() + "\\tmp\\" + f"{model_name}_20000000_25600" + "\\best_model.zip")
                             
        if write == False:
            log_dir = None
        simulator = Simulator(running_time, planner, config_type=f'{model_name}', reward_function='cycle_time', write_to=log_dir, arrival_rate=arrival_rate)

        if type(planner) == PPOPlanner or type(planner) == ParkSong or type(planner) == DDQNPlanner:
            planner.linkSimulator(simulator)
            if type(planner) == DDQNPlanner:
                planner.create_model()

        if write == True and i == 0:
            resource_str = ''
            for resource in simulator.resources:
                resource_str += resource + ','
            #with open(os.path.join(sys.path[0], f'{simulator.write_to}{planner}_results_{simulator.config_type}.txt'), "w") as file:
            with open(simulator.write_to + f'\\{planner}_{simulator.config_type}.txt', "w") as file:
                # Writing data to a file
                file.write(f"uncompleted_cases,{resource_str}total_reward,mean_cycle_time,std_cycle_time\n")

        result = simulator.run()
        results.append(result)  


def main():
    for model_name in ['high_utilization']:#['n_system', 'down_stream', 'high_utilization', 'low_utilization', 'slow_server', 'parallel']:#,
        simulate_competition(model_name)#['n_system', 'down_stream', 'high_utilization', 'low_utilization', 'slow_server', 'parallel', 
        print('\n')

if __name__ == "__main__":
    main()



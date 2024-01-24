from simulator import Simulator
from planners import GreedyPlanner, ShortestProcessingTime, FIFO, Random, PPOPlanner
from time import time
import numpy as np
import os
import sys


running_time = 3000
write = True
interval = True # True or None

# Original main
def simulate_competition(model_name):
    results = []
    times = []
    log_dir= './results'

    os.makedirs(log_dir, exist_ok=True)
    #log_dir=None
    for i in range(100):
        if i % 5 == 0:
            print(i)
        #planner = DedicatedResourcePlanner()
        #planner = ShortestProcessingTime()
        #planner = FIFO()
        #planner = Random()       
        model_folder = f"{model_name}_10000000_0.1"
        if interval == True:
            check_interval = float(model_folder.split('_')[-1])
        else:
            check_interval = None
        planner = PPOPlanner("./tmp/slow_server_10000000_0.1/best_model.zip", check_interval=check_interval)
        
                             
        if write == False:
            log_dir = None
        simulator = Simulator(running_time, planner, config_type=f'{model_name}', reward_function='AUC', check_interval=check_interval, write_to=log_dir)

        if type(planner) == PPOPlanner:
            planner.linkSimulator(simulator)

        if write == True and i == 0:
            resource_str = ''
            for resource in simulator.resources:
                resource_str += resource + ','
            #with open(os.path.join(sys.path[0], f'{simulator.write_to}{planner}_results_{simulator.config_type}.txt'), "w") as file:
            with open(simulator.write_to + f'\\{planner}_interval_{simulator.config_type}.txt', "w") as file:
                # Writing data to a file
                file.write(f"uncompleted_cases,{resource_str}total_reward,mean_cycle_time,std_cycle_time\n")



        t1 = time()
        result = simulator.run()
        #print(f'Simulation finished in {time()-t1} seconds')
        #print(result)
        times.append(time()-t1)
        results.append(result)  
        #print('\n') 

    # with open(f'{simulator.write_to}{planner}_results_{simulator.config_type}.txt', "w") as out_file:
    #     for i in range(len(results)):
    #         out_file.write(f'{times[i]},{results[i]}\n')

def main():
    for model_name in ['slow_server']:#['complete','complete_reversed', 'complete_parallel', 'n_system', 'down_stream', 'high_utilization', 'low_utilization', 'slow_server', 'parallel']:
        simulate_competition(model_name)#['n_system', 'down_stream', 'high_utilization', 'low_utilization', 'slow_server', 'parallel', 
        print('\n')

if __name__ == "__main__":
    main()



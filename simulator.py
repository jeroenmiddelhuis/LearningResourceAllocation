import numpy as np
import os
import sys
from enum import Enum, auto
from collections import deque
import random
import json
import pickle

class Event:
    def __init__(self, event_type, moment, task=None, resource=None):
        self.event_type = event_type
        self.moment = moment              
        self.task = task
        self.resource = resource
        self.cycle_time = 0

    def __lt__(self, other):
        return self.moment < other.moment


class EventType(Enum):
    CASE_ARRIVAL = auto()
    CASE_DEPARTURE = auto()
    TASK_START = auto()
    TASK_COMPLETE = auto()
    PLAN_TASKS = auto()
    RETURN_REWARD = auto()


class Case:
    _id = 0

    def __init__(self, moment):
        self.id = Case._id
        Case._id += 1
        self.arrival_time = moment
        self.departure_time = None
        self.tasks = []
        self.uncompleted_tasks = []
        self.completed_tasks = []        

    def add_task(self, task):
        self.uncompleted_tasks.append(task)

    def complete_task(self, task):
        self.uncompleted_tasks.remove(task)
        self.completed_tasks.append(task)


class Task:
    _id = 0
    _parallel_id = 0

    def __init__(self, moment, case_id, task_type, parallel=False):
        self.id = Task._id
        Task._id += 1
        self.start_time = moment
        self.case_id = case_id
        self.task_type = task_type
        self.parallel = parallel
        self.nr_parallel = None
        self.parallel_id = None
        if self.parallel == True:
            self.parallel_id = Task._parallel_id


class Simulator:    
    def __init__(self, running_time, planner, config_type, reward_function='cycle_time', write_to=None, arrival_rate=0.5):
        self.config_type = config_type
        with open(os.path.join(sys.path[0], "config.txt"), "r") as f:
            data = f.read()

        config = json.loads(data)        
        config = config[config_type]
            
        self.running_time = running_time
        self.status = "RUNNING"
        self.debug = False

        self.now = 0
        self.events = []
        
        self.sumx = 0
        self.sumxx = 0
        self.sumw = 0

        #self.cv = cv # for Gamma distribution

        self.available_tasks = []
        self.waiting_parallel_tasks = []
        self.reserved_tasks = []
        self.completed_tasks = []

        self.available_resources = []        
        self.reserved_resources = []
        
        self.uncompleted_cases = {}
        self.completed_cases = {}

        self.arrival_rate = arrival_rate
        self.arrival_pattern = [(73, 0.01), (85, 0.5), (120, 1.15), (135, 0.8), (168, 1.15), (175, 0.4), (209, 0.73), (250, 0.01)]#[(10, 0.01), (35, 0.2), (90, 0.75), (115, 0.5), (160, 0.75), (175, 0.25), (225, 0.47), (240, 0.01)]
        #self.mean_interarrival_time = 1/self.arrival_rate#1.81818181#config['mean_interarrival_time']*1.81818181
        self.task_types = list(config['task_types'])
        self.resources = list(config['resources'])    
        self.resource_pools = config['resource_pools']
        self.transitions = config['transitions']

        # KDE plot based on the BPI12 dataset and parameters to scale the rate to lambda=0.5
        with open('kde_model.pkl', 'rb') as file:
            self.kde = pickle.load(file)
        self.kde_max_rate = 2.5164926873799365
        self.kde_mean = 1.4284215305048689
        self.kde_max_rate_scaled = 0.8808648685414641



        # self.task_types = sorted(list(self.task_types))
        # self.resources = sorted(list(set(np.hstack(list(self.resources))))) 

        self.planner = planner 
        if self.planner != None: self.planner.resource_pools = self.resource_pools

        self.resource_total_busy_time = {resource:0 for resource in self.resources}
        self.resource_last_start = {resource:0 for resource in self.resources}

        # Reinforcement learning parameters
        self.input = [resource + '_availability' for resource in self.resources] + \
                     [resource + '_to_task' for resource in self.resources] + \
                     [task_type for task_type in self.task_types if task_type != 'Start'] 
        self.output = [(resource, task) for task in self.task_types[1:] for resource in self.resources if resource in self.resource_pools[task]] + ['Postpone']
        #print(len(self.input), len(self.output))
        self.reward_function = reward_function
        self.write_to = write_to
        self.current_reward = 0
        self.total_reward = 0

        self.reward_case = 1
        self.reward_task_start = 0
        self.reward_task_complete = {task:i+1 for i, task in enumerate(self.task_types)}

        self.last_reward_moment = 0
        self.last_mask = []
        self.bpo_state = []

        # init simulation
        self.available_resources = [resource for resource in self.resources]
        self.events.append(Event(EventType.CASE_ARRIVAL, self.sample_interarrival_time()))


    def generate_initial_task(self, case_id):
        return self.generate_next_task(Task(self.now, case_id, 'Start'))
    
    def generate_next_task(self, current_task, predict=False):
        #print(current_task.task_type, self.transitions[current_task.task_type])
        if predict == False: case = self.uncompleted_cases[current_task.case_id]
        if sum(self.transitions[current_task.task_type]) <= 1: # Activity not in parallel
            rvs = random.random()
            prob = 0
            for i, p in enumerate(self.transitions[current_task.task_type]):
                if i == len(self.transitions[current_task.task_type]) - 1: # If the next task is not another task, complete the case
                    next_task_type = 'Complete'
                    break
                prob += p
                if rvs <= prob:
                    next_task_type = self.task_types[i]
                    break
            if predict == False: case.add_task(next_task_type)
            return [Task(self.now, current_task.case_id, next_task_type)]
        
        else: # Task is in parallel
            if sum(self.transitions[current_task.task_type]) % 1 > 0.0001:
                raise 'Sum of transition probabilities should be less or equal to 1, or an integer (parallelism)'

            Task._parallel_id += 1
            next_tasks = []
            current_prob = 0
            rvs = random.random()
            generated_xor = False
            for i, p in enumerate(self.transitions[current_task.task_type]):
                #print(p)
                if p == 1:
                    #print(p, 'one')
                    if i == len(self.transitions[current_task.task_type]) - 1:
                        next_task_type = 'Complete'
                    else:
                        next_task_type = self.task_types[i]
                    next_tasks.append(Task(self.now, current_task.case_id, next_task_type, parallel=True))
                elif p != 0:                    
                    current_prob += p
                    #print(p, 'else', rvs, current_prob)
                    if rvs <= current_prob and generated_xor == False:
                        generated_xor = True
                        next_task_type = self.task_types[i]
                        #print('gen', next_task_type)
                        next_tasks.append(Task(self.now, current_task.case_id, next_task_type, parallel=True))

            for next_task in next_tasks:
                if predict == False: case.add_task(next_task.task_type)
                next_task.nr_parallel = len(next_tasks)
            #print([task.task_type for task in next_tasks])
            return next_tasks


    def generate_case(self):
        case = Case(self.now)
        self.uncompleted_cases[case.id] = case
        return case

    def process_assignment(self, assignment):

        if self.reward_function == 'task':
            self.current_reward += self.reward_task_start  
            self.total_reward += self.reward_task_start
        if self.reward_function == 'queue':
            self.current_reward += 1/(1 + self.now - assignment[1].start_time)
            self.total_reward += 1/(1 + self.now - assignment[1].start_time)

        if assignment[0] in self.available_resources and assignment[1] in self.available_tasks:
            self.available_resources.remove(assignment[0])
            self.available_tasks.remove(assignment[1])

            self.state = self.get_state()

            self.resource_last_start[assignment[0]] = self.now
            pt = self.sample_processing_time(assignment[0], assignment[1].task_type)
            self.events.append(Event(EventType.TASK_COMPLETE, self.now + pt, assignment[1], assignment[0]))
        self.events.sort()

    def find_bin_index(self, t):
        for index, (bin_end, _) in enumerate(self.arrival_pattern):
            if t < bin_end:
                return index
        return None  # or handle the case where t is outside the bins range

    def sample_interarrival_time(self):        
        if self.arrival_rate == 'pattern':
            next_arrival_time = self.now
            while True:
                scaled_time = 0.2 + (next_arrival_time % 250 - 0) * (0.9 - 0.2) / (250 - 0)
                rate = self.kde(scaled_time) / self.kde_mean * 0.5 # Multiply by 0.5 so that the average rate is 0.5
                #print(next_arrival_time, scaled_time, scaled_time, rate)
                next_arrival_time += random.expovariate(self.kde_max_rate_scaled)
                if rate[0]/self.kde_max_rate_scaled > random.random():
                    #print(next_arrival_time - self.now, self.now, '\n')
                    return next_arrival_time - self.now # interarrival time

            # time = self.now % self.arrival_pattern[-1][0]
            # interarrival_time = 0
            # bin_index = self.find_bin_index(time)
            # while True:
            #     rate = self.arrival_pattern[bin_index][1]
            #     sampled_time = random.expovariate(self.arrival_pattern[bin_index][1])
            #     if time + sampled_time <= self.arrival_pattern[bin_index][0]: # Arrival falls in the current bin
            #         interarrival_time += sampled_time
            #         return interarrival_time
            #     else:
            #         if bin_index != len(self.arrival_pattern)-1:                        
            #             interarrival_time += self.arrival_pattern[bin_index][0] - time
            #             time = self.arrival_pattern[bin_index][0]
            #             bin_index += 1
            #         else:
            #             bin_index = 0
            #             time = 0
        return random.expovariate(self.arrival_rate)

    def sample_processing_time(self, resource, task):
        return random.expovariate(1/self.resource_pools[task][resource][0])

    def get_state(self):
        ### Resource binary, busy time, assigned to + nr of each task
        resources_available = [1 if x in self.available_resources else 0 for x in self.resources]

        #resources_busy_time = [0 for _ in range(len(self.resources))]
        resources_assigned = [0.0 for _ in range(len(self.resources))]
        for event in self.events:
            if event.event_type == EventType.TASK_COMPLETE:
                resource_index = self.resources.index(event.resource)
                #resources_busy_time[resource_index] = self.now - event.task.start_time
                resources_assigned[resource_index] = self.task_types.index(event.task.task_type)/(len(self.task_types)-1)

        if len(self.available_tasks) > 0:
            task_types_num = [min(1.0, sum([1.0 if task.task_type == el else 0.0 for task in self.available_tasks])/100) for el in self.task_types if el != 'Start'] # len(self.available_tasks)
        else:
            task_types_num = [0.0 for el in self.task_types if el != 'Start']

        return np.array(resources_available + resources_assigned + task_types_num)

    def define_action_masks(self):
        action_masks = [True if resource in self.available_resources and task in [_task.task_type for _task in self.available_tasks] else False 
                for resource, task in self.output[:-1]] + [True]
        return action_masks

    def run(self):
        while self.now <= self.running_time:
            event = self.events.pop(0)
            self.now = event.moment
            if self.now <= self.running_time: # To prevent next event time after running time
                if event.event_type == EventType.CASE_ARRIVAL:

                    case = self.generate_case() # Automatically added to dict of uncompleted cases
                    for task in self.generate_initial_task(case.id):
                        self.available_tasks.append(task)
                    self.events.append(Event(EventType.CASE_ARRIVAL, self.now + self.sample_interarrival_time())) # Schedule new arrival
                    if len(self.available_tasks) > 0 and len(self.available_resources) > 0:
                        self.events.append(Event(EventType.PLAN_TASKS, self.now))
                    self.events.sort()

                if event.event_type == EventType.PLAN_TASKS:
                    if self.planner == None: # DRL algorithm handles processing of assignments (training and inference)
                        # there only is an assignment if there are free resources and tasks
                        if sum(self.define_action_masks()) > 1:
                            break # Return to gym environment
                    else: #at inference time, we call the plan function of the planner
                        assignments = self.planner.plan(self.available_tasks, self.available_resources, self.resource_pools)
                        for assignment in assignments:
                            self.process_assignment(assignment) # Reserves the task and resource, schedules TASK_START event

                if event.event_type == EventType.TASK_COMPLETE:
                    case = self.uncompleted_cases[event.task.case_id]
                    case.complete_task(event.task.task_type)

                    # Release resource
                    self.available_resources.append(event.resource)
                    self.resource_total_busy_time[event.resource] += self.now - self.resource_last_start[event.resource]
                    # Complete task
                    self.completed_tasks.append(event.task)

                    # All parallel
                    next_tasks = []
                    if self.config_type == 'complete_parallel':
                        if self.transitions[event.task.task_type][-1] == 1 and sum(self.transitions[event.task.task_type][:-1]) == 0:
                            # Not all parallel tasks are completed, wait for remaining
                            if sum([1 if event.task.parallel_id == task.parallel_id else 0 for task in self.waiting_parallel_tasks]) != 7 - 1: # Hard coded nr of parallel tasks
                                self.waiting_parallel_tasks.append(event.task)
                            # All parallel tasks completed, generate next event and remove tasks from waiting
                            else:# sum([1 if event.task.parallel_id == task.parallel_id else 0 for task in self.waiting_parallel_tasks]): # Hard coded nr of parallel tasks
                                # Remove all waiting tasks
                                for task in self.waiting_parallel_tasks:
                                    if task.parallel_id == event.task.parallel_id:
                                        self.waiting_parallel_tasks.remove(task)
                                # Generate next task
                                next_tasks = self.generate_next_task(event.task)
                        else:
                            next_tasks = self.generate_next_task(event.task)
                            for next_task in next_tasks:
                                next_task.parallel = True
                                next_task.parallel_id = event.task.parallel_id
                    elif self.config_type == 'dominic':
                        if event.task.task_type in ["Start", 'Task A', 'Task B']:                            
                            next_tasks = self.generate_next_task(event.task)
                        elif event.task.task_type == 'Task C':
                            if len([task for task in self.completed_tasks if task.case_id == event.task.case_id and task.task_type in ['Task D', 'Task E']]) == 2:
                                next_tasks = self.generate_next_task(event.task)
                        elif event.task.task_type == 'Task D':
                            if len([task for task in self.completed_tasks if task.case_id == event.task.case_id and task.task_type in ['Task C', 'Task E']]) == 2:
                                next_tasks = self.generate_next_task(event.task)
                        elif event.task.task_type == 'Task E':
                            if len([task for task in self.completed_tasks if task.case_id == event.task.case_id and task.task_type in ['Task C', 'Task D']]) == 2:
                                next_tasks = self.generate_next_task(event.task)
                    else:
                        # If the completed task is a parallel task and other parallel tasks are still being processed, then wait for completion
                        if event.task.parallel == True:
                            #print(event.task.task_type, event.task.case_id)
                            if sum([1 if event.task.parallel_id == task.parallel_id else 0 for task in self.waiting_parallel_tasks]) != event.task.nr_parallel - 1:
                                self.waiting_parallel_tasks.append(event.task)
                            else: # sum([1 if event.task.parallel_id == task.parallel_id else 0 for task in self.waiting_parallel_tasks]):
                                # Remove all waiting tasks
                                for task in self.waiting_parallel_tasks:
                                    if task.parallel_id == event.task.parallel_id:
                                        self.waiting_parallel_tasks.remove(task)
                                next_tasks = self.generate_next_task(event.task)
                        else:
                            next_tasks = self.generate_next_task(event.task)                                

                    for next_task in next_tasks:
                        if next_task.task_type == 'Complete':
                            self.events.append(Event(EventType.CASE_DEPARTURE, self.now, event.task))
                        else:
                            self.available_tasks.append(next_task)
                        
                    if len(self.available_tasks) > 0 and len(self.available_resources) > 0:
                        self.events.append(Event(EventType.PLAN_TASKS, self.now))
                    self.events.sort()

                if event.event_type == EventType.CASE_DEPARTURE:
                    case = self.uncompleted_cases[event.task.case_id]
                    #print(case.completed_tasks, case.uncompleted_tasks)
                    case.departure_time = self.now
                    del self.uncompleted_cases[case.id]
                    self.completed_cases[event.task.case_id] = case

                    cycle_time = case.departure_time - case.arrival_time
                    self.sumx += cycle_time
                    self.sumxx += cycle_time * cycle_time
                    self.sumw += 1


                    if self.reward_function == 'case_completion':
                        self.current_reward += 1
                        self.total_reward += 1

                    # Calculate reward
                    if self.reward_function == 'cycle_time':                        
                        reward = 1 / (1 + cycle_time)
                        self.current_reward += reward #- len(self.uncompleted_cases)
                        self.total_reward += reward



        if self.now > self.running_time:
            self.status = "FINISHED"
            total_CT = 0
            for case in self.completed_cases.values():
                cycle_time = case.departure_time - case.arrival_time
                total_CT += cycle_time

            
            for event in self.events:
                if event.event_type == EventType.TASK_COMPLETE:
                    self.resource_total_busy_time[event.resource] += self.running_time - self.resource_last_start[event.resource]
                    
            # Uncomment to include the cycle time of uncompleted cases
            for case in self.uncompleted_cases.values():
                cycle_time = self.running_time - case.arrival_time
                total_CT += cycle_time
                self.sumx += cycle_time
                self.sumxx += cycle_time * cycle_time
                self.sumw += 1

            print(f'Uncompleted cases: {len(self.uncompleted_cases)}. Completed cases: {len(self.completed_cases)}.')
            print(f'Resource utilisation: {[(resource, busy_time/self.running_time) for resource, busy_time in self.resource_total_busy_time.items()]}')
            print(f'Total reward: {self.total_reward}. Total CT: {self.sumx}')
            print(f'Mean cycle time: {self.sumx/self.sumw}. Standard deviation: {np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)}')
            print(f'Total cycle time: {total_CT}\n')
            
            if self.write_to != None:
                utilisation = [busy_time/self.running_time for resource, busy_time in self.resource_total_busy_time.items()]
                resource_str = ''
                for i in range(len(self.resources)):
                    resource_str += f'{utilisation[i]},'
                if self.planner != None:
                    with open(self.write_to + f'\\{self.planner}_{self.config_type}.txt', "a") as file:
                        file.write(f"{len(self.uncompleted_cases)},{resource_str}{self.total_reward},{self.sumx/self.sumw},{np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)}\n")

            return len(self.uncompleted_cases),self.total_reward,self.sumx/self.sumw,np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)
        


                
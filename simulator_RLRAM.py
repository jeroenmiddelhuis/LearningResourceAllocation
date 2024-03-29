import numpy as np
import os
import sys
from enum import Enum, auto
from collections import deque
import random
import json

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
    def __init__(self, running_time, planner, config_type, reward_function=None, write_to=None):
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

        self.mean_interarrival_time = config['mean_interarrival_time']
        self.task_types = config['task_types']
        self.resources = config['resources']       
        self.resource_pools = config['resource_pools']
        self.transitions = config['transitions']

        # self.task_types = sorted(list(self.task_types))
        # self.resources = sorted(list(set(np.hstack(list(self.resources))))) 

        self.planner = planner 
        if self.planner != None: self.planner.resource_pools = self.resource_pools

        self.resource_total_busy_time = {resource:0 for resource in self.resources}
        self.resource_last_start = {resource:0 for resource in self.resources}

        # Reinforcement learning
        self.input = [resource + '_availability' for resource in self.resources] + \
                     [resource + '_to_task' for resource in self.resources] + \
                     [task_type for task_type in self.task_types if task_type != 'Start'] #+\
                     #['Total tasks'] # Should be lists of strings
        #                      [resource + '_busy_time' for resource in self.resources] + \
        self.output = [(resource, task) for task in self.task_types[1:] for resource in self.resources if resource in self.resource_pools[task]] + ['Postpone']

        self.reward_function = reward_function
        self.write_to = write_to
        self.current_reward = 0
        self.total_reward = 0

        self.reward_case = 1
        self.reward_task_start = 0
        self.reward_task_complete = {task:i+1 for i, task in enumerate(self.task_types)}
        #self.reward_task_complete = {task:0 for task in self.task_types}

        #print(self.reward_task_complete)

        self.last_reward_moment = 0

        # RLRAM
        self.resource_queues = {resource:[] for resource in self.resources}


        self.init_simulation()


    def generate_initial_task(self, case_id):
        # rvs = random.random()
        # prob = 0
        # for tt, p in self.initial_task_dist.items():
        #     prob += p
        #     if rvs <= prob:
        #         task_type = tt
        #         break
        # return Task(self.now, case_id, task_type)
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
        #print(assignment, [task.task_type for task in self.available_tasks], [resource for resource in self.available_resources])
        self.available_resources.remove(assignment[0])
        self.available_tasks.remove(assignment[1])
        # self.events.append(Event(EventType.TASK_START, self.now, assignment[0], assignment[1]))
        # self.events.sort()      

        self.resource_last_start[assignment[0]] = self.now
        pt = self.sample_processing_time(assignment[0], assignment[1].task_type)
        if self.reward_function == 'task':
            self.current_reward -= pt  
            self.total_reward -= pt
        self.events.append(Event(EventType.TASK_COMPLETE, self.now + pt, assignment[1], assignment[0]))
        self.events.sort()

    def sample_interarrival_time(self):
        return random.expovariate(1/self.mean_interarrival_time)

    def sample_processing_time(self, resource, task):
        # Gamma
        # (mu, sigma) = self.resource_pools[task][resource]
        # sigma = self.cv * mu
        # alpha = mu**2/sigma**2
        # beta = mu/sigma**2
        # return random.gammavariate(alpha, 1/beta)

        # Exponential
        return random.expovariate(1/self.resource_pools[task][resource][0])

        # Gaussian
        (mu, sigma) = self.resource_pools[task][resource]
        pt = random.gauss(mu, sigma)
        while pt < 0:
            pt = random.gauss(mu, sigma)
        return pt

    def init_simulation(self):
        self.available_resources = [resource for resource in self.resources]
        self.events.append(Event(EventType.CASE_ARRIVAL, self.sample_interarrival_time()))

    def get_state(self):
        ### Resource binary, busy time, assigned to + nr of each task
        resources_available = [1 if x in self.available_resources else 0 for x in self.resources]

        #resources_busy_time = [0 for _ in range(len(self.resources))]
        resources_assigned = [0 for _ in range(len(self.resources))]
        for event in self.events:
            if event.event_type == EventType.TASK_COMPLETE:
                resource_index = self.resources.index(event.resource)
                #resources_busy_time[resource_index] = self.now - event.task.start_time
                resources_assigned[resource_index] = self.task_types.index(event.task.task_type)/len(self.task_types) # no +1 because of start task


        if len(self.available_tasks) > 0:
            task_types_num = [min(1, sum([1 if task.task_type == el else 0 for task in self.available_tasks])/100) for el in self.task_types if el != 'Start'] # len(self.available_tasks)
        else:
            task_types_num = [0 for el in self.task_types if el != 'Start']
        return resources_available + resources_assigned + task_types_num

    def run(self):
        while self.now <= self.running_time:
            #print(self.get_state())
            event = self.events.pop(0)            
            self.now = event.moment
            if self.now <= self.running_time: # To prevent next event time after running time
                if event.event_type == EventType.CASE_ARRIVAL:

                    case = self.generate_case() # Automatically added to dict of uncompleted cases
                    for task in self.generate_initial_task(case.id):
                        self.available_tasks.append(task)
                        self.events.append(Event(EventType.PLAN_TASKS, self.now, task))

                    self.events.append(Event(EventType.CASE_ARRIVAL, self.now + self.sample_interarrival_time())) # Schedule new arrival                        
                    self.events.sort()


                if event.event_type == EventType.PLAN_TASKS:
                    if self.planner == None: # DRL algorithm handles processing of assignments (training and inference)
                        self.current_task = event.task
                        break
                    else: #at inference time, we call the plan function of the planner
                        assignment = self.planner.plan(event.task, self.resource_queues, self.resource_pools) # Assignment is the resource
                        #print(assignment, event.task.task_type)
                        #print('before',self.available_resources, [(k, len(v)) for k, v in self.resource_queues.items()])
                        self.resource_queues[assignment].append(event.task)
                        
                        #print(self.available_resources, [(k, len(v)) for k, v in self.resource_queues.items()], '\n')
                        # Start new task is there is a resoruce available and a task waiting
                        if assignment in self.available_resources:
                            self.process_assignment((assignment, self.resource_queues[assignment][0]))


                if event.event_type == EventType.TASK_COMPLETE:
                    case = self.uncompleted_cases[event.task.case_id]
                    case.complete_task(event.task.task_type)
                    self.resource_total_busy_time[event.resource] += self.now - self.resource_last_start[event.resource]
                    # Release resource
                    self.available_resources.append(event.resource)
                    self.resource_queues[event.resource].remove(event.task)
                    if len(self.resource_queues[event.resource]) > 0:
                        self.process_assignment((event.resource, self.resource_queues[event.resource][0]))
                    
                    #print(self.now - self.resource_last_start[event.resource])
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
                            self.events.append(Event(EventType.PLAN_TASKS, self.now, next_task))
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

                    # Calculate reward
                    if self.reward_function == 'cycle_time': # cost, not reward
                        reward = cycle_time
                        self.current_reward -= reward
                        self.total_reward -= reward


        if self.now > self.running_time:
            if self.reward_function == 'AUC':
                current_reward = (self.running_time - self.last_reward_moment) * len(self.uncompleted_cases)
                self.current_reward -= current_reward
                self.total_reward -= current_reward
                self.last_reward_moment = self.now

            self.status = "FINISHED"
            for event in self.events:
                if event.event_type == EventType.TASK_COMPLETE:
                    self.resource_total_busy_time[event.resource] += self.running_time - self.resource_last_start[event.resource]
                    
            # Uncomment to include the cycle time of uncompleted cases
            for case in self.uncompleted_cases.values():
                cycle_time = self.running_time - case.arrival_time
                self.sumx += cycle_time
                self.sumxx += cycle_time * cycle_time
                self.sumw += 1

            print(f'Uncompleted cases: {len(self.uncompleted_cases)}')
            print(f'Resource utilisation: {[(resource, busy_time/self.running_time) for resource, busy_time in self.resource_total_busy_time.items()]}')
            print(f'Total reward: {self.total_reward}. Total CT: {self.sumx}')
            print(f'Mean cycle time: {self.sumx/self.sumw}. Standard deviation: {np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)}')
            
            if self.write_to != None:
                utilisation = [busy_time/self.running_time for resource, busy_time in self.resource_total_busy_time.items()]
                resource_str = ''
                for i in range(len(self.resources)):
                    resource_str += f'{utilisation[i]},'
                if self.planner != None:
                    #with open(os.path.join(sys.path[0], f'{self.write_to}{self.planner}_results_{self.config_type}.txt'), "a") as file:
                    with open(self.write_to + f'\\{self.planner}_{self.config_type}.txt', "a") as file:
                        file.write(f"{len(self.uncompleted_cases)},{resource_str}{self.total_reward},{self.sumx/self.sumw},{np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)}\n")
                # else:
                #     with open(f'{self.write_to}results_{self.config_type}.txt', "a") as file:
                #         file.write(f"{len(self.uncompleted_cases)},{resource_str}{self.total_reward},{self.sumx/self.sumw},{np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)}\n")

            return len(self.uncompleted_cases),self.total_reward,self.sumx/self.sumw,np.sqrt(self.sumxx / self.sumw - self.sumx / self.sumw * self.sumx / self.sumw)
        


                
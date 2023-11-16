import random
from abc import ABC, abstractmethod
# from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
from typing import List
import pandas as pd
from sb3_contrib import MaskablePPO
import json

class Planner(ABC):
    """Abstract class that all planners must implement."""

    @abstractmethod
    def plan(self):
        """
        Assign tasks to resources from the simulation environment.

        :param environment: a :class:`.Simulator`
        :return: [(task, resource, moment)], where
            task is an instance of :class:`.Task`,
            resource is one of :attr:`.Problem.resources`, and
            moment is a number representing the moment in simulation time
            at which the resource must be assigned to the task (typically, this can also be :attr:`.Simulator.now`).
        """
        raise NotImplementedError


# Greedy assignment
class GreedyPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources in an anything-goes manner."""
    def __str__(self) -> str:
        return 'GreedyPlanner'

    def plan(self, available_tasks, available_resources):
        assignments = []
        available_resources = available_resources.copy()
        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.
        for task in available_tasks:
            for resource in available_resources:
                available_resources.remove(resource)
                assignments.append((task, resource))
                break
        return assignments


class ShortestProcessingTime(Planner):
    def __str__(self) -> str:
        return 'ShortestProcessingTime'

    def __init__(self):        
        self.resource_pools = None # passed through simulator

    def get_possible_assignments(self, available_tasks, available_resources, resource_pools):
        possible_assignments = []
        for task_type in set([task.task_type for task in available_tasks]):
            for resource in available_resources:
                if resource in resource_pools[task_type]:
                    possible_assignments.append((resource, task_type))
        return list(set(possible_assignments))
    
    def plan(self, available_tasks, available_resources, resource_pools):
        available_tasks = available_tasks.copy()
        available_resources = available_resources.copy()        
        assignments = []

        possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        while len(possible_assignments) > 0:            
            spt = 999999
            for assignment in possible_assignments: #assignment[0] = task, assignment[1]= resource
                processing_time = self.resource_pools[assignment[1]][assignment[0]][0]
                if processing_time < spt:
                    spt = processing_time
                    best_assignment = assignment

            assignment = (best_assignment[0], (next((x for x in available_tasks if x.task_type == best_assignment[1]), None)))
            available_tasks.remove(assignment[1])
            available_resources.remove(assignment[0])
            assignments.append(assignment)
            possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        return assignments 


class FIFO(Planner):
    def __str__(self) -> str:
        return 'FIFO'

    def __init__(self):        
        self.resource_pools = None # passed through simulator
        self.task_types = None

    def get_possible_assignments(self, available_tasks, available_resources, resource_pools):
        possible_assignments = []
        for task_type in set(task.task_type for task in available_tasks):
            for resource in available_resources:
                if resource in resource_pools[task_type]:
                    possible_assignments.append((resource, task_type))
        return list(set(possible_assignments))
    
    def plan(self, available_tasks, available_resources, resource_pools):
        available_tasks = available_tasks.copy()
        available_resources = available_resources.copy()        
        self.task_types = list(self.resource_pools.keys())

        assignments = []   
        case_priority_order = sorted(list(set([task.case_id for task in available_tasks])))
        priority_case = 0
        possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        while len(possible_assignments) > 0:
            priority_task_types = [task.task_type for task in available_tasks if task.case_id == case_priority_order[priority_case]]
            #print(possible_assignments)
            #print([task.task_type for task in available_tasks_sorted])
            
            best_assignments = []
            while len(best_assignments) == 0:            
                for possible_assignment in possible_assignments:
                    if possible_assignment[1] in priority_task_types:
                        best_assignments.append(possible_assignment)
                if len(best_assignments) == 0:
                    priority_case += 1
                    priority_task_types = [task.task_type for task in available_tasks if task.case_id == case_priority_order[priority_case]]        
            
            if len(best_assignments) > 0:
                spt = 999999
                best_assignment = random.choice(best_assignments)
                #print()
                # for assignment in best_assignments:
                #     processing_time = self.resource_pools[assignment[1]][assignment[0]][0]
                #     if processing_time < spt:
                #         best_assignment = assignment
                #         spt = processing_time

                assignment = (best_assignment[0], (next((x for x in available_tasks if x.task_type == best_assignment[1]), None)))
                available_tasks.remove(assignment[1])
                available_resources.remove(assignment[0])
                assignments.append(assignment)
                possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
                case_priority_order = sorted(list(set([task.case_id for task in available_tasks])))
                priority_case = 0
        return assignments 
 

class Random(Planner):
    def __str__(self) -> str:
        return 'Random'

    def __init__(self):        
        self.resource_pools = None
        self.task_types = None

    def get_possible_assignments(self, available_tasks, available_resources, resource_pools):
        possible_assignments = []
        for task_type in set(task.task_type for task in available_tasks):
            for resource in available_resources:
                if resource in resource_pools[task_type]:
                    possible_assignments.append((resource, task_type))
        return list(set(possible_assignments))
    
    def plan(self, available_tasks, available_resources, resource_pools):
        available_tasks = available_tasks.copy()
        available_resources = available_resources.copy()        
        self.task_types = list(self.resource_pools.keys())
        assignments = []

        possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        while len(possible_assignments) > 0:
            best_assignment = random.choice(possible_assignments)
            assignment = (best_assignment[0], (next((x for x in available_tasks if x.task_type == best_assignment[1]), None)))
            available_tasks.remove(assignment[1])
            available_resources.remove(assignment[0])
            assignments.append(assignment)
            possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        return assignments
    

from simulator import EventType
from itertools import combinations

class ParkSong(Planner):
    def __str__(self) -> str:
        return 'ParkSong'
    
    def __init__(self):        
        self.resource_pools = None
        self.task_types = None

    def linkSimulator(self, simulator):
        self.simulator = simulator

    def get_possible_assignments(self, available_tasks, available_resources, resource_pools):
        possible_assignments = []
        for task in available_tasks:
            for resource in available_resources:
                if resource in resource_pools[task.task_type]:
                    possible_assignments.append((resource, task.task_type))
        return list(set(possible_assignments))
    
    def get_upcoming_tasks_resources(self, available_tasks, available_resources, resource_pools): # Function that returns upcoming task types and resources
        upcoming_resources = {}        
        upcoming_tasks = {} # (task type, expected time until it becomes available)
        for event in self.simulator.events:
            if event.event_type == EventType.TASK_COMPLETE:
                expected_avail_time = resource_pools[event.task.task_type][event.resource][0]
                upcoming_resources[event.resource] = expected_avail_time
                next_tasks = self.simulator.generate_next_task(event.task, predict=True)
                for task in next_tasks:
                    if task.task_type != 'Complete':
                        upcoming_tasks[task] = expected_avail_time
        return upcoming_resources, upcoming_tasks
    
    def get_cost(self, assignment):
        pass

    def get_edges(self, available_tasks, available_resources, resource_pools):
        #possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        #current_resources = available_resources
        #current_tasks = [task for resource, task in possible_assignments]
        upcoming_resources, upcoming_tasks = self.get_upcoming_tasks_resources(available_tasks, available_resources, resource_pools)

        edges = {}

        for task in available_tasks:
            for resource in resource_pools[task.task_type].keys():
                if resource in available_resources:
                    edges[(resource, task)] = resource_pools[task.task_type][resource][0]
                else:
                    edges[(resource, task)] = resource_pools[task.task_type][resource][0] + upcoming_resources[resource]

        for task in upcoming_tasks.keys(): # upcoming assignments
            for resource in resource_pools[task.task_type].keys():
                if resource in available_resources: # Resource available
                    edges[(resource, task)] = upcoming_tasks[task] + resource_pools[task.task_type][resource][0]
                else: # Wait for resource
                    edges[(resource, task)] = max(upcoming_resources[resource], upcoming_tasks[task]) + resource_pools[task.task_type][resource][0]
        
        return edges

    def get_combinations(self, edges):
        tasks = list(set([task for (resource, task) in edges.keys()]))
        task_types = [task.task_type for task in tasks]

        nr_combinations = min(task_types.count('Task A') + task_types.count('Task B'), 2) +\
                          min(task_types.count('Task C') + task_types.count('Task D'), 2) +\
                          min(task_types.count('Task E') + task_types.count('Task F'), 2) +\
                          min(task_types.count('Task G') + task_types.count('Task H'), 2) +\
                          min(min(task_types.count('Task I'), 1) + min(task_types.count('Task J'), 2), 2) +\
                          min(task_types.count('Task K') + task_types.count('Task L'), 2)
        #print(nr_combinations, task_types)
        return nr_combinations

    def get_cost(self, combi, edges):
        if len(set([resource for (resource, task) in combi])) == len(combi) and len(set([task for (resource, task) in combi])) == len(combi):
            cost = 0
            for assignment in combi:
                cost += edges[assignment]
        else:
            cost = np.inf
        return cost

    def plan(self, available_tasks, available_resources, resource_pools):
        #print(self.simulator.now)
        available_tasks = available_tasks.copy()
        available_resources = available_resources.copy()
        self.task_types = list(self.resource_pools.keys())
        assignments = []

        edges = self.get_edges(available_tasks, available_resources, resource_pools)
        #print('dafasdfadsf', edges)
        #print(self.get_combinations(edges), len(edges), len(list(set([task for (resource, task) in edges.keys()]))))
        combis = list(combinations(edges, self.get_combinations(edges)))
        #print(combis)
        costs = [self.get_cost(combi, edges) for combi in combis]
        #print(costs)

        lowest_combi = combis[costs.index(min(costs))]
        lowest_cost = min(costs)

        if lowest_combi != None and lowest_cost != np.inf:
            for assignment in lowest_combi:
                if assignment[0] in available_resources and assignment[1] in available_tasks: #:'
                    #print('Yes', assignment[0], assignment[1].task_type)
                      
                    # available_tasks.remove(assignment[1])
                    # available_resources.remove(assignment[0])
                    assignments.append(assignment)
                    #print('in loop', assignments)
                # elif assignment[0] in available_resources and assignment[1] in [task.task_type for task in available_tasks]:     
                #     print('Yes 2', assignment[0], assignment[1].task_type)               
                #     assignment = (assignment[0], (next((x for x in available_tasks if x.task_type == assignment[1].task_type), None))) 
                #     available_tasks.remove(assignment[1])
                #     available_resources.remove(assignment[0])
                #     assignments.append(assignment)
                    # print('assignmetns')
                    # print(assignment[0], available_resources)
                    # print(assignment[1], available_tasks)
        # else:
        #     print('No solution fouond')
        #     print([task.task_type for (resource, task) in edges.keys()])
        #     print([resource for (resource, task) in edges.keys()])

        # print('lowest combi', [(resource, task) for (resource, task) in lowest_combi])
        # print('lowest combi', [(resource, task.task_type) for (resource, task) in lowest_combi])
        # print('lowest cost', lowest_cost)
        # print('Resources', available_resources)
        # print('Tasks', [(task.task_type, task) for task in available_tasks])
        # print('Tasks edges', [task.task_type for (resource, task) in edges.keys()])
        #print('final', assignments, '\n')

        return assignments
    


class RLRAM(Planner):
    def __str__(self) -> str:
        return 'RLRAM'

    def __init__(self, qtable_path): 
        self.workloads = ['available', 'low', 'normal', 'high', 'overloaded']

        with open(qtable_path, "r") as f:
            self.qtable = json.loads(f.read())   
    

    def get_workload(self, queue, resource):
        if len(queue[resource]) == 0:
            return 'available'
        elif len(queue[resource]) <= 5:
            return 'low'
        elif len(queue[resource]) <= 10:
            return 'normal'
        elif len(queue[resource]) <= 20:
            return 'high'
        else:
            return 'overloaded'


    def plan(self, task, resource_queues, resource_pools):
        #print(resource_pools)
        best_q_value = np.inf
        best_action = None
        for i, resource in enumerate(list(resource_queues.keys())):
            if resource in list(resource_pools[task.task_type].keys()): # Eligible action
                workload = self.get_workload(resource_queues, resource)
                #print(resource, workload, task.task_type, self.qtable[task.task_type][i][self.workloads.index(workload)])
                if self.qtable[task.task_type][i][self.workloads.index(workload)] < best_q_value:
                    #print('best', self.qtable[task.task_type][i][self.workloads.index(workload)])
                    best_action = resource
                    best_q_value = self.qtable[task.task_type][i][self.workloads.index(workload)]
                    #print('best action', resource, task.task_type)
        #print(best_action, '\n')
        return best_action

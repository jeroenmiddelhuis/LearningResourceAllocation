import random
from abc import ABC, abstractmethod
# from sb3_contrib.ppo_mask import MaskablePPO
import numpy as np
from typing import List
import pandas as pd
from sb3_contrib import MaskablePPO


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


class Bayes_planner(Planner):

    def __str__(self) -> str:
        return 'Bayesian'

    def __init__(self, a1, a2, a3, a4, a5, a6, a7,simulator1):

        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.a7 = a7

        df = pd.DataFrame([])

        tasks = [key for key in simulator1.resource_pools.keys()]
        for ind in range(len(tasks)):
            df.loc[ind, 'task'] = tasks[ind]
            df.loc[ind, 'prob_finish'] = simulator1.transitions[tasks[ind]][-1]
            for key in simulator1.resource_pools[tasks[ind]].keys():
                df.loc[ind, key] = simulator1.resource_pools[tasks[ind]][key][0]

        self.df = df

        task_ranking_dict = {}

        Tasks_names = tasks

        res_list = [col for col in list(df.columns) if 'Resource' in col]
        for task in Tasks_names:


            curr_df = pd.DataFrame([])
            for ind, res in enumerate(res_list):
                curr_df.loc[ind, 'Resource'] = res
                curr_df.loc[ind, 'time'] = df.loc[df['task'] == task, res].item()

            curr_df = curr_df.sort_values(by=['time'])
            curr_df = curr_df.reset_index()
            task_ranking = []

            for ind in range(curr_df.shape[0]):
                if curr_df.loc[ind, 'time'].item() > 0:
                    task_ranking.append(curr_df.loc[ind, 'Resource'])

            # if df.loc[df['task'] == task, 'Resource 1'].item() < df.loc[df['task'] == task, 'Resource 2'].item():
            #     task_ranking = ['Resource 1', 'Resource 2']
            # else:
            #     task_ranking = ['Resource 2', 'Resource 1']

            task_ranking_dict[task] = task_ranking



        resourse_ranking_dict = {}

        # res_names = ['Resource 1', 'Resource 2']

        for res in res_list:

            num_none_nan = np.sum(df.sort_values(by=[res])[res] > 0)
            res_ranking = list(df.sort_values(by=[res])['task'][:num_none_nan])

            # if df.loc[df['task'] == 'Task A', res].item() < df.loc[df['task'] == 'Task B', res].item():
            #     res_ranking = ['Task A', 'Task B']
            # else:
            #     res_ranking = ['Task B', 'Task A']

            resourse_ranking_dict[res] = res_ranking



        resource_ranking_dict_score = {}

        for key in resourse_ranking_dict.keys():
            for ind_task, task in enumerate(resourse_ranking_dict[key]):
                resource_ranking_dict_score[(key, resourse_ranking_dict[key][ind_task])] = ind_task + 1

            # resource_ranking_dict_score[(key, resourse_ranking_dict[key][0])] = 1
            # resource_ranking_dict_score[(key, resourse_ranking_dict[key][1])] = 2

        self.resource_ranking_dict_score = resource_ranking_dict_score

        task_ranking_dict_score = {}

        for key in task_ranking_dict.keys():
            for ind_res, res in enumerate(task_ranking_dict[key]):
                task_ranking_dict_score[(key, task_ranking_dict[key][ind_res])] = ind_res + 1

            # task_ranking_dict_score[(key, task_ranking_dict[key][0])] = 1
            # task_ranking_dict_score[(key, task_ranking_dict[key][1])] = 2

        self.task_ranking_dict_score = task_ranking_dict_score

    def give_queue_lenght(self, available_tasks):

        queue_len = {}
        keys_lens = [key for key in queue_len]
        for ind in range(len(available_tasks)):

            if available_tasks[ind].task_type in keys_lens:
                queue_len[available_tasks[ind].task_type] += 1
            else:
                queue_len[available_tasks[ind].task_type] = 1
                keys_lens = [key for key in queue_len]

        return queue_len

    def get_possible_assignments(self, available_tasks, available_resources, resource_pools):
        possible_assignments = []
        for task in available_tasks:
            for resource in available_resources:
                if resource in resource_pools[task.task_type]:
                    possible_assignments.append((resource, task))
        return list(set(possible_assignments))


    def plan(self, available_tasks, available_resources, resource_pools):

        available_resources = available_resources.copy()
        available_tasks = available_tasks.copy()

        # assign the first unassigned task to the first available resource, the second task to the second resource, etc.

        assignments = []
        possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        while len(possible_assignments) > 0:

            queue_lens = self.give_queue_lenght(available_tasks)
            queue_len_keys = [key for key in queue_lens.keys()]
            df_scores = pd.DataFrame([])

            for assignment in possible_assignments:
                resource = assignment[0]
                task = assignment[1]

                mean_val = self.df.loc[self.df['task'] == task.task_type, resource].item()
                var_val = self.df.loc[self.df['task'] == task.task_type, resource].item()
                prob_fin = self.df.loc[self.df['task'] == task.task_type, 'prob_finish'].item()
                if task.task_type in queue_len_keys:
                    queue_lenght = queue_lens[task.task_type]
                else:
                    queue_lenght = 0

                score = self.a1 * mean_val+self.a2 * var_val - self.a3 * prob_fin - self.a4 * queue_lenght\
                        + self.a5*self.resource_ranking_dict_score[(resource, task.task_type)]+self.a6*self.task_ranking_dict_score[(task.task_type, resource)]
                curr_ind = df_scores.shape[0]


                df_scores.loc[curr_ind, 'task_type'] = task.task_type
                df_scores.loc[curr_ind, 'task'] = task
                df_scores.loc[curr_ind, 'resource'] = resource
                df_scores.loc[curr_ind, 'score'] = score

            df_scores = df_scores.sort_values(by=['score']).reset_index()
            best_score = df_scores.loc[0, 'score']
            if best_score < self.a7:
                available_resources.remove(df_scores.loc[0, 'resource'])
                available_tasks.remove(df_scores.loc[0, 'task'])
                assignments.append((df_scores.loc[0, 'resource'], df_scores.loc[0, 'task']))
                possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
            else:
                break


        return assignments



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
            

        # while len(possible_assignments) > 0:
        #     best_assignments = []
        #     for task in available_tasks_sorted: # check if a task_type starting at the end is possible
        #         for possible_assignment in possible_assignments:
        #             if possible_assignment[1] == task.task_type:
        #                 best_assignments.append(possible_assignment)
        #                 break
        #         if len(best_assignments) > 0:
        #             break

        #     best_assignment = random.choice(best_assignments)
        #     assignment = (best_assignment[0], (next((x for x in available_tasks if x.task_type == best_assignment[1]), None)))
        #     available_tasks.remove(assignment[1])
        #     available_resources.remove(assignment[0])
        #     assignments.append(assignment)
        #     possible_assignments = self.get_possible_assignments(available_tasks, available_resources, resource_pools)
        # return assignments 


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

# DRL based assignment
class PPOPlanner(Planner):
    """A :class:`.Planner` that assigns tasks to resources following policy dictated by (pretrained) DRL algorithm."""

    def __str__(self) -> str:
        return 'PPOPlanner'

    def __init__(self, model_name, check_interval) -> None:
        self.model = MaskablePPO.load(f'{model_name}')
        self.check_interval = check_interval
        self.resources = None
        self.task_types = None
        self.inputs = None
        self.output = []
        self.resource_pools_indexes = {}
        self.reward_interval = 3

        self.simulator = None

    #pass the simulator for bidirectional communication
    def linkSimulator(self, simulator):
        self.simulator = simulator

    def get_possible_assignments(self, available_tasks, available_resources, resource_pools):
        possible_assignments = []
        for task_type in set(task.task_type for task in available_tasks):
            for resource in available_resources:
                if resource in resource_pools[task_type]:
                    possible_assignments.append((resource, task_type))
        return list(set(possible_assignments))

    def define_action_masks(self, state) -> List[bool]:
        #state = self.simulator.get_state()
        mask = [0 for _ in range(len(self.simulator.output))]

        for task_type in self.simulator.task_types:
            if state[self.simulator.input.index(task_type)] > 0:
                for resource in self.simulator.resource_pools[task_type]:
                    if state[self.simulator.input.index(resource + '_availability')] > 0:
                        mask[self.simulator.output.index((resource, task_type))] = 1

        if self.check_interval != None:
            if sum(mask) > 0:
                mask[-2] = 1 # Set postpone action to 1
            else:
                mask[-1] = 1 # Fake transtition
        else:
            mask[-2] = 1

        return list(map(bool, mask))

    def take_action(self, action):   
        return self.simulator.output[action]



    def plan(self, available_resources, unassigned_tasks, resource_pools):
        #if len(self.get_possible_assignments(self, available_resources, unassigned_tasks, resource_pools)) == 0:
        state = self.simulator.get_state()
        #print(self.simulator.now)
        
        #print(state)
        #mask = self.define_action_masks(state)        
        
        assignments = []

        # PROBLEM: like this, if a resource and a task are available but the net tells to skip the assignment
        while (sum(self.define_action_masks(state)) > 1): #the do nothing action is always available
            state = self.simulator.get_state()
            mask = self.define_action_masks(state)

            action, _states = self.model.predict(state, action_masks=mask)
            assignment = self.simulator.output[action]
            #task, resource = self.take_action(action)
            #print(action)

            if assignment != 'Postpone' and assignment != 'Wait':
                #print(f"AVAILABLE RESOURCES: {available_resources}")
                #print(f"UNASSIGNED TASKS: {unassigned_tasks}")
                #print(f"NUMBER OF POSSIBLE ASSIGNMENTS: {sum(self.getActionMasks(self.getState(available_resources, unassigned_tasks, busy_resources))) - 1}")
                assignment = (assignment[0], (next((x for x in self.simulator.available_tasks if x.task_type == assignment[1]), None)))
                #print(assignment)
                self.simulator.process_assignment(assignment)
            else:
                break # return to simulator
        return []
        #print(f"ASSIGNMENTS: {assignments}")
        #return assignments
        

    def report(self, event):
        pass#print(event)

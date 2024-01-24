import gymnasium as gym
from gymnasium import spaces, Env
import random
import numpy as np
from typing import List

from simulator import Simulator

class BPOEnv(Env):
<<<<<<< HEAD
    def __init__(self, running_time, config_type, reward_function=None, check_interval=None, write_to=None) -> None:
        super().__init__()
=======
    def __init__(self, running_time, config_type, reward_function=None, postpone_penalty=0, write_to=None) -> None:
>>>>>>> 963524a6db488a8b60cdc2cfd22e9a61686f17cf
        self.num_envs = 1
        self.running_time = running_time
        self.counter = 0
        self.nr_bad_assignments = 0
        self.nr_postpone = 0
        self.nr_fake_transitions = 0
        self.config_type = config_type
        self.reward_function = reward_function
<<<<<<< HEAD
        self.check_interval = check_interval
=======
        self.postpone_penalty = postpone_penalty
>>>>>>> 963524a6db488a8b60cdc2cfd22e9a61686f17cf
        self.write_to = write_to
        self.action_number = [0, 0, 0]
        self.action_time = [0, 0, 0]
        self.step_print = False
        self.last_reward = 0
        self.additional_rewards = 0
        self.previous_reward_time = 0
        self.t_now = 0
        self.reward_intervals = []

<<<<<<< HEAD
        self.simulator = Simulator(running_time=self.running_time, planner=None, config_type=self.config_type,
                                   reward_function=self.reward_function, check_interval=self.check_interval,
                                   write_to=self.write_to)

=======
        self.simulator = Simulator(running_time=self.running_time, planner=None, config_type=self.config_type, reward_function=self.reward_function, write_to=self.write_to)
        #print(self.simulator.input)
        #print(self.simulator.output)
>>>>>>> 963524a6db488a8b60cdc2cfd22e9a61686f17cf
        #define lows and highs for different sections of the input
        lows = np.array([0 for x in range(len(self.simulator.input))])

        # Availability, busy times, assigned to, task numbers proportion, total
        highs = np.array([1 for x in range(len(self.simulator.resources))] +\
                         [float(len(self.simulator.task_types)) for x in range(len(self.simulator.resources))] +\
                         [1 for x in range(len(self.simulator.task_types) - 1)]) #+\
<<<<<<< HEAD
                         #[np.finfo(np.float64).max])/                       
        
        #                          [np.finfo(np.float64).max for x in range(len(self.simulator.resources))] +\
        # highs = np.array([1 for x in range(len(self.simulator.resources))] +\
        #                  [np.finfo(np.float64).max for x in range(len(self.simulator.resources))] +\
        #                  [float(len(self.simulator.task_types)) for x in range(len(self.simulator.resources))] +\
        #                  [np.finfo(np.float64).max for x in range(len(self.simulator.task_types))])
=======
                         #[np.finfo(np.float64).max])/        
>>>>>>> 963524a6db488a8b60cdc2cfd22e9a61686f17cf
        
        self.observation_space = spaces.Box(low=lows,
                                            high=highs,
                                            shape=(len(self.simulator.input),), dtype=np.float64) #observation space is the cartesian product of resources and tasks

        # spaces.Discrete returns a number between 0 and len(self.simulator.output)
        self.action_space = spaces.Discrete(len(self.simulator.output)) #action space is the cartesian product of tasks and resources in their resource pool
<<<<<<< HEAD

        #while (sum(self.define_action_masks()) <= 1):
        self.simulator.run() # Run the simulator to get to the first decision


    def step(self, action):
        self.t_now = self.simulator.now

        if self.step_print: print('Time:\t', self.simulator.now)
        if self.step_print: print('State:\t', self.simulator.get_state())
        self.counter += 1

        if action == len(self.simulator.output) - 2:
            self.nr_postpone += 1
        if action == len(self.simulator.output) - 1:
            self.nr_fake_transitions += 1

        print_every = 5000
=======
    

    def step(self, action):
        if action == len(self.simulator.output)-1:
            self.nr_postpone += 1
        self.counter += 1
        print_every = 20000
>>>>>>> 963524a6db488a8b60cdc2cfd22e9a61686f17cf
        if self.counter % print_every == 0:
            print(f'Nr of postpones: {self.nr_postpone}/{print_every}')
            print(f'Nr of fake transitions: {self.nr_fake_transitions}/{print_every}')
            print(self.simulator.now)
            self.nr_fake_transitions = 0
            self.nr_postpone = 0
            state = self.simulator.get_state()
            print(state, '\n')

        # 1 Process action
        # 2 Do the timestep
        # 3 Return reward
        # Assign one resources per iteration. If possible, another is assigned in next step without advancing simulator
        assignment = self.simulator.output[action] # (Task X, Resource Y)
<<<<<<< HEAD
        #print('Before:',self.simulator.get_state(), self.define_action_masks(), assignment, self.simulator.now)
        # state = self.simulator.get_state()
        # if assignment == 'Postpone':
        #     if state[0] > 0 and state[1] > 0:
        #         self.simulator.current_reward -= 100
        #     elif state[0] > 0 or state[1] > 0:
        #         self.simulator.current_reward -= 50
        #print(assignment, self.simulator.get_state(), self.action_masks(), self.simulator.output)
        #print('Moment of decisions: ', self.simulator.now, '. Action:', assignment, self.simulator.get_state())
        #shortened_state = [np.round(i, 2) if n in [5, 6] else i for n, i in enumerate(self.simulator.get_state())]
        #print(f'Action at t={t_now}. -- Nr cases: {len(self.simulator.uncompleted_cases)}. -- State: {shortened_state}. -- Action: {assignment}.')

        # unassigned_tasks = [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types[:-1]] # sum of unassigned tasks per type

        # available_resources = [resource for resource in self.simulator.available_resources]

        #print(self.simulator.now, '\t', unassigned_tasks, '\t', available_resources, '\t', assignment, '\t', reward)

        if self.step_print: print('Action:\t', assignment)
        if assignment != 'Postpone' and assignment != 'Wait':

            assignment = (assignment[0], (next((x for x in self.simulator.available_tasks if x.task_type == assignment[1]), None)))
            if assignment[0] in self.simulator.available_resources and assignment[1] != None:
                self.simulator.process_assignment(assignment)

        self.simulator.run()        

        self.reward_intervals.append(self.simulator.now - self.t_now)

        if self.step_print: print('Reward:\t', reward)
        if self.step_print: print('Time after reward:\t', self.simulator.now, '\t Elapsed time:\t', self.simulator.now-self.t_now, '\n')
        if assignment == 'Postpone':
            self.action_time[0] += self.simulator.now-self.t_now
            self.action_number[0] += 1
        elif self.simulator.now-self.t_now > 0:
            self.action_time[1] += self.simulator.now-self.t_now
            self.action_number[1] += 1
            self.action_time[2] += self.simulator.now-self.t_now
            self.action_number[2] += 1
        else:
            self.action_time[2] += self.simulator.now-self.t_now
            self.action_number[2] += 1


        self.t_now = self.simulator.now
        reward = self.simulator.current_reward
        self.simulator.current_reward = 0


        self.last_reward = reward

=======
        #print(assignment)
        if self.step_print: print('Action:\t', assignment)
        if assignment != 'Postpone':            
            assignment = (assignment[0], (next((x for x in self.simulator.available_tasks if x.task_type == assignment[1]), None)))
            self.simulator.process_assignment(assignment)
            while (sum(self.simulator.define_action_masks()) <= 1) and (self.simulator.status != 'FINISHED'):
                self.simulator.run()
        else: # Postpone
            self.simulator.current_reward -= self.postpone_penalty
            unassigned_tasks =  [min(1.0, sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks])/100) for el in self.simulator.task_types] # sum of unassigned tasks per type
            available_resources = [resource for resource in self.simulator.available_resources]
            while (self.simulator.status != 'FINISHED') and ((sum(self.simulator.define_action_masks()) <= 1) or (unassigned_tasks == [min(1.0, sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks])/100) for el in self.simulator.task_types] and \
                    available_resources == [resource for resource in self.simulator.available_resources])):
                self.simulator.run()

        reward = self.simulator.current_reward
        self.simulator.current_reward = 0

>>>>>>> 963524a6db488a8b60cdc2cfd22e9a61686f17cf
        if self.simulator.status == 'FINISHED':
            if self.simulator.reward_function == 'AUC':
<<<<<<< HEAD
                current_reward = (self.simulator.now - self.simulator.last_reward_moment) * len(self.simulator.uncompleted_cases)
                self.simulator.current_reward -= current_reward
                self.simulator.total_reward -= current_reward
                self.simulator.last_reward_moment = self.simulator.now
            # self.simulator.last_reward_moment = self.simulator.running_time

            # reward = self.simulator.current_reward
            # self.simulator.current_reward = 0
            # print('FINAL REWARD', current_reward)
            print('mean', np.mean(self.reward_intervals), 'min', min(self.reward_intervals), 'max', max(self.reward_intervals))
            return self.simulator.get_state(), reward, True, {}, {}
        else:
            # current_reward = (self.simulator.now - self.simulator.last_reward_moment) * len(self.simulator.uncompleted_cases)
            # self.simulator.current_reward -= current_reward
            # self.simulator.total_reward -= current_reward
            # self.simulator.last_reward_moment = self.simulator.now

            # reward = self.simulator.current_reward
            # self.simulator.current_reward = 0
            # shortened_state = [np.round(i, 2) if n in [5, 6] else i for n, i in enumerate(self.simulator.get_state())]
            # print(f'Reward at t={self.simulator.now} after {self.simulator.now - t_now} time untis. -- Nr cases: {len(self.simulator.uncompleted_cases)}. -- State: {shortened_state}. -- Reward: {reward}. \n')

            return self.simulator.get_state(), reward, False, {}, {}
=======
                    current_reward = (self.simulator.now - self.simulator.last_reward_moment) * len(self.simulator.uncompleted_cases)
                    self.simulator.current_reward -= current_reward
                    self.simulator.total_reward -= current_reward
                    self.simulator.last_reward_moment = self.simulator.now
            return self.simulator.state, reward, True, {}, {}
        else:
            return self.simulator.state, reward, False, {}, {}
>>>>>>> 963524a6db488a8b60cdc2cfd22e9a61686f17cf


    def reset(self, seed: int | None = None):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """

        print("-------Resetting environment-------")
        self.__init__(self.running_time, self.config_type, reward_function=self.reward_function, 
<<<<<<< HEAD
                      check_interval=self.check_interval, write_to=self.reward_function)

        # self.simulator = Simulator(running_time=self.running_time, planner=None, config_type=self.config_type, reward_function=self.reward_function, write_to=self.write_to)
=======
                      postpone_penalty=self.postpone_penalty, write_to=self.write_to)
        
        while (sum(self.simulator.define_action_masks()) <= 1):
            self.simulator.run() # Run the simulator to get to the first decision
            #self.simulator.state = self.simulator.get_state()
        
>>>>>>> 963524a6db488a8b60cdc2cfd22e9a61686f17cf
        # while (sum(self.define_action_masks()) <= 1):
        #     self.simulator.run() # Run the simulator to get to the first decision

        #self.finished = False
<<<<<<< HEAD
        return self.simulator.get_state(), {}
=======
        
        self.previous_state = self.simulator.state
        self.previous_mask = self.action_masks()
        return self.simulator.state, {}
>>>>>>> 963524a6db488a8b60cdc2cfd22e9a61686f17cf


    def render(self, mode='human', close=False):
        print(f"Average reward: {self.average_cycle_time}")

<<<<<<< HEAD

    def action_masks(self) -> List[bool]:
        return self.simulator.define_action_masks()

    """
    TRAINING
    Needed:
        >Functions:
            -Simulator step function: continues until plan is called
            -Check output function
            -Get state function
            -Action function
                *If multiple actions necessary -> better to invoke step function multiple times
                and pass the assignments to the simulator once
            -Reward function
        >Adjustments:
            -Sample interarrivals during training (no fixed file)

    Optional:
        -Use Env.close() -> disposes all garbage

    INFERENCE
    """
=======

    # define mask based on current environment state (only the 3 vectors that are also known at inference time!)
    # def define_action_masks(self) -> List[bool]:
    #     mask = [True if resource in self.simulator.available_resources and task in [_task.task_type for _task in self.simulator.available_tasks] else False 
    #             for resource, task in self.simulator.output[:-1]] + [True]
    #     return mask

        


    # # define mask based on current environment state (only the 3 vectors that are also known at inference time!)
    # def define_action_masks(self) -> List[bool]:
    #     #state = self.simulator.get_state()
    #     state = self.simulator.state
    #     mask = [0 for _ in range(len(self.simulator.output))]

    #     for task_type in self.simulator.task_types:
    #         if task_type != 'Start':
    #             if state[self.simulator.input.index(task_type)] > 0:
    #                 for resource in self.simulator.resource_pools[task_type]:
    #                     if state[self.simulator.input.index(resource + '_availability')] > 0:
    #                         mask[self.simulator.output.index((resource, task_type))] = 1

    #     mask[-1] = 1 # Set postpone action to 1

    #     mask = [True if _mask == 1 else False for _mask in mask]

    #     self.simulator.last_mask = mask
    #     self.simulator.bpo_state = state
    #     return mask#list(map(bool, mask))

    def action_masks(self) -> List[bool]:
        return self.simulator.define_action_masks()
>>>>>>> 963524a6db488a8b60cdc2cfd22e9a61686f17cf

import gymnasium as gym
from gymnasium import spaces, Env
import random
import numpy as np
from typing import List

from simulator import Simulator

class BPOEnv(Env):
    def __init__(self, running_time, config_type, reward_function=None, postpone_penalty=0, write_to=None) -> None:
        self.num_envs = 1
        self.running_time = running_time
        self.counter = 0
        self.nr_bad_assignments = 0
        self.nr_postpone = 0
        self.config_type = config_type
        self.reward_function = reward_function
        self.postpone_penalty = postpone_penalty
        self.write_to = write_to
        self.step_print = False
        self.last_reward = 0
        self.additional_rewards = 0
        self.previous_reward_time = 0

        self.simulator = Simulator(running_time=self.running_time, planner=None, config_type=self.config_type, reward_function=self.reward_function, write_to=self.write_to)

        #define lows and highs for different sections of the input
        lows = np.array([0 for x in range(len(self.simulator.input))])

        # Availability, busy times, assigned to, task numbers proportion, total
        highs = np.array([1 for x in range(len(self.simulator.resources))] +\
                         [float(len(self.simulator.task_types)) for x in range(len(self.simulator.resources))] +\
                         [1 for x in range(len(self.simulator.task_types) - 1)]) #+\
                         #[np.finfo(np.float64).max])
        
        
        self.observation_space = spaces.Box(low=lows,
                                            high=highs,
                                            shape=(len(self.simulator.input),), dtype=np.float64) #observation space is the cartesian product of resources and tasks

        # spaces.Discrete returns a number between 0 and len(self.simulator.output)
        self.action_space = spaces.Discrete(len(self.simulator.output)) #action space is the cartesian product of tasks and resources in their resource pool
        
        while (sum(self.define_action_masks()) <= 1):
            self.simulator.run() # Run the simulator to get to the first decision


    def step(self, action):
        if action == len(self.simulator.output)-1:
            self.nr_postpone += 1
        self.counter += 1
        print_every = 20000
        if self.counter % print_every == 0:
            print(f'nr of postpones: {self.nr_postpone}/{print_every}')
            self.nr_bad_assignments = 0
            self.nr_postpone = 0
            state = self.simulator.get_state()
            print(state, '\n')

        # 1 Process action
        # 2 Do the timestep
        # 3 Return reward
        # Assign one resources per iteration. If possible, another is assigned in next step without advancing simulator
        assignment = self.simulator.output[action] # (Task X, Resource Y)
        if self.step_print: print('Action:\t', assignment)
        if assignment != 'Postpone':            
            assignment = (assignment[0], (next((x for x in self.simulator.available_tasks if x.task_type == assignment[1]), None)))
            self.simulator.process_assignment(assignment)
            while (sum(self.define_action_masks()) <= 1) and (self.simulator.status != 'FINISHED'):
                self.simulator.run()
        else: # Postpone
            unassigned_tasks = [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types] # sum of unassigned tasks per type
            available_resources = [resource for resource in self.simulator.available_resources]
            while (self.simulator.status != 'FINISHED') and ((sum(self.define_action_masks()) <= 1) or (unassigned_tasks == [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types] and \
                    available_resources == [resource for resource in self.simulator.available_resources])):
                self.simulator.run()

        reward = self.simulator.current_reward
        self.simulator.current_reward = 0

        if self.simulator.status == 'FINISHED':
            if self.simulator.reward_function == 'AUC':
                    current_reward = (self.simulator.now - self.simulator.last_reward_moment) * len(self.simulator.uncompleted_cases)
                    self.simulator.current_reward -= current_reward
                    self.simulator.total_reward -= current_reward
                    self.simulator.last_reward_moment = self.simulator.now
            return self.simulator.get_state(), reward, True, {}, {}
        else:
            return self.simulator.get_state(), reward, False, {}, {}


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
                      postpone_penalty=self.postpone_penalty, write_to=self.write_to)

        # while (sum(self.define_action_masks()) <= 1):
        #     self.simulator.run() # Run the simulator to get to the first decision

        #self.finished = False
        
        state = self.simulator.get_state()
        self.previous_state = state
        self.previous_mask = self.action_masks()
        return state, {}


    def render(self, mode='human', close=False):
        print(f"Average reward: {self.average_cycle_time}")

    # define mask based on current environment state (only the 3 vectors that are also known at inference time!)
    def define_action_masks(self) -> List[bool]:
        state = self.simulator.get_state()
        mask = [0 for _ in range(len(self.simulator.output))]

        for task_type in self.simulator.task_types:
            if task_type != 'Start':
                if state[self.simulator.input.index(task_type)] > 0:
                    for resource in self.simulator.resource_pools[task_type]:
                        if state[self.simulator.input.index(resource + '_availability')] > 0:
                            mask[self.simulator.output.index((resource, task_type))] = 1

        mask[-1] = 1 # Set postpone action to 1

        return list(map(bool, mask))

    def action_masks(self) -> List[bool]:
        return self.define_action_masks()

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
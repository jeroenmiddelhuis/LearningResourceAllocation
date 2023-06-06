import gym
from gym import spaces, Env
import random
import numpy as np
from typing import List

from simulator import Simulator

class BPOEnv(Env):
    def __init__(self, running_time, config_type, reward_function=None, write_to=None, cv=0.25) -> None:
        super().__init__()
        self.num_envs = 1
        self.running_time = running_time
        self.counter = 0
        self.nr_bad_assignments = 0
        self.nr_postpone = 0
        self.config_type = config_type
        self.reward_function = reward_function
        self.write_to = write_to
        self.cv = cv
        self.action_number = [0, 0, 0]
        self.action_time = [0, 0, 0]
        self.step_print = False
        self.last_reward = 0
        self.additional_rewards = 0
        self.previous_reward_time = 0

        self.simulator = Simulator(running_time=self.running_time, planner=None, config_type=self.config_type, reward_function=self.reward_function, write_to=self.write_to, cv=self.cv)

        #define lows and highs for different sections of the input
        lows = np.array([0 for x in range(len(self.simulator.input))])

        # Availability, busy times, assigned to, task numbers proportion, total
        highs = np.array([1 for x in range(len(self.simulator.resources))] +\
                         [float(len(self.simulator.task_types)) for x in range(len(self.simulator.resources))] +\
                         [1 for x in range(len(self.simulator.task_types))] +\
                         [np.finfo(np.float64).max])                         
        
        #                          [np.finfo(np.float64).max for x in range(len(self.simulator.resources))] +\
        # highs = np.array([1 for x in range(len(self.simulator.resources))] +\
        #                  [np.finfo(np.float64).max for x in range(len(self.simulator.resources))] +\
        #                  [float(len(self.simulator.task_types)) for x in range(len(self.simulator.resources))] +\
        #                  [np.finfo(np.float64).max for x in range(len(self.simulator.task_types))])
        
        self.observation_space = spaces.Box(low=lows,
                                            high=highs,
                                            shape=(len(self.simulator.input),), dtype=np.float64) #observation space is the cartesian product of resources and tasks

        # spaces.Discrete returns a number between 0 and len(self.simulator.output)
        self.action_space = spaces.Discrete(len(self.simulator.output)) #action space is the cartesian product of tasks and resources in their resource pool

        while (sum(self.define_action_masks()) <= 1):
            self.simulator.run() # Run the simulator to get to the first decision


    def step(self, action):
        t_now = self.simulator.now
        if self.step_print: print('Time:\t', self.simulator.now)
        if self.step_print: print('State:\t', self.simulator.get_state())
        self.counter += 1
        if action == 2:
            self.nr_bad_assignments += 1
        elif action == len(self.simulator.output)-1:
            self.nr_postpone += 1

        print_every = 500
        if self.counter % print_every == 0:
            print(f'nr of postpones: {self.nr_postpone}/{print_every}')
            self.nr_bad_assignments = 0
            self.nr_postpone = 0
            state = self.simulator.get_state()
            print(state, '\n')
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """


        # 1 Process action
        # 2 Do the timestep
        # 3 Return reward
        # Assign one resources per iteration. If possible, another is assigned in next step without advancing simulator
        assignment = self.simulator.output[action] # (Task X, Resource Y)
        # state = self.simulator.get_state()
        # if assignment == 'Postpone':
        #     if state[0] > 0 and state[1] > 0:
        #         self.simulator.current_reward -= 100
        #     elif state[0] > 0 or state[1] > 0:
        #         self.simulator.current_reward -= 50
        #print(assignment, self.simulator.get_state(), self.action_masks(), self.simulator.output)
        #print('Moment of decisions: ', self.simulator.now, '. Action:', assignment, self.simulator.get_state())
        unassigned_tasks = [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types[:-1]] # sum of unassigned tasks per type

        available_resources = [resource for resource in self.simulator.available_resources]

        #print(self.simulator.now, '\t', unassigned_tasks, '\t', available_resources, '\t', assignment, '\t', reward)


        if self.step_print: print('Action:\t', assignment)
        if assignment != 'Postpone':
            if self.simulator.planner == None:
                assignment = (assignment[0], (next((x for x in self.simulator.available_tasks if x.task_type == assignment[1]), None)))
            #print('stuck 1', assignment, self.simulator.now)
            self.simulator.process_assignment(assignment)

            #print('Before action: ', self.simulator.now)
            # While assignment not possible and simulator not finished (postpone always possible)
            while (sum(self.define_action_masks()) <= 1) and (self.simulator.status != 'FINISHED'):
                #print('ASSIGNED', self.simulator.now)
                self.simulator.run() # breaks each time at resource assignment, continues if no assignment possible
            #print('After action: ', self.simulator.now)
        else: # Postpone action
            unassigned_tasks = [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types] # sum of unassigned tasks per type

            available_resources = [resource for resource in self.simulator.available_resources]

            #print('Before postpone: ', self.simulator.now)
            while (self.simulator.status != 'FINISHED') and ((sum(self.define_action_masks()) <= 1) or (unassigned_tasks == [sum([1 if task.task_type == el else 0 for task in self.simulator.available_tasks]) for el in self.simulator.task_types] and \

                    available_resources == [resource for resource in self.simulator.available_resources])):

                self.simulator.run()
            # Penatly for postponing
            postpone_penalty = -0.001 * len(self.simulator.available_tasks) #.05#-0.005
            self.simulator.current_reward += postpone_penalty
            self.simulator.total_reward += postpone_penalty
            #print('After postpone: ', self.simulator.now)



        reward = self.simulator.current_reward
        # if reward > 0:
        #     reward = reward / (1 + (self.simulator.now-self.previous_reward_time))
        #     self.previous_reward_time = self.simulator.now
        self.simulator.current_reward = 0

        # if assignment == 'Postpone':
        #     reward = 0
        # elif reward != 0:
        #     reward = min(2, 1/-reward)

        # if reward == 0 :
        #     reward = self.last_reward
        #     self.additional_rewards += reward # reward = negative

        self.last_reward = reward

        if self.step_print: print('Reward:\t', reward)
        if self.step_print: print('Time after reward:\t', self.simulator.now, '\t Elapsed time:\t', self.simulator.now-t_now, '\n')
        if assignment == 'Postpone':
            self.action_time[0] += self.simulator.now-t_now
            self.action_number[0] += 1
        elif self.simulator.now-t_now > 0:
            self.action_time[1] += self.simulator.now-t_now
            self.action_number[1] += 1
            self.action_time[2] += self.simulator.now-t_now
            self.action_number[2] += 1
        else:
            self.action_time[2] += self.simulator.now-t_now
            self.action_number[2] += 1

        # Simulation is finished, return current reward (with penalties)

        # if self.step_print and self.action_number[0] > 0 and self.action_number[1] > 0: print('Mean postpone time:', self.action_time[0]/self.action_number[0],
        #                                                                                        '\nMean action time:', self.action_time[2]/self.action_number[2],
        #                                                                                        '\nMean action time (without 0):', self.action_time[1]/self.action_number[1])

        if self.simulator.status == 'FINISHED':
            # print(f'running: {self.simulator.running_time}, last_moment: {self.simulator.last_reward_moment}')
            # current_reward = (self.simulator.running_time - self.simulator.last_reward_moment) * len(self.simulator.uncompleted_cases)
            if self.simulator.reward_function == 'AUC':
                    current_reward = (self.simulator.now - self.simulator.last_reward_moment) * len(self.simulator.uncompleted_cases)
                    self.simulator.current_reward -= current_reward
                    self.simulator.total_reward -= current_reward
                    self.simulator.last_reward_moment = self.simulator.now
            # self.simulator.last_reward_moment = self.simulator.running_time

            # reward = self.simulator.current_reward
            # self.simulator.current_reward = 0
            # print('FINAL REWARD', current_reward)
            return self.simulator.get_state(), reward, True, {}
        else:
            # current_reward = (self.simulator.now - self.simulator.last_reward_moment) * len(self.simulator.uncompleted_cases)
            # self.simulator.current_reward -= current_reward
            # self.simulator.total_reward -= current_reward
            # self.simulator.last_reward_moment = self.simulator.now

            # reward = self.simulator.current_reward
            # self.simulator.current_reward = 0
            return self.simulator.get_state(), reward, False, {}


    def reset(self):
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

        self.simulator = Simulator(running_time=self.running_time, planner=None, config_type=self.config_type, reward_function=self.reward_function, write_to=self.write_to, cv=self.cv)
        while (sum(self.define_action_masks()) <= 1):
            self.simulator.run() # Run the simulator to get to the first decision

        #self.finished = False
        return self.simulator.get_state()


    def render(self, mode='human', close=False):
        print(f"Average reward: {self.average_cycle_time}")

    # define mask based on current environment state (only the 3 vectors that are also known at inference time!)
    def define_action_masks(self) -> List[bool]:
        state = self.simulator.get_state()
        mask = [0 for _ in range(len(self.simulator.output))]

        for task_type in self.simulator.task_types:
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
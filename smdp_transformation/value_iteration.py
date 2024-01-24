import numpy as np
import random

class Environment:
    def __init__(self, initial_state, gamma) -> None:        
        self.initial_state = initial_state
        self.state = self.initial_state
        self.gamma = gamma
        self.now = 0
        self.value_function = {0:0,
                               1:0}

        self.transitions = {
            0:{0:[0.5, 0.5],
                1:[0.3, 0.7]},
            1:{0:[0.4, 0.6],
                   1:[0.8, 0.2]}
        }
        self.rewards = {
            0:{0:-1,
                   1:-8},
            1:{0:-2,
                    1:-8}
        }
        self.times = {
            0:{0:1,
                   1:10},
            1:{0:2,
                   1:7}
        }

        self.iteration = 0

        # random.random()

    def reset(self):
        self.state = self.initial_state
        self.now = 0

    def step(self, action):
        reward = self.rewards[self.state][action]
        time = self.times[self.state][action]
        self.now += time
        rvs = random.random()
        if rvs < self.transitions[self.state][action][0]:
            self.state = 0
        else:
            self.state = 1

        return reward, time

    def evaluate_duration(self, running_time, policy):
        total_reward, total_time, total_steps = 0, 0, 0
        while self.now < running_time:
            action = policy[self.state]
            reward, time = self.step(action)
            total_reward += reward
            total_time += time
            total_steps += 1
        self.reset()
        return total_reward, total_time, total_steps


    def evaluate_steps(self, max_steps, policy):
        total_reward, total_time, total_steps = 0, 0, 0
        while total_steps < max_steps:
            action = policy[self.state]
            reward, time = self.step(action)
            total_reward += reward
            total_time += time
            total_steps += 1
        self.reset()
        return total_reward, total_time, total_steps






    def value_iteration(self, iterations, q_function=[[0,0], [0,0]], value_function=[0,0]):
        for _ in range(iterations):
            # env.transitions[current_state][action][next_state_prob]
            # env.rewards[current_state][action]

            # State 0, action 0
            q_function[0][0] = self.rewards[0][0] + self.gamma * (self.transitions[0][0][0] * value_function[0] + self.transitions[0][0][1] * value_function[1])
            q_function[0][1] = self.rewards[0][1] + self.gamma * (self.transitions[0][1][0] * value_function[0] + self.transitions[0][1][1] * value_function[1])

            q_function[1][0] = self.rewards[1][0] + self.gamma * (self.transitions[1][0][0] * value_function[0] + self.transitions[1][0][1] * value_function[1])
            q_function[1][1] = self.rewards[1][1] + self.gamma * (self.transitions[1][1][0] * value_function[0] + self.transitions[1][1][1] * value_function[1])

            policy = [q_function[0].index(max(q_function[0])), q_function[1].index(max(q_function[1]))]
            value_function = [max(q_function[0]), max(q_function[1])]

        return q_function, value_function, policy
        

    def value_iteration_transformation(self, iterations, tau, q_function=[[0,0], [0,0]], value_function=[0,0]):
        transformed_rewards = {0:{0:self.rewards[0][0]/self.times[0][0],
                                  1:self.rewards[0][1]/self.times[0][1]},
                               1:{0:self.rewards[1][0]/self.times[1][0],
                                  1:self.rewards[1][1]/self.times[1][1]}}
        transformed_p = {0:{0:tau/self.times[0][0],
                            1:tau/self.times[0][1]},
                         1:{0:tau/self.times[1][0],
                            1:tau/self.times[1][1]}}
        
        for _ in range(iterations):
            q_function[0][0] = transformed_rewards[0][0] + self.gamma * (transformed_p[0][0] * (self.transitions[0][0][0] * value_function[0] + self.transitions[0][0][1] * value_function[1])\
                                                                         + ((1 - transformed_p[0][0]) * value_function[0]))
            q_function[0][1] = transformed_rewards[0][1] + self.gamma * (transformed_p[0][1] * (self.transitions[0][1][0] * value_function[0] + self.transitions[0][1][1] * value_function[1])\
                                                                         + ((1 - transformed_p[0][1]) * value_function[0]))
            
            q_function[1][0] = transformed_rewards[1][0] + self.gamma * (transformed_p[1][0] * (self.transitions[1][0][0] * value_function[0] + self.transitions[1][0][1] * value_function[1])\
                                                                         + ((1 - transformed_p[1][0]) * value_function[1]))
            q_function[1][1] = transformed_rewards[1][1] + self.gamma * (transformed_p[1][1] * (self.transitions[1][1][0] * value_function[0] + self.transitions[1][1][1] * value_function[1])\
                                                                         + ((1 - transformed_p[1][1]) * value_function[1]))
        
            policy = [q_function[0].index(max(q_function[0])), q_function[1].index(max(q_function[1]))]
            value_function = [max(q_function[0]), max(q_function[1])]
            

        return q_function, value_function, policy


env = Environment(0, gamma=0.999)
SMDP_policy = env.value_iteration(100000)[2]
MDP_policy = env.value_iteration_transformation(100000, 1)[2]
print(SMDP_policy, MDP_policy)
# print('SMDP: \t\t', env.value_iteration(100000))
# print('Transformed MDP:', env.value_iteration_transformation(10000, 1))

SMDP_total_rewards_duration = []
SMDP_total_time_duration = []
SMDP_total_steps_duration = []

MDP_total_rewards_duration = []
MDP_total_time_duration = []
MDP_total_steps_duration = []

for _ in range(100):
    total_reward, total_time, total_steps = env.evaluate_duration(1000, SMDP_policy)
    SMDP_total_rewards_duration.append(total_reward)
    SMDP_total_time_duration.append(total_time)
    SMDP_total_steps_duration.append(total_steps)

    total_reward, total_time, total_steps = env.evaluate_duration(1000, MDP_policy)
    MDP_total_rewards_duration.append(total_reward)
    MDP_total_time_duration.append(total_time)
    MDP_total_steps_duration.append(total_steps)
    

SMDP_total_rewards_steps = []
SMDP_total_time_steps = []
SMDP_total_steps_steps = []

MDP_total_rewards_steps = []
MDP_total_time_steps = []
MDP_total_steps_steps = []

for _ in range(100):
    total_reward, total_time, total_steps = env.evaluate_steps(1000, SMDP_policy)
    SMDP_total_rewards_steps.append(total_reward)
    SMDP_total_time_steps.append(total_time)
    SMDP_total_steps_steps.append(total_steps)

    total_reward, total_time, total_steps = env.evaluate_steps(1000, MDP_policy)
    MDP_total_rewards_steps.append(total_reward)
    MDP_total_time_steps.append(total_time)
    MDP_total_steps_steps.append(total_steps)   
# print(SMDP_total_rewards_steps)




import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


sns.set(style="darkgrid")
ax = sns.boxplot(data=pd.DataFrame(list(zip(SMDP_total_rewards_duration, MDP_total_rewards_duration, SMDP_total_rewards_steps, MDP_total_rewards_steps)),
                                columns= ['SMDP duration', 'MDP duration', 'SMDP steps', 'MDP steps']))

ax.set(xlabel='Model', ylabel='Cumulative reward', title='Cumulative reward')
plt.show()



ax = sns.boxplot(data=pd.DataFrame(list(zip(SMDP_total_time_duration, MDP_total_time_duration, SMDP_total_time_steps, MDP_total_time_steps)),
                                columns= ['SMDP duration', 'MDP duration', 'SMDP steps', 'MDP steps']))

ax.set(xlabel='Model', ylabel='Total duration', title='Total duration')
plt.show()



ax = sns.boxplot(data=pd.DataFrame(list(zip(SMDP_total_steps_duration, MDP_total_steps_duration, SMDP_total_steps_steps, MDP_total_steps_steps)),
                                columns= ['SMDP duration', 'MDP duration', 'SMDP steps', 'MDP steps']))

ax.set(xlabel='Model', ylabel='Total steps', title='Total steps')
plt.show()
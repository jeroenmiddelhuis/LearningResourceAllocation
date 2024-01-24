from collections import deque
from subprocess import call
import gymnasium as gym
import os
import numpy as np
from bpo_env import BPOEnv
import sys
import random
from simulator_RLRAM import Simulator
import time
import json


class RLRAM_train:
	def __init__(self, running_time, learning_rate, reward_function, gamma, epsilon, config_type):

		self.running_time = running_time
		self.reward_function = reward_function
		self.learning_rate = learning_rate
		self.gamma = gamma
		self.epsilon = epsilon
		self.config_type = config_type		
		self.simulator = Simulator(running_time=self.running_time, planner=None, config_type=self.config_type, reward_function=self.reward_function)
		#print(self.simulator.resource_pools['Task E'])

		self.workloads = ['available', 'low', 'normal', 'high', 'overloaded']
		self.task_types = [task_type for task_type in self.simulator.task_types if task_type != 'Start' and task_type != 'Complete']
		self.qtable = {}

		# For every task_type, several actions are possible where actions = resources
		# For each chosen resource, the state can be in one of several workload levels
		for task_type in self.task_types:
			# Dictionary with an array [x, y] where x is the index of the action/resource and y the workload
			self.qtable[task_type] = [[0.0 for workload in self.workloads] for action in self.simulator.resources]


	def get_random_action(self, task_type):
		return random.choice(list(self.simulator.resource_pools[task_type].keys()))

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

	def epsilon_greedy_policy(self, state):
		if random.random() > self.epsilon: #exploitation
			best_q_value = np.inf
			best_action = []
			for i, resource in enumerate(self.simulator.resources):
				if resource in list(self.simulator.resource_pools[state[0].task_type].keys()): # Eligible action
					workload = self.get_workload(state[1], resource)
					if self.qtable[state[0].task_type][i][self.workloads.index(workload)] < best_q_value:
						best_action = [resource]
						best_q_value = self.qtable[state[0].task_type][i][self.workloads.index(workload)]
					elif self.qtable[state[0].task_type][i][self.workloads.index(workload)] == best_q_value:
						best_action.append(resource)
			if len(best_action) > 1:
				best_action = random.choice(best_action)
			else:
				best_action = best_action[0]
			return best_action, self.get_workload(state[1], best_action)

		else: #exploration
			action = self.get_random_action(state[0].task_type)
			workload = self.get_workload(state[1], action)
			return action, workload

	def reset(self):

		#print(len(self.simulator.available_tasks))
		#print([(k, len(v)) for k, v in self.simulator.resource_queues.items()])
		#print(self.qtable)

		self.simulator = Simulator(running_time=self.running_time, planner=None, config_type=self.config_type, reward_function=self.reward_function)
		self.simulator.run()
		state = [self.simulator.current_task, self.simulator.resource_queues]
		return state

	# Off policy Q-learning
	def q_learning(self, total_episodes, plot_every=100):
		rewards = []   # List of rewards
		times = [] # List of times
		tmp_scores = deque(maxlen=plot_every)     # deque for keeping track of scores
		avg_scores = deque(maxlen=total_episodes)   # average scores over every plot_every episodes

		for episode in range(total_episodes + 1):
			t0 = time.time()
			state = self.reset() # Reset the environment
			#step = 0 
			done = False
			total_rewards = 0 # collected reward within an episode
			if episode % 100 == 0: #monitor progress
				print("\rEpisode {}/{} \n".format(episode, total_episodes), end="") 
			
			while True:
				action, workload = self.epsilon_greedy_policy(state) # epsilon greedy action 
				if self.reward_function == 'task_time':
					self.simulator.current_reward += self.simulator.resource_pools[state[0].task_type][action][0] # we minimize this time
				#print(state)

				# Process the action
				#self.simulator.available_tasks.remove(state[0])
				self.simulator.resource_queues[action].append(state[0])

				# Start new task is there is a resoruce available and a task waiting
				if action in self.simulator.available_resources:
					#print('yes', self.simulator.available_resources, [(k, len(v)) for k, v in self.simulator.resource_queues.items()])
					self.simulator.process_assignment((action, self.simulator.resource_queues[action][0]))
				self.simulator.run()



				new_state = [self.simulator.current_task, self.simulator.resource_queues]

				max_value_next_state = min([self.qtable[new_state[0].task_type][self.simulator.resources.index(resource)][self.workloads.index(self.get_workload(new_state[1], resource))] 
																	for resource in self.simulator.resources if resource in self.simulator.resource_pools[new_state[0].task_type].keys()])
				
				#print()
				#print('max', [self.qtable[new_state[0].task_type][self.simulator.resources.index(resource), self.workloads.index(self.get_workload(new_state[1], resource))] 
				#													for resource in self.simulator.resources if resource in self.simulator.resource_pools[new_state[0].task_type].keys()])
				#print(self.qtable[new_state[0].task_type])
				
				# Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
				reward = self.simulator.current_reward

				#print(reward)
				self.simulator.current_reward = 0
				#print('this',(1 - self.learning_rate) * self.qtable[state[0].task_type][self.simulator.resources.index(action), self.workloads.index(workload)] +\
				#																								self.learning_rate * (reward + self.gamma * max_value_next_state))
				self.qtable[state[0].task_type][self.simulator.resources.index(action)][self.workloads.index(workload)] = (1 - self.learning_rate) * self.qtable[state[0].task_type][self.simulator.resources.index(action)][self.workloads.index(workload)] +\
																												self.learning_rate * (reward + self.gamma * max_value_next_state)
				
				#print(self.qtable[state[0].task_type][self.simulator.resources.index(action), self.workloads.index(workload)])
				total_rewards += reward # sum the rewards collected within an episode
				state = new_state
				if self.simulator.status == "FINISHED":
					tmp_scores.append(total_rewards)
					print(self.qtable)
					break
			if (episode % plot_every == 0): #for plot
				avg_scores.append(np.mean(tmp_scores))
			
			times.append(time.time() - t0)
			rewards.append(total_rewards)


if __name__ == '__main__':
	#if true, load model for a new round of training
	for config_type in ['complete', 'complete_reversed', 'complete_parallel']:
		running_time = 5000
		num_cpu = 1
		load_model = False

		#config_type = 'down_stream'
		print(config_type)
		reward_function = 'task_time'

		rlram_train = RLRAM_train(running_time=running_time, learning_rate=0.1, reward_function=reward_function, gamma=1, epsilon=0.1, config_type=config_type)
		rlram_train.q_learning(5000)

		with open(f"./scenarios/RLRAM_models/{config_type}.json", 'w') as json_file: 
			json.dump(rlram_train.qtable, json_file)

		time_steps = 5e7 # Total timesteps
		n_steps = 51200 # Number of steps for each network update
		# Create log dir
		log_dir = f"./tmp/{config_type}_{int(time_steps)}_{n_steps}/" # Logging training results

		#os.makedirs(log_dir, exist_ok=True)


		# resource_str = ''
		# for resource in env.simulator.resources:
		# 	resource_str += resource + ','
		# with open(f'{log_dir}results_{config_type}.txt', "w") as file:
		# 	# Writing data to a file
		# 	file.write(f"uncompleted_cases,{resource_str}total_reward,mean_cycle_time,std_cycle_time\n")

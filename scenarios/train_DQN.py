import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
from collections import deque
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time
import shutil

from simulator_RLRAM import Simulator


class DDQN_train:
    def __init__(self, running_time, learning_rate, reward_function, gamma, epsilon, batch_size, config_type):

        self.running_time = running_time
        self.reward_function = reward_function
        self.learning_rate = learning_rate
        self.discount_rate = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.config_type = config_type		
        self.simulator = Simulator(running_time=self.running_time, planner=None, config_type=self.config_type, reward_function=self.reward_function)
        #print(self.simulator.resource_pools['Task E'])


        self.memory_length = 24000 # In original paper 400 episodes * 600 steps per epsisode * 0.1
        self.replay_memory = deque(maxlen=self.memory_length)

        self.optimizer = keras.optimizers.Adam(lr=0.001)
        self.loss_fn = keras.losses.mean_squared_error


        available_resources = self.simulator.resources

        self.observation_space = self.simulator.input
        self.action_space = self.simulator.output

        self.nmb_of_inputs = len(self.observation_space)
        self.nmb_of_outputs = len(self.action_space)

        self.model = keras.models.Sequential([
                keras.layers.Flatten(input_shape=(self.nmb_of_inputs,)),
                keras.layers.Dense(32),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Dense(32),
                keras.layers.BatchNormalization(),
                keras.layers.ReLU(),
                keras.layers.Dense(self.nmb_of_outputs)
            ])

        self.target_model = keras.models.clone_model(self.model)

        self.q_values_mean = 0
        self.q_values_counter = 0.001

        if tf.test.gpu_device_name():
            print('GPU found')
        else:
            print("No GPU found")

    def training_step(self, training_batch_size):
        experiences = self.sample_experiences(training_batch_size)
        states, actions, rewards, next_states = experiences

        next_Q_values = self.model(next_states)
        best_next_actions = np.argmax(next_Q_values, axis=1)
        next_mask = tf.one_hot(best_next_actions, self.nmb_of_outputs)
        next_best_Q_values = (self.target_model(next_states).numpy() * next_mask.numpy()).sum(axis=1)
        target_Q_values = (rewards + self.discount_rate * next_best_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.nmb_of_outputs)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss.numpy()


    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.replay_memory), size=batch_size)
        batch = [self.replay_memory[index] for index in indices]
        states, actions, rewards, next_states = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(4)]
        return states, actions, rewards, next_states


    def epsilon_greedy_policy(self, prediction_model, environment, state, nmb_of_tasks, epsilon=0, type_of_epsilon_exp="random"):
        """

        :param prediction_model:
        :param environment:
        :param state:
        :param nmb_of_tasks:
        :param epsilon:
        :param type_of_epsilon_exp: IMPORTANT - using anything besides random here may result in learning policy that is
                    dependent on particular (in this case fifo) strategy present with epsilon probability
        :return:
        """
        if np.random.rand() < epsilon:
            if type_of_epsilon_exp == "random":
                action = np.random.randint(self.nmb_of_outputs)
                sim_action = self.get_action_from_int(action, nmb_of_tasks)
                next_state, reward = self.env_state_mask(environment.step(sim_action))
                return next_state, reward, action, "r"
            elif type_of_epsilon_exp == "fifo":
                next_state, reward, action = self.env_state_mask(environment.step_fifo())
                return next_state, reward, self.get_int_from_sim_action(action), "r"
        else:
            Q_values = prediction_model(state[np.newaxis], training=False)
            global q_values_mean
            global q_values_counter
            q_values_mean += np.mean(Q_values[0])
            q_values_counter += 1
            action = np.argmax(Q_values[0])
            sim_action = self.get_action_from_int(action, nmb_of_tasks)
            next_state, reward = self.env_state_mask(environment.step(sim_action))
            return next_state, reward, action, "q"


    def get_action_from_int(self, int_action_value, nmb_of_tasks):
        """
        :param nmb_of_tasks: number of tasks
        :param nmb_of_resources: number of resources
        :param int_action_value: action value from the range NxM where N - nmb_of_resources, M - nmb_of_tasks
        :return: [resource, task] Vector of length 2 of actions to be taken by the environment
        """
        if int_action_value == 0:
            return [-1, -1]
        else:
            action_coded_value = int_action_value - 1
            resource = math.floor(action_coded_value / nmb_of_tasks)
            task = action_coded_value % nmb_of_tasks
            return [resource, task]


    def get_int_from_sim_action(self, sim_action):
        return sim_action[0]*sim_action[1] + sim_action[1] + 1


    def play_one_step_and_collect_memory(self, env, state, epsilon, type_of_epsilon_exp="random"):
        next_state, reward, action, action_type = self.epsilon_greedy_policy(self.model, env, state, self.action_space[1],
                                                                        epsilon, type_of_epsilon_exp)
        self.replay_memory.append((state, action, reward, next_state))
        return next_state, reward, action, action_type


    def env_state_mask(env_step_return_tuple):
        # loc_list = list(env_step_return_tuple)
        # loc_list[0] = loc_list[0][0:action_space[0], ]
        # return tuple(loc_list)
        return  env_step_return_tuple

    def main(self):

        train_rewards = []
        train_loss = []
        train_episodes_actions_average = []
        nmb_of_train_episodes = 600
        nmb_of_test_episodes = 100
        nmb_of_iterations_per_episode = 400
        nmb_of_episodes_before_training = 50
        type_of_epsilon_exp = "random"
        # 1 - epsilon decreases to 0.1 over nmb_of_train_episodes
        # 0.5 - epsilon decreases to 0.1 over nmb_of_train_episodes * 0.5 etc.
        epsilon_decreasing_factor = 0.09
        nmb_of_episodes_for_e_to_anneal = nmb_of_train_episodes * epsilon_decreasing_factor

        results = {}

        start_time_all = time.time()

        print("Action-space: {}".format(self.action_space))
        print("Number of inputs: {} and outputs: {}".format(self.nmb_of_inputs, self.nmb_of_outputs))

        self.model.summary()

        best_episode_reward_sum = 0

        for episode_nmb in range(nmb_of_train_episodes):
            obs = env.reset().to_numpy() # Reset environment
            train_episode_reward_sum = 0
            train_episode_action_sum = 0
            global q_values_mean
            global q_values_counter
            q_values_mean = 0
            q_values_counter = 0.0001
            train_episode_model_actions_count = {}
            start_time = time.time()

            episode_loss = 0
            for step in range(nmb_of_iterations_per_episode):
                epsilon = max(1 - episode_nmb / nmb_of_episodes_for_e_to_anneal, 0.1)
                obs, reward, action, action_type = self.play_one_step_and_collect_memory(env, obs, epsilon, type_of_epsilon_exp)

                obs = obs.to_numpy()
                if episode_nmb > nmb_of_episodes_before_training:
                    episode_loss += self.training_step(self.batch_size)
                if episode_nmb > nmb_of_episodes_before_training and (episode_nmb * nmb_of_iterations_per_episode + step) % 10000 == 0:
                    print("target_model.set_weights(model.get_weights())")
                    self.target_model.set_weights(self.model.get_weights())

                # statistics
                if action_type == "q":
                    if action in train_episode_model_actions_count:
                        train_episode_model_actions_count[action] += 1
                    else:
                        train_episode_model_actions_count[action] = 1
                train_episode_reward_sum += reward
                train_episode_action_sum += action

            train_rewards.append(train_episode_reward_sum)
            train_episode_action_average = train_episode_action_sum / nmb_of_iterations_per_episode
            train_episodes_actions_average.append(train_episode_action_average)
            if best_episode_reward_sum < train_episode_reward_sum and episode_nmb > nmb_of_episodes_for_e_to_anneal:
                best_weights = self.model.get_weights()
                best_episode_reward_sum = train_episode_reward_sum

                results["best_train_episode_reward_sum"] = train_episode_reward_sum
                results["best_train_episode_reward_sum_episode_nmb"] = episode_nmb
            print("Episode: {}, loss: {}, rewards: {}, q_value_mean: {} actions: {} eps: {:.3f} time: {}".format(episode_nmb,
                                                                                                                episode_loss/nmb_of_iterations_per_episode,
                                                                                                                train_episode_reward_sum,
                                                                                                                q_values_mean/q_values_counter,
                                                                                                                train_episode_action_average ,
                                                                                                                epsilon,
                                                                                                                time.time() - start_time), end="\n")
            train_loss.append(episode_loss/nmb_of_iterations_per_episode)
            print(train_episode_model_actions_count, end="\n")

        train_avg_rewards = sum(train_rewards) / len(train_rewards)

        print("average sum for training is: {}".format(train_avg_rewards))



        ### TEST for best model
        # best_score_model = keras.models.clone_model(self.model)
        # best_score_model.set_weights(best_weights)
        # test_rewards = []
        # for e in range(nmb_of_test_episodes):
        #     state = self.reset().to_numpy()
        #     test_episode_reward_sum = 0
        #     for step in range(nmb_of_iterations_per_episode):
        #         state, reward, action, action_type = self.epsilon_greedy_policy(best_score_model, env, state, action_space[1])
        #         state = state.to_numpy()
        #         test_episode_reward_sum += reward
        #     test_rewards.append(test_episode_reward_sum)
        #     print("Episode test for best: {}, rewards: {}, eps: {:.3f}".format(e, test_episode_reward_sum, epsilon), end="\n")

        # test_best_avg_rewards = sum(test_rewards) / len(test_rewards)

        # print("Average reward on BEST from best model on 100 runs {}".format(test_best_avg_rewards))

        # ### Test for last model
        # test_last_model_rewards = []

        # for e in range(nmb_of_test_episodes):
        #     state = env.reset().to_numpy()
        #     test_episode_reward_sum = 0
        #     for step in range(nmb_of_iterations_per_episode):
        #         state, reward, action, action_type = epsilon_greedy_policy(model, env, state, action_space[1])
        #         state = state.to_numpy()
        #         test_episode_reward_sum += reward
        #     test_last_model_rewards.append(test_episode_reward_sum)
        #     print("Episode test for last: {}, rewards: {}, eps: {:.3f}".format(e, test_episode_reward_sum, epsilon), end="\n")

        # test_last_model_avg_rewards = sum(test_last_model_rewards) / len(test_last_model_rewards)

        # print("Average reward on final LAST MODEL from best model on 100 runs {}".format(test_last_model_avg_rewards))

        # model.set_weights(best_weights)

        # results['nmb_of_train_episodes'] = nmb_of_train_episodes
        # results['nmb_of_iterations_per_episode'] = nmb_of_iterations_per_episode
        # results['nmb_of_episodes_before_training'] = nmb_of_episodes_before_training
        # results['nmb_of_episodes_for_e_to_anneal'] = nmb_of_episodes_for_e_to_anneal
        # results['memory_length'] = memory_length
        # results['batch_size'] = batch_size
        # results['type_of_epsilon_exp'] = type_of_epsilon_exp
        # results['discount_rate'] = discount_rate
        # results['nmb_of_test_episodes'] = nmb_of_test_episodes
        # results['loss_function'] = loss_fn.__name__

        # results['action_space'] = str(env.action_space)

        # results['train_avg_reward'] = train_avg_rewards
        # results['test_last_model_avg_rewards'] = test_last_model_avg_rewards
        # results['test_best_avg_reward'] = test_best_avg_rewards

        # results['train_episodes_actions_average'] = train_episodes_actions_average

        # results['train_rewards'] = train_rewards
        # results['train_loss'] = train_loss
        # results['test_rewards'] = test_rewards
        # results['test_last_model_rewards'] = test_last_model_rewards

        # save_model(best_score_model, model, results)

        # print("Whole process took: {}", time.time() - start_time_all)

    def save_model(best_score_model, last_model, parameters_dict):
        path = "results_" + datetime.now().strftime("%Y%m%d_%H_%M")
        os.makedirs(path)
        os.makedirs(path + "/conf")
        os.makedirs(path + "/json_models")
        os.makedirs(path + "/weights")

        best_score_model.save(path + "/best_model")
        last_model.save(path + "/last_model")

        with open(path + '/report.txt', 'w') as file:
            file.write(json.dumps(parameters_dict))

        best_score_model.save_weights(path + '/weights/best_weights.h5')
        last_model.save_weights(path + '/weights/last_weights.h5')

        with open(path + '/json_models/model_json.json', 'w') as file:
            file.write(best_score_model.to_json())
        with open(path + '/json_models/last_model_json.json', 'w') as file:
            file.write(last_model.to_json())

        plt.figure(figsize=(8, 4))
        plt.plot(parameters_dict['train_rewards'])
        plt.ylim(ymin=0)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Sum of rewards", fontsize=14)
        plt.savefig(path + "/dqn_rewards_plot_train.png", format="png", dpi=300)

        plt.figure(figsize=(8, 4))
        plt.plot(parameters_dict['test_rewards'])
        plt.ylim(ymin=0)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Sum of TEST rewards", fontsize=14)
        plt.savefig(path + "/dqn_rewards_plot_test_best.png", format="png", dpi=300)

        plt.figure(figsize=(8, 4))
        plt.plot(parameters_dict['test_last_model_rewards'])
        plt.ylim(ymin=0)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Sum of LAST MODEL rewards", fontsize=14)
        plt.savefig(path + "/dqn_rewards_plot_test_last.png", format="png", dpi=300)

        plt.figure(figsize=(8, 4))
        plt.plot(parameters_dict['train_loss'])
        plt.ylim(ymin=0)
        plt.xlabel("Episode", fontsize=14)
        plt.ylabel("Train loss", fontsize=14)
        plt.savefig(path + "/dqn_rewards_plot_training_loss.png", format="png", dpi=300)

        shutil.copytree('./conf', path + "/conf", dirs_exist_ok=True)


if __name__ == '__main__':
    class DDQN_train:
    def __init__(self, running_time, learning_rate, reward_function, gamma, epsilon, batch_size, config_type):

    running_time = 5000
    learning_rate = 0.001 # Original parameter
    reward_function = 'case'
    gamma = 1
    epsilon = 1 # epsilon linearly annealed from 1 to 0.1 over the first 10% of episodes
    batch_size = 1 
    config_type = 'slow_server'
    
    ddqn_train = DDQN_train(running_time, learning_rate, reward_function, gamma, epsilon, batch_size, config_type)
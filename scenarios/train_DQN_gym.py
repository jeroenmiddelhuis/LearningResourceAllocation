from collections import deque
from subprocess import call
import gymnasium as gym
import os
import numpy as np
from bpo_env import BPOEnv
import sys

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure

from gymnasium.wrappers import normalize
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from callbacks import SaveOnBestTrainingRewardCallback
from callbacks import custom_schedule, linear_schedule


nr_layers = 3
nr_neurons = 64
clip_fraction = 0.1
n_steps = 51200
batch_size = 256

net_arch = dict(pi=[nr_neurons for _ in range(nr_layers)], vf=[nr_neurons for _ in range(nr_layers)])

class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=net_arch)


if __name__ == '__main__':
    #if true, load model for a new round of training
    running_time = 5000
    num_cpu = 1
    load_model = False
    config_type= 'n_system'
    print(config_type)
    reward_function = 'cycle_time'
    postpone_penalty = 0
    time_steps = 2e7 # Total timesteps
    n_steps = 51200 # Number of steps for each network update
    # Create log dir
    log_dir = f"./tmp/{config_type}_{int(time_steps)}_{n_steps}/" # Logging training results

    os.makedirs(log_dir, exist_ok=True)

    print(f'Training agent for {config_type} with {time_steps} timesteps in updates of {n_steps} steps.')
    # Create and wrap the environment
    # Reward functions: 'AUC', 'case_task'
    env = BPOEnv(running_time=running_time, config_type=config_type, 
                reward_function=reward_function, postpone_penalty=postpone_penalty,
                write_to=log_dir)  # Initialize env
    env = Monitor(env, log_dir)

    resource_str = ''
    for resource in env.simulator.resources:
        resource_str += resource + ','
    with open(f'{log_dir}results_{config_type}.txt', "w") as file:
        # Writing data to a file
        file.write(f"uncompleted_cases,{resource_str}total_reward,mean_cycle_time,std_cycle_time\n")
 

    # Create the model
    model = MaskablePPO(CustomPolicy, env, clip_range=0.2, learning_rate=linear_schedule(3e-4), n_steps=int(n_steps), batch_size=batch_size, gamma=0.999, verbose=1) #

    #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
    # tensorboard --logdir ./tmp/
    # then, in a browser page, access localhost:6006 to see the board
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))


    # Train the agent
    eval_env = BPOEnv(running_time=running_time, config_type=config_type, 
                reward_function=reward_function, postpone_penalty=postpone_penalty,
                write_to=log_dir)  # Initialize env

    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                log_path=log_dir, eval_freq=int(n_steps),
                                deterministic=True, render=False)


    callback = SaveOnBestTrainingRewardCallback(check_freq=int(n_steps), log_dir=log_dir)
    model.learn(total_timesteps=int(time_steps), callback=callback)

    # For episode rewards, use env.get_episode_rewards()
    # env.get_episode_times() returns the wall clock time in seconds of each episode (since start)
    # env.rewards returns a list of ALL rewards. Of current episode?
    # env.episode_lengths returns the number of timesteps per episode
    print(env.get_episode_rewards())
    #     print(env.get_episode_times())

    model.save(f'{log_dir}/{config_type}_{running_time}_final')

    #import matplotlib.pyplot as plt
    #plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, f"{model_name}")
    #plt.show()
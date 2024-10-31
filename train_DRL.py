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

from callbacks import SaveOnBestTrainingRewardCallback, EvalPolicyCallback
from callbacks import custom_schedule, linear_schedule


# Input parameters
nr_layers = 2
nr_neurons = 128
clip_range = 0.2
n_steps = 25600
batch_size = 256
lr = 3e-05

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
    config_type= 'complete_reversed'# sys.argv[1]#'slow_server' # Config types as given in config.txt
    print(config_type)
    reward_function = 'cycle_time'
    arrival_rate = 'pattern'#float(sys.argv[2]) if sys.argv[2] != 'pattern' else sys.argv[2]
    postpone_penalty = 0
    time_steps = 5e7 # Total timesteps for training
    t_in_state = True
    #n_steps = 25600 # Number of steps for each network update
    # Create log dir
    log_dir = f"./tmp_t_in_state/{config_type}_{arrival_rate}/" # Logging training results

    os.makedirs(log_dir, exist_ok=True)

    print(f'Training agent for {config_type} with {time_steps} timesteps in updates of {n_steps} steps.')
    # Create and wrap the environment
    # Reward functions: 'AUC', 'case_task'
    env = BPOEnv(running_time=running_time, config_type=config_type, 
                reward_function=reward_function, postpone_penalty=postpone_penalty,
                write_to=log_dir, arrival_rate=arrival_rate, t_in_state=t_in_state)  # Initialize env
    env = Monitor(env, log_dir)

    resource_str = ''
    for resource in env.simulator.resources:
        resource_str += resource + ','
    with open(f'{log_dir}results_{config_type}.txt', "w") as file:
        # Writing data to a file
        file.write(f"uncompleted_cases,{resource_str}total_reward,mean_cycle_time,std_cycle_time\n")
 

    # Create the model
    model = MaskablePPO(CustomPolicy, env, clip_range=clip_range, learning_rate=linear_schedule(lr), n_steps=int(n_steps), batch_size=batch_size, gamma=0.999, verbose=1) #

    #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
    # tensorboard --logdir ./tmp/
    # then, in a browser page, access localhost:6006 to see the board
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))

    # Train the agent
    eval_env = BPOEnv(running_time=running_time, config_type=config_type, 
                reward_function=reward_function, postpone_penalty=postpone_penalty,
                write_to=None, arrival_rate=arrival_rate, t_in_state=t_in_state)  # Initialize env
    eval_env = Monitor(eval_env, log_dir)
    eval_callback = EvalPolicyCallback(check_freq=3*int(n_steps), nr_evaluations=10, log_dir=log_dir, eval_env=eval_env)
    best_reward_callback = SaveOnBestTrainingRewardCallback(check_freq=int(n_steps), log_dir=log_dir)


    model.learn(total_timesteps=int(time_steps))#, callback=eval_callback)#

    # For episode rewards, use env.get_episode_rewards()®
    # env.get_episode_times() returns the wall clock time in seconds of each episode (since start)
    # env.rewards returns a list of ALL rewards. Of current episode?
    # env.episode_lengths returns the number of timesteps per episode
    print(env.get_episode_rewards())
    #     print(env.get_episode_times())

    model.save(f'{log_dir}/{config_type}_{running_time}_final')

    #import matplotlib.pyplot as plt
    #plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, f"{model_name}")
    #plt.show()
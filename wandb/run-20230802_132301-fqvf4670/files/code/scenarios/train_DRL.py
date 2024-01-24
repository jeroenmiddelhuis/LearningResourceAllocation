from collections import deque
from subprocess import call
import gymnasium as gym
import os
import numpy as np
from bpo_env import BPOEnv
import sys

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy, MaskableMultiInputActorCriticPolicy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
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

import wandb
from wandb.integration.sb3 import WandbCallback



class CustomPolicy(MaskableActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128],
                                                          vf=[128])])


if __name__ == '__main__':
    #if true, load model for a new round of training
    nr_of_episodes = 100
    running_time = 300
    num_cpu = 1
    load_model = False
    model_name = "ppo_masked"
    config_type= 'complete'#sys.argv[1]
    print(config_type)
    reward_function = 'AUC'
   
    check_interval = 0.01
    n_steps = int(running_time/check_interval) # Number of steps for each network update
    time_steps = n_steps * nr_of_episodes # Total timesteps 
    # Create log dir
    log_dir = f"./scenarios/tmp/{config_type}_{time_steps}_{check_interval}/" # Logging training results

    os.makedirs(log_dir, exist_ok=True)

    print(f'Training agent for {config_type} with {time_steps} timesteps in updates of {n_steps} steps.')
    # Create and wrap the environment
    # Reward functions: 'AUC', 'case_task'
    env = BPOEnv(running_time=running_time, config_type=config_type, 
                 reward_function=reward_function, check_interval=check_interval,
                 write_to=log_dir, interval_mode=True)  # Initialize env
    env = Monitor(env, log_dir)  

    # resource_str = ''
    # for resource in env.simulator.resources:
    #     resource_str += resource + ','
    # with open(f'{log_dir}results_{config_type}.txt', "w") as file:
    #     # Writing data to a file
    #     file.write(f"uncompleted_cases,{resource_str}total_reward,mean_cycle_time,std_cycle_time\n")
 

    config = {
        "policy_type": "MaskableActorCriticPolicy",
        "total_timesteps": time_steps,
        "env_name": config_type,
    }
    run = wandb.init(
        project="sb3",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )


    # Create the model
    model = MaskablePPO(MaskableActorCriticPolicy, env, learning_rate=0.0001, n_steps=int(n_steps), clip_range=0.1, gamma=1, verbose=1, tensorboard_log=f'{log_dir}/tensorboard/')
    #learning_rate=linear_schedule(0.0001)
    #Logging to tensorboard. To access tensorboard, open a bash terminal in the projects directory, activate the environment (where tensorflow should be installed) and run the command in the following line
    # tensorboard --logdir ./tmp/
    # then, in a browser page, access localhost:6006 to see the board
    model.set_logger(configure(log_dir, ["stdout", "csv", "tensorboard"]))


    # Train the agent
    callback = SaveOnBestTrainingRewardCallback(check_freq=int(n_steps), log_dir=log_dir)
    model.learn(total_timesteps=int(time_steps), callback=WandbCallback())

    # For episode rewards, use env.get_episode_rewards()
    # env.get_episode_times() returns the wall clock time in seconds of each episode (since start)
    # env.rewards returns a list of ALL rewards. Of current episode?
    # env.episode_lengths returns the number of timesteps per episode
    print(env.get_episode_rewards())
    #     print(env.get_episode_times())

    model.save(f'{log_dir}/{model_name}_{time_steps}_{check_interval}_final')

    #import matplotlib.pyplot as plt
    #plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, f"{model_name}")
    #plt.show()
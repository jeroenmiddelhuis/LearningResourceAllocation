import os

import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from typing import Callable

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, model_name: str = 'best_model', verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, model_name)
        self.best_mean_reward = -np.inf
        self.prev_steps = 0
        self.episode_lengths = []

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            #print(x, y)
            # if len(x) > len(self.episode_lengths):
            #     if len(x) == 1:
            #         self.episode_lengths.append(x[0])
            #     else:
            #         self.episode_lengths.append(x[-1]-x[-2])
            #print(x,y, self.episode_lengths)
            if len(x) > 5:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-5:])
                # nr_eps = 5
                # mean_reward = np.mean([y[-i]/self.episode_lengths[-i] for i in range(len(y[-nr_eps:]))]) 
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

            return True

def custom_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if progress_remaining > 0.8: #0.95
            return initial_value
        elif progress_remaining > 0.5: #0.9
            return initial_value * 0.5
        else:
            return initial_value * 0.1

    return func


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

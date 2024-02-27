# LearningResourceAllocation

This repository contains the supplementary code for the paper "Learning policies for resource allocation in business processes" (Middelhuis et al., 2024)

To use the code, run the following commands to create an environment and install relevant libraries:
*```conda create -n drl_env python=3.11.4```
*```pip install gymnasium==0.29.0```
*```pip install sb3_contrib==2.0.0```
*```pip install tensorboard==2.15.0```

Alternatively, you can use the `requirements.txt` file to install the appropriate libraries.

The following files are used to train and evaluate the models:
* To train the DRL model, use `train_DRL.py`.
* To train the SVFA method, use `train_SVFA.py`.
* To evaluate the models, use `__main__.py` and select your planner (i.e., model or heuristic).
* For some of the benchmarks, copies of these existing files were used. To run these, different libraries may need to be installed.
It is possible to create your own planner function to interact with the simulation. To do this, inherit the Planner class in `planners.py` and implement the `plan()` function. This function takes as input the simulation variables `available_tasks`, `available_resources` and `resource_pools`, and should return a list of (resource, task) assignments.

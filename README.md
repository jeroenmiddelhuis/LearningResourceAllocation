# LearningResourceAllocation

This repository contains the supplementary code for the paper "Learning policies for resource allocation in business processes" (Middelhuis et al., 2023)

The Python version used was 3.9.7. To create an environment and install required packages, use the following commands:
* ```python -m venv env .\env\Scripts\activate``` 
* ```python -m pip install -r requirements.txt```

The following files are used to train and evaluate the models:
* To train the DRL model, use `train_DRL.py`.
* To train the SVFA method, use `train_SVFA.py`.
* To evaluate the models, use `__main__.py` and select your planner (i.e., model or heuristic).

It is possible to create your own planner function to interact with the simulation. To do this, inherit the Planner class in `planners.py` and implement the `plan()` function. This function takes as input the simulation variables `available_tasks`, `available_resources` and `resource_pools`, and should return a list of (resource, task) assignments.

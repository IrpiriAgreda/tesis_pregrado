# Delivery Time RL Agent

This repository is focused on developing a RL agent that looks learn how to provide delivery time regulations for a city zone.

## Main dependencies

* gym (optional, a custom env is implemented but the main tests don't use it)
* numpy
* pytorch
* sumo (and traci for agent interaction with the simulation)

This repository will include a DQN and Policy Gradient implementation of the task, these are adapted from Pytorch's DQN implementation and Machine Learning with Phil's implementation.

## Scenario parameters

The enviroment is divided into three scenarios with the following parameters:

* Total actions: 5981
* Number of trucks: 500, 1000, 2000
* Number of cars: 800, 1600, 3200
* Number of buses/motorbikes: 250, 500, 1000

The training scenario is based on Lima's historic center, the test scenario is generated within a residential zone. This repo does include three test scenarios that change the probability distribution, sampled with a Cauchy, Gaussian and Chi 2 PDF.

## Usage

To train the agents, build the scenarios by running build.bat to have config files in each of the km2 centro scenario folders. When the files are generated, run the Euro standard assignment notebook to have the adequate proportions and emission metrics for training.

Two training scripts are provided (dqn and ac_pg) which train the agents por each algorithm. **PLEASE CHANGE THE SUMO PATH ACCORDINGLY** by default it is set to '/usr/bin/sumo/bin/sumo', but it may need to change according to your needs. A set of pre-trained weights are provided, only optimizer states are provided for policy gradient options. 

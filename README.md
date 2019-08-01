# Delivery Time RL Agent

This repository is focused on developing a RL agent that looks learn how to provide delivery time regulations for a city zone.

## Main dependencies

* gym
* numpy
* tfp-nightly
* tensorflow-gpu==2.0.0-alpha0
* tf-agents-nightly-gpu

This repository will include a DQN (or DDQN or A3C) implementation over a custom OpenAI Gym environment.

## Scenario parameters

The simulation scenarios were implemented using SUMO's OSM WebWizard, using the parameters:

* Default zoom level
* 1000 cars
* 650 trucks
* 50 buses
* 200 motorcycles

In total, 26 scenarios were generated for the agent to generalize

#!/usr/bin/env python
# coding: utf-8

import math
import random
import warnings
import os
import sys
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import _pickle as cPickle
from collections import namedtuple
from gc import collect

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

warnings.filterwarnings('ignore')

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):

    def __init__(self, outputs, fc_2=False, fc_3=False):
        super(DQN, self).__init__()
        self.mlp1 = nn.Linear(6,32)
        self.use_mlp2 = fc_2
        self.use_mlp3 = fc_3
        
        if self.use_mlp2:
            self.mlp2 = nn.Linear(32,64)
            if self.use_mlp3:
                self.mlp3 = nn.Linear(64,128)
                self.head = nn.Linear(128, outputs)
            else:
                self.head = nn.Linear(64, outputs)
        else:
            self.head = nn.Linear(32, outputs)

    def forward(self, x):
        x = F.relu(self.mlp1(x))
        
        if self.use_mlp2:
            
            x = F.relu(self.mlp2(x))
            
            if self.use_mlp3:
                x = F.relu(self.mlp3(x))

        return F.relu(self.head(x.view(x.size(0), -1)))

BATCH_SIZE = 64
GAMMA = 0.001
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
n_actions = 5981
steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) *         math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.view([-1,6]))
    else:
        return torch.randn(n_actions, device=device)


episode_durations = []

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state).view(-1,6)
    #action_batch = torch.cat(batch.action).view(-1,n_actions)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)#.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros((BATCH_SIZE, 5981), device=device)

    nsv = target_net(non_final_next_states.view(-1,6)).detach()

    next_state_values[nsv > 0] = nsv[nsv > 0]

    # Compute the expected Q values
    expected_state_action_values = ((next_state_values * GAMMA).t() + reward_batch).t()

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    collect()

def run_experiments(num_episodes, date, exp_name, simconfig, layers):
    #Main parameters
    current_ep = 0
    os.mkdir('./output_weights/policy/{}'.format(exp_name))
    os.mkdir('./output_weights/target/{}'.format(exp_name))

    if simconfig == 0:
        sumoCmd = ['/usr/bin/sumo',
             '-c',
             '../sumo_simulation/sim_config/km2_centro/scenario/osm.sumocfg',
             '-e', '86400']
    else:
        sumoCmd = ["/usr/bin/sumo", "-c", 
                   "../sumo_simulation/sim_config/km2_centro/scenario_{}/osm.sumocfg".format(simconfig),
                  '-e', '86400']

    action_dict = cPickle.load(open('../sumo_simulation/input/action_to_zone_km2_centro.pkl', 'rb'))


    with open('output/rewards_gamma001_experiment_{}_layers_{}'.format(date,layers), 'a') as reward_file:
        for i_episode in range(num_episodes):
            print("Episode {}".format(i_episode))

            state = np.zeros(6)
            reward = 0
            done = 0

            #Start simulation
            traci.start(sumoCmd)
            id_list = traci.edge.getIDList()
            lane_id_list = traci.lane.getIDList()

            #Run simulation steps
            traci_ep = 0
            for w in range(86400):
                if traci_ep >= 2*86400:
                    reward -= 800000
                    break

                if traci_ep % 3600 == 0 and traci_ep != 0:

                    #Start agent interaction
                    action = select_action(torch.tensor(state, device=device, dtype=torch.float))

                    #Apply regulation and run steps
                    reg_action = action > 0
                    #lane_indices = (reg_action == 1).nonzero().view(-1)

                    for index, lane_id in enumerate(reg_action.view(-1)):
                    #for lane_id in lane_indices:
                        if lane_id.item() == 1:
                            if action_dict[index] is not None:
                                traci.lane.setDisallowed(action_dict[index], ['truck'])
                            else:
                                reward -= 10000
                        else:
                            if action_dict[index] is not None:
                                traci.lane.setAllowed(action_dict[index], ['truck'])
                            else:
                                pass

                    #Get simulation values
                    co2 = [traci.lane.getCO2Emission(edge_id) for edge_id in lane_id_list]
                    co = [traci.lane.getCOEmission(edge_id) for edge_id in lane_id_list]
                    nox = [traci.lane.getNOxEmission(edge_id) for edge_id in lane_id_list]
                    pmx = [traci.lane.getPMxEmission(edge_id) for edge_id in lane_id_list]
                    noise = [traci.lane.getNoiseEmission(edge_id) for edge_id in lane_id_list]
                    fuel = [traci.lane.getFuelConsumption(edge_id) for edge_id in lane_id_list]

                    sim_results = np.array([co2, co, pmx, nox, noise, fuel])

                    next_state = np.transpose(sim_results).mean(axis=0)

                    vehicle_id_list = traci.vehicle.getIDList()
                    vehicle_types = [traci.vehicle.getTypeID(v_id) for v_id in vehicle_id_list]
                    vehicle_co2 = [traci.vehicle.getCO2Emission(v_id) for i, v_id in enumerate(vehicle_id_list)
                                  if 'truck' in vehicle_types[i]]

                    try:
                        reward += (sum(vehicle_co2)/len(vehicle_co2)) * -1
                    except:
                        reward += 0

                    #Convert to torch tensors
                    reward = torch.tensor([reward], device=device, dtype=torch.float)
                    state = torch.tensor(state, device=device, dtype=torch.float)
                    next_state = torch.tensor(next_state, device=device, dtype=torch.float)

                    print('Reward at timestep {t}: {r}'.format(t=traci_ep/3600,r=reward.item()))
                    reward_file.write(','.join([str(i_episode), str(traci_ep/3600), str(reward.item())])+r'\n')


                    action = torch.tensor(action, device=device, dtype=torch.long).view(-1)
                    action = (action == 1)#.nonzero().view(-1)

                    if state.shape[0] == 0:
                        memory.push(torch.zeros((1,12), device=device), action, next_state, reward)
                    else:
                        memory.push(state, action, next_state, reward)

                    state += next_state
                    # Perform one step of the optimization (on the target network)
                    optimize_model()

                traci.simulationStep()
                traci_ep += 1

            traci.close(False)

             # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                 target_net.load_state_dict(policy_net.state_dict())

            torch.save(target_net.state_dict(), './output_weights/target/{}/target_net_weights_experiment_ep_{}.pt'.format(exp_name, i_episode))
            torch.save(policy_net.state_dict(), './output_weights/policy/{}/policy_net_weights_experiment_ep_{}.pt'.format(exp_name, i_episode))
        print('Complete')
        torch.save(target_net.state_dict(), './output_weights/target/{}/target_net_weights_experiment_ep_{}.pt'.format(exp_name, i_episode))
        torch.save(policy_net.state_dict(), './output_weights/policy/{}/policy_net_weights_experiment_ep_{}.pt'.format(exp_name, i_episode))


today = str(datetime.datetime.today())[:-7]
simconfig = 0

for layers in [1,2,3]:
    
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs")
        
        if layers == 2:
            policy_net = DQN(n_actions, fc_2=True, fc_3=False).to(device)
            target_net = DQN(n_actions, fc_2=True, fc_3=False).to(device)
        elif layers == 3:
            policy_net = DQN(n_actions, fc_2=True, fc_3=True).to(device)
            target_net = DQN(n_actions, fc_2=True, fc_3=True).to(device)
	else:
	    policy_net = DQN(n_actions, fc_2=True, fc_3=True).to(device)
            target_net = DQN(n_actions, fc_2=True, fc_3=True).to(device)
		
            
        policy_net = nn.DataParallel(policy_net)
        target_net = nn.DataParallel(target_net)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

    else:
        print('Using 1 GPU')
        if layers == 2:
            policy_net = DQN(n_actions, fc_2=True, fc_3=False).to(device)
            target_net = DQN(n_actions, fc_2=True, fc_3=False).to(device)
        else:
            policy_net = DQN(n_actions, fc_2=True, fc_3=True).to(device)
            target_net = DQN(n_actions, fc_2=True, fc_3=True).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)
    for i in [0,2,3]:
        run_experiments(3000, today, 'km2_centro_test_day_scenario_{}_layer_{}_date_{}'.format(i, layers, today), i, layers)


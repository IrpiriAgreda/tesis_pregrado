#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import math
import random
import numpy as np
import city_simulation
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import warnings
import os
import datetime
from collections import namedtuple
from itertools import count
from gc import collect


# %matplotlib inline
# output_notebook()

# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


step_fn = lambda x: 1 if x > 0.5 else 0


# In[4]:


step_f = np.vectorize(step_fn)


# In[5]:


env = gym.make('city_simulation-v0').unwrapped


# In[6]:


# if gpu is to be used
#device = torch.device('cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[7]:


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


# In[8]:


class DQN(nn.Module):

    def __init__(self, outputs):
        super(DQN, self).__init__()

#         self.conv1 = nn.Conv2d(1, 32, kernel_size=2, padding=1)
#         self.mp1 = nn.MaxPool2d(kernel_size=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=2, padding=1)
#         self.mp2 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=2, padding=1)
#         self.mp3 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
        self.mlp1 = nn.Linear(12,32)
#         self.mlp2 = nn.Linear(32,64)
#         self.mlp3 = nn.Linear(64,128)
#         self.mlp4 = nn.Linear(128,256)
#         self.mlp5 = nn.Linear(256,256)
        self.head = nn.Linear(32, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
#         x = F.relu(self.bn1(self.mp1(self.conv1(x))))
#         x = F.relu(self.bn2(self.mp2(self.conv2(x))))
        #x = F.relu(self.bn3(self.mp3(self.conv3(x))))
        x = F.relu(self.mlp1(x))
#         x = F.relu(self.mlp2(x))
#         x = F.relu(self.mlp3(x))
#         x = F.relu(self.mlp4(x))
#         x = F.relu(self.mlp5(x))
    
        return self.head(x.view(x.size(0), -1))


# In[9]:


BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


# In[10]:


# Get number of actions from gym action space
n_actions = 46114


# In[11]:


policy_net = DQN(n_actions).to(device)
target_net = DQN(n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()


# In[12]:


optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(500)


# In[13]:


steps_done = 0


# In[14]:


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
            return policy_net(state.view([-1,12]))
    else:
        return torch.randn(n_actions, device=device)


# In[15]:


episode_durations = []


# In[16]:


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    state_batch = torch.cat(batch.state)#.view(-1,1,12)
    #action_batch = torch.cat(batch.action).view(-1,n_actions)
    reward_batch = torch.cat(batch.reward)
    
    #print(state_batch.shape, action_batch.shape)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch)#.gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    nsv = target_net(non_final_next_states).max(1)[0].detach()
    
    #print(state_action_values.shape, nsv.shape, next_state_values.shape, non_final_mask.shape)
    
    next_state_values[non_final_mask] = nsv
    
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    collect()


# In[17]:


#p = figure(title="Training Reward per Timestep", plot_height=350, plot_width=800)
#target = show(p, notebook_handle=True)
current_ep = 0

with open('output/rewards_experiment_{}'.format(str(datetime.datetime.today())[:-7]), 'a') as reward_file:

    num_episodes = 400
    for i_episode in range(num_episodes):
        print("Episode {}".format(i_episode))
        # Initialize the environment and state
        env.reset()  
        for t in count():
            current_ep += 1
            state = env.render()
            action = select_action(torch.tensor(state, device=device, dtype=torch.float))
            next_state, reward, done, _ = env.step(action > 0.5)
            reward = torch.tensor([reward], device=device)
            state = torch.tensor(state, device=device, dtype=torch.float)
            next_state = torch.tensor(next_state, device=device, dtype=torch.float)

            print('Reward at timestep {t}: {r}'.format(t=t,r=reward.item()))
            
            reward_file.write(','.join([str(i_episode), str(current_ep), str(reward.item())])+r'\n')
            #rewards.append(reward.item())
            #episodes.append(current_ep)

            #p.line(episodes, rewards)
            #push_notebook(handle=target)

            if state.view(-1).shape == 12:
                state = state.view(-1)
                next_state = next_state.view(-1)
            else:
                state = state.view(-1,12)
                next_state = next_state.view(-1,12)

            action = torch.tensor(action, device=device, dtype=torch.long).view(-1)
            action = (action == 1).nonzero().view(-1)

            if state.shape[0] == 0:
                memory.push(torch.zeros((1,12), device=device), action, next_state, reward)
            else:
                memory.push(state, action, next_state, reward)

            # Perform one step of the optimization (on the target network)
            optimize_model()
            collect()
            
            if done:
                break
                collect()
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())
            collect()
            
        torch.save(target_net.state_dict(), './output_weights/target/target_net_weights_{}_ep_{}.pt'.format(str(datetime.datetime.today())[:-7], i_episode))
        torch.save(policy_net.state_dict(), './output_weights/policy/policy_net_weights_{}_ep_{}.pt'.format(str(datetime.datetime.today())[:-7], i_episode))

print('Complete')
env.render()
env.close()
torch.save(target_net.state_dict(), './output_weights/target_net.pt')
torch.save(policy_net.state_dict(), './output_weights/policy_net.pt')


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys
import datetime
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import _pickle as cPickle
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


n_actions = 5981


# In[4]:


device = T.device("cuda" if T.cuda.is_available() else "cpu")


# In[5]:


class ActorCriticNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, fc3_dims,
                 n_actions):
        super(ActorCriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
            
        if type(self.fc2_dims) != bool:
            self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
                
            if type(self.fc3_dims) != bool:
                self.fc3 = nn.Linear(self.fc2_dims, self.fc3_dims)
                    
                self.pi = nn.Linear(self.fc3_dims, n_actions)
                self.v = nn.Linear(self.fc3_dims, n_actions)
            else:
                self.pi = nn.Linear(self.fc2_dims, n_actions)
                self.v = nn.Linear(self.fc2_dims, n_actions)
        else:
            self.pi = nn.Linear(self.fc1_dims, n_actions)
            self.v = nn.Linear(self.fc1_dims, n_actions)
            
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = device
        self.to(self.device)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        
        if type(self.fc2_dims) != bool:
            x = F.relu(self.fc2(x))
                
            if type(self.fc3_dims) != bool:
                x = F.relu(self.fc3(x))
        
        pi = self.pi(x)
        v = self.v(x)
        return (pi, v)


# In[6]:


class Agent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments such as
        the discrete lunar lander
    """
    def __init__(self, alpha, input_dims, gamma=0.001,
                 layer1_size=32, layer2_size=64,layer3_size=128, n_actions=n_actions):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, layer3_size, n_actions)

        self.log_probs = None

    def choose_action(self, observation):
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities)
        action_probs = T.distributions.Categorical(probabilities)
        action = action_probs.sample()
        #log_probs = action_probs.log_prob(action)
        log_probs = action_probs.log_prob(T.tensor(range(5981), device=device))
        self.log_probs = log_probs

        return action_probs.probs

    def learn(self, state, reward, new_state, done):
        self.actor_critic.optimizer.zero_grad()

        _, critic_value_ = self.actor_critic.forward(new_state)
        _, critic_value = self.actor_critic.forward(state)

        delta = reward + self.gamma*critic_value_*(1-int(done)) - critic_value

        actor_loss = -self.log_probs * delta
        critic_loss = delta**2
        
        (actor_loss + critic_loss).sum().backward()

        self.actor_critic.optimizer.step()


# In[7]:


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import traci
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


# In[8]:


sumoCmd = ["/usr/bin/sumo/bin/sumo", "-c", "../sumo_simulation/sim_config/km2_centro/scenario/osm.sumocfg"]


# In[9]:


def run_experiments(num_episodes, date, exp_name, simconfig, layers, agent):
    #Main parameters
    current_ep = 0
    
    if not os.path.exists('./output_weights/policy/{}'.format(exp_name)):
        os.mkdir('./output_weights/policy/{}'.format(exp_name))
    if not os.path.exists('./output_weights/target/{}'.format(exp_name)):
        os.mkdir('./output_weights/target/{}'.format(exp_name))

    if simconfig == 0:
        sumoCmd = ["/usr/bin/sumo/bin/sumo", "-c", "../sumo_simulation/sim_config/km2_centro/scenario/osm.sumocfg"]
    else:
        sumoCmd = ["/usr/bin/sumo/bin/sumo", "-c", "../sumo_simulation/sim_config/km2_centro/scenario_{}/osm.sumocfg".format(simconfig)]

    action_dict = cPickle.load(open('../sumo_simulation/input/action_to_zone_km2_centro.pkl', 'rb'))


    with open('output/rewards_gamma001_experiment_sc{}_layers_{}_{}'.format(simconfig, layers, date), 'a') as reward_file:
        for i_episode in range(num_episodes):
            #print("Episode {}".format(i_episode))

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

                if traci_ep % 3600 == 0 and traci_ep != 0:
                    state = T.tensor(state, device=device, dtype=T.float)
                    
                    #Start agent interaction
                    action = agent.choose_action(state)

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
                    reward = T.tensor([reward], device=device, dtype=T.float)
                    state = T.tensor(state, device=device, dtype=T.float)
                    next_state = T.tensor(next_state, device=device, dtype=T.float)

                    #print('Reward at timestep {t}: {r}'.format(t=traci_ep/3600,r=reward.item()))
                    reward_file.write(','.join([str(i_episode), str(traci_ep/3600), str(reward.item())])+'\n')


                    action = T.tensor(action, device=device, dtype=T.long).view(-1)
                    
                    #Optimize agent
                    agent.learn(state, reward, next_state, done)
                    
                    state += next_state

                traci.simulationStep()
                traci_ep += 1

            traci.close(False)

            T.save(agent.actor_critic.state_dict(), './output_weights/policy/{}/ac_weights_experiment_ep_{}.pt'.format(exp_name, i_episode))
            T.save(agent.actor_critic.optimizer.state_dict(), './output_weights/policy/{}/ac_optimizer_experiment_ep_{}.pt'.format(exp_name, i_episode))
        print('Complete')
        T.save(agent.actor_critic.state_dict(), './output_weights/policy/{}/ac_weights_experiment_ep_{}.pt'.format(exp_name, i_episode))
        T.save(agent.actor_critic.optimizer.state_dict(), './output_weights/policy/{}/ac_optimizer_experiment_ep_{}.pt'.format(exp_name, i_episode))


# In[10]:


from tqdm import tqdm


# In[11]:


today = str(datetime.datetime.today())[:10]

for exp in tqdm([0,2,3]):
    for layers in tqdm([1,2,3]):
        if layers == 1:
            agent = Agent(alpha=0.001, input_dims=[6], gamma=0.001,
                  n_actions=n_actions, layer2_size=False, layer3_size=False)
        elif layers == 2:
            agent = Agent(alpha=0.001, input_dims=[6], gamma=0.001,
                  n_actions=n_actions, layer3_size=False)
        else:
            agent = Agent(alpha=0.001, input_dims=[6], gamma=0.001,
                  n_actions=n_actions)
        
        run_experiments(1500, today, 'km2_centro_pg_scenario_{}_layers_{}_date_{}'.format(exp, layers, today), exp, layers, agent)


# In[ ]:





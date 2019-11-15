import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

class Agent(object):
    """ Agent class for use with a single actor critic network that shares
        the lowest layers. For use with more complex environments such as
        the discrete lunar lander
    """
    def __init__(self, alpha, input_dims, n_actions, gamma=0.001,
                 layer1_size=32, layer2_size=64,layer3_size=128):
        self.gamma = gamma
        self.actor_critic = ActorCriticNetwork(alpha, input_dims, layer1_size,
                                    layer2_size, layer3_size, n_actions)

        self.log_probs = None

    def choose_action(self, observation):
        probabilities, _ = self.actor_critic.forward(observation)
        probabilities = F.softmax(probabilities)
        action_probs = torch.distributions.Categorical(probabilities)
        action = action_probs.sample()
        #log_probs = action_probs.log_prob(action)
        log_probs = action_probs.log_prob(torch.tensor(range(5981), device=device))
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

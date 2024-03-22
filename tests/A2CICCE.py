from ICCE.interfaces import ICCEInterface

import numpy as np
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import csv

SavedAction = namedtuple('SavedAction', ('log_prob', 'value'))

# Create class which implements both actor and critic as ONE model
class Policy(nn.Module):
    def __init__(self, n_observations, n_actions, n_hidden):
        super().__init__()
        self.fc1 = nn.Linear(n_observations, n_hidden) # Input layer
        self.actor_layer = nn.Linear(n_hidden, n_actions) # Actor's output layer
        self.critic_layer = nn.Linear(n_hidden, 1) # Critic's output layer - outputs value

        # Action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """forward pass for both actor and critic"""
        x = F.relu(self.fc1(x))

        # Actor: chooses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.actor_layer(x), dim=-1)

        # Critic: evaluates being in the state s_t
        state_values = self.critic_layer(x)

        # return values for both actor and critic as a tuple over 2 values
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

class A2CICCE(ICCEInterface):
    def __init__(self):
        super().__init__(frequency_hz=60)
        self.n_observations = 30
        self.n_actions = 4

        # Instantiate model and optimizer
        self.model = Policy(self.n_observations, 2**5, 128)
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-2)
        self.loss = None
        # Load from checkpoint
        checkpoint = torch.load('models/a2c_18_mar_24_E10k')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss = checkpoint['loss']
        
        # CONSTANTS
        self.eps = np.finfo(np.float32).eps.item()
        self.GAMMA = 0.99

        # Reward tracking
        self.ep_reward = 10
        self.running_reward = 10
        self.i_episode = 0
        self.all_returns = []

        # Set as training mode
        self.model.train()

    def post_sample(self, observation: np.ndarray, reward: float):
        self.model.rewards.append(reward)
        self.ep_reward += reward

    def post_episode(self):
        self.running_reward = 0.05 * self.ep_reward + (1 - 0.05) * self.running_reward

        self.finish_episode()
        
        self.all_returns.append(self.ep_reward)
        print(f'Episode {self.i_episode}\tLast reward: {self.ep_reward}\tAverage reward: {self.running_reward}')
        
        if self.i_episode == 999:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
            }, 'a2c_22_mar_24_E1000')

            csv_file = "a2c_22_mar_24_returns_E1000.csv"
            with open(csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(self.all_returns)

        
        
        self.i_episode += 1
        self.ep_reward = 10


    def act(self, observation: np.ndarray) -> np.ndarray:
        # Use policy to select discrete action
        discrete_action = self.select_action(observation.astype(dtype=np.float32))

        # output action
        action = np.zeros(shape=(4), dtype=np.float32)

        # Map key to action
        if discrete_action & (1 << 0): # Left
            action[0] = -1.0
        if discrete_action & (1 << 1): # Right
            action[0] = 1.0
        if discrete_action & (1 << 2): # Up
            action[1] = 1.0
        if discrete_action & (1 << 3): # Down
            action[1] = -1.0
        if discrete_action & (1 << 4): # Space
            action[3] = 1.0

        return action
    

    """UTILITIES"""
    def select_action(self, state):
        """Utility to select an action"""
        state = torch.from_numpy(state).float()
        probs, state_value = self.model(state)

        # Create a categorical distribution over the list of action probabilities
        m = Categorical(probs)

        # Sample it using the distribution
        action = m.sample()

        # Save the chosen action to action buffer
        self.model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        # The actual action to take (one of 32 combinations)
        return action.item()
    
    def finish_episode(self):
        """ Training code. Calculates actor and critic loss and performs backprop. """
        R = 0
        saved_actions = self.model.saved_actions
        policy_losses = [] # list to save actor loss (policy)
        value_losses = [] # list to save critic loss (value)
        returns = [] # list to save the true values

        # Calculate the true value using rewards returned by the environment
        for r in self.model.rewards[::-1]:
            # Calculate discounted reward
            R = r + self.GAMMA * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.optimizer.zero_grad()

        # sum all policy_losses and value_losses
        self.loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # backprop
        self.loss.backward()
        self.optimizer.step()

        # reset rewards and action buffer
        del self.model.rewards[:]
        del self.model.saved_actions[:]
        

def main():
    icce = A2CICCE()
    icce.run()

if __name__ == '__main__':
    main()
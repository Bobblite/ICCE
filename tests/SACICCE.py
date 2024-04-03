from ICCE.interfaces import ICCEInterface

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import csv
#from utils import plot_rewards, visualize_learning, plot_entropy

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64,64)
        self.mean = nn.Linear(64, output_dim)
        self. log_std = nn.Linear(64, output_dim)
        nn.init.constant_(self.log_std.weight, -0.5)

    def forward(self, x):                    
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        
        mean[torch.isnan(mean)] = 0.0
        log_std[torch.isnan(log_std)] = 0.0 

        return mean, log_std
    
class Critic(nn.Module):
    def __init__(self, input_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64,64)
        self.value = nn.Linear(64, 1)
        
    def forward(self, x):
        x =F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.value(x)
        return value
    
class SACAgent:
    def __init__(self, obs_dim, action_dim):
        self.actor = Actor(obs_dim, action_dim)
        self.critic = Critic(obs_dim)
        self.target_critic = Critic(obs_dim) 
        
        self.actor_optim = optim.Adam(self.actor.parameters(), lr = 0.001)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=0.001)
        self.target_entropy = -action_dim
        
        self.actor_loss = None
        self.critic_loss = None 
    
    def select_action(self, state):
        state = torch.FloatTensor(np.copy(state)).unsqueeze(0)
        mean, log_prob = self.actor(state)
        std = torch.exp(log_prob)
        std = torch.clamp(std, min=1e-8, max=float('inf'))
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z).detach().numpy()
        return action
    
    def learn(self, states, actions, rewards, next_states, done):
        # Convert lists to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(done)) 
        # Handle NaN values in observations 
        states[torch.isnan(states)] = 0
        next_states[torch.isnan(next_states)] = 0

        # Critic Loss 
        # Initialize critic_loss
        critic_loss = torch.tensor(0.0)
        # Iterate over each timestep
        for i in range(len(states)):
            state = states[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]

            # Compute target value using the soft update of the target critic network
            target_value = reward + (1 - done) * 0.99 * self.target_critic(next_state)

            # Compute predicted value using the current critic network
            predicted_value = self.critic(state)    

            # Accumulate squared errors
            critic_loss += F.mse_loss(predicted_value, target_value) 

        # Take the mean over all timesteps
        critic_loss = critic_loss / len(states) 
        self.critic_loss = critic_loss

        # Actor Loss
        mean, log_std = self.actor(states) 
        std = torch.exp(log_std)
        std = torch.clamp(std, min=1e-8, max=float('inf'))
        normal = torch.distributions.Normal(mean, std)
        z = normal.sample()
        log_prob = normal.log_prob(z)
        entropy = normal.entropy().mean()
        actor_loss = (self.target_entropy * log_prob - critic_loss).mean()
        self.actor_loss = actor_loss 

        # Update networks
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # print("Actor state_dict:", self.actor.state_dict())

        # Soft Target Update
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)

        return entropy.item()


            
class SACEICCE(ICCEInterface):
    def __init__(self):
        super().__init__(frequency_hz=120)
        self.agent = None
        self.n_observations = 30
        self.n_actions = 4 
         
        #Store total rewards per episode to be graphed
        self.rewards_per_epi = [] 
        self.my_epi = 0 

        self.current_obs_list =[]
        self.action_list=[]
        self.cumulative_rewards_list = []
        self.next_obs_list=[]
        self.done_list=[]

        self.entropy=[]
        
    def post_sample(self, observation:np.ndarray, reward:float):
        self.next_obs_list.append(observation)
        self.cumulative_rewards_list.append(reward)
        self.done_list.append(False)
        
    def post_episode(self):   
        # Displays the rewards and plot
        self.my_epi+=1  
 
        print("At EPI : ", self.my_epi)    
        self.process_list()
        entropy = self.agent.learn(self.current_obs_list, self.action_list, self.cumulative_rewards_list, self.next_obs_list, self.done_list)
        self.entropy.append(entropy) 
        
        total_sum = sum(self.cumulative_rewards_list)
        print("Total Reward per epi: ", total_sum) 
        self.rewards_per_epi.append(total_sum)  

        # if self.my_epi % 500 == 0:
        #     self.save_model() 
        #     self.save_list_to_file(self.rewards_per_epi, 'rewards_list_'+str(self.my_epi)+'.csv',"Rewards")
        #     self.save_list_to_file(self.entropy, 'entropy_list_'+str(self.my_epi)+'.csv', "Entropy") 

        self.current_obs_list.clear()
        self.action_list.clear()
        self.cumulative_rewards_list.clear()
        self.next_obs_list.clear() 
        self.done_list.clear() 
    
    def act(self, observation: np.ndarray) -> np.ndarray:
        if not self.agent:
            self.agent =SACAgent(len(observation), self.n_actions) 
            #self.load_models()
        action = self.agent.select_action(observation) 
        self.current_obs_list.append(observation)
        self.action_list.append(action)
        
        return action
    
    def save_model(self):
        print("Saving Model")
        torch.save({"model_state_dict":self.agent.actor.state_dict(),
                    "optim_state_dict": self.agent.actor_optim.state_dict(),
                    "loss": self.agent.actor_loss},
                    'actor_model_'+str(self.my_epi)+'.pth') 
        
        torch.save({"model_state_dict":self.agent.critic.state_dict(),
                    "optim_state_dict": self.agent.critic_optim.state_dict(),
                    "loss": self.agent.critic_loss},
                    'critic_model_'+str(self.my_epi)+'.pth') 

    def load_models(self):
        print("Loading Model")
        self.agent.actor.load_state_dict(torch.load("actor_model_2500.pth")["model_state_dict"])
        self.agent.actor_optim.load_state_dict(torch.load("actor_model_2500.pth")["optim_state_dict"])
        self.agent.actor_loss = torch.load("actor_model_2500.pth")["loss"]
        self.agent.critic.load_state_dict(torch.load("critic_model_2500.pth")["model_state_dict"])
        self.agent.critic_optim.load_state_dict(torch.load("critic_model_2500.pth")["optim_state_dict"])
        self.agent.critic_loss = torch.load("critic_model_2500.pth")["loss"] 
        # print(self.agent.actor_loss)
        # print(self.agent.critic_loss)

    def process_list(self):  
        for i in range(len(self.cumulative_rewards_list) - 1):  # Iterate through all items except the last one
                if self.cumulative_rewards_list[i] < -125.0:  # If the item is -125.0, set it to 0.0
                    self.cumulative_rewards_list[i] = 0.0 

    def save_list_to_file(self, data_list, file_path, header=None):
        # Open the file for writing
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header row if provided
            if header is not None:
                writer.writerow(header)

            # Write each element of the data list to the file
            for item in data_list:
                writer.writerow([item])
def main():

    icce = SACEICCE()
    icce.run()
    
if __name__ == '__main__':
    main()
        
        
        

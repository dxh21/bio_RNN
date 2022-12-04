import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
from torch.distributions import Categorical 
import random
from statistics import mean
from scipy.stats import entropy
import matplotlib.pyplot as plt

device = "cpu" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class A2C(nn.Module):
    def __init__(self, input_size, n_actions, bandits, hidden_size=48, num_layers=1, gamma = 0.75):
        super(A2C, self).__init__()

        self.num_layers = num_layers
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.bandits = Bandits()

        self.rnn = nn.LSTM(input_size, self.hidden_size)

        self.pi = nn.Linear(self.hidden_size, n_actions)
        self.v = nn.Linear(self.hidden_size, 1)

        self.actions = []
        self.rewards = []
        self.x = [[0, 1, 0, 1]]
        self.x = torch.tensor(self.x, dtype=torch.float, device=device)

        self.policies = []
        self.values = []
        
        self.h_t = (torch.zeros(self.x.size(0), self.hidden_size, dtype=torch.float32, device=device))
        self.c_t = (torch.zeros(self.x.size(0), self.hidden_size, dtype=torch.float32, device=device))

        self.hiddens = [[self.h_t, self.c_t]]
        self.xs = []
        
    def remember(self, action, reward, policy, value):
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)

    def clear_memory(self):
        self.actions = []
        self.rewards = []
        self.policies = []
        self.values = []

    def form_x(self, action, reward, timestep):
        if action == 0:
            x = [[reward, 1, 0, timestep]]
        if action == 1:
            x = [[reward, 0, 1, timestep]]
        self.x = torch.tensor(x, dtype=torch.float, device=device)

    def forward(self,x):
        lstm_output, (self.h_t,self.c_t) = self.rnn(x, (self.h_t,self.c_t))
        self.hiddens.append([self.h_t, self.c_t])
        pre_softmax_policy = self.pi(lstm_output)
        value = self.v(lstm_output)

        policy = torch.softmax(pre_softmax_policy, dim=1)

        return pre_softmax_policy, value

    def calc_loss(self, done, t_step, T_MAX):
        beta_e = 0.05
        beta_v = 0.05

        pi, value = self.forward(self.x)
        #probs = pi
        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().cpu().numpy()[0] #choosing action
        action = int(action)


        reward, p_0, p_1 = self.bandits.calculate_reward_independent(action, t_step, T_MAX)

        self.remember(action, reward, self.x, value)
        values = torch.tensor(self.values, dtype=torch.float, device=device)
        actions = torch.tensor(self.actions, dtype=torch.float, device=device)
        
        t_step_input = (t_step+1) % 100
        if t_step_input == 0:
            t_step_input = 100

        self.form_x(action, reward, t_step_input)
        #self.actor_critic.xs.append(self.actor_critic.x)

        R = values[-1]*(1-int(done))

        batch_return = []

        for rewardcounter in self.rewards[::-1]:
            R = rewardcounter + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float, device=device)
        #print("batch return mean:", batch_return.mean())
        #print("values mean", values.mean())
        returns = batch_return

        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        values = values.squeeze()   
        critic_loss = (returns-values)**2

        entropy = dist.entropy()

        total_loss = (beta_v*critic_loss + actor_loss + beta_e*entropy).mean()

        return total_loss, reward, action, p_0, p_1 

    def choose_action(self, x):
        pi,value = self.forward(x)
        pi = torch.softmax(pi, dim=1)
        dist = Categorical(pi)
        action = dist.sample().cpu().numpy()[0]
        action = int(action)
        return action, value
    
class Bandits():
    def __init__(self):
        super(Bandits, self).__init__()
        self.p_0 = round(random.random(), 2)
        self.p_1 = round(random.random(), 2)
    
    def calculate_reward_independent(self, action, t_step, T_MAX):
        if (t_step-1) % (T_MAX) == 0:
            self.p_0 = round(random.random(), 2)
            self.p_1 = round(random.random(), 2)

        reward = 0 
        if action == 0:
            if random.random() <= self.p_0:
                reward += 1
        if action == 1:
            if random.random() <= self.p_1:
                reward += 1
        return reward, self.p_0, self.p_1
        

class Agent():
    def __init__(self, actor_critic, optimizer, input_size, n_actions, gamma, lr, bandits, episode):
        super(Agent, self).__init__() 
        self.actor_critic = actor_critic
        self.episode = episode 
        self.optimizer = optimizer 
        self.bandits = Bandits()
        self.actor_critic.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not
        next(self.actor_critic.parameters()).device

    def run(self):
        t_step = 1
        self.terminal_memory = []
        self.action_history = []
        self.graph_episode = []
        self.graph_reward = []
        self.graph_loss = []
        
        while self.episode < (TOTAL_EPISODES):
            done = False
            score = 0 
            self.actor_critic.clear_memory()
            while not done:
                loss, reward, action, p_0, p_1 = self.actor_critic.calc_loss(done, t_step, T_MAX)
                score += reward
                if t_step % T_MAX == 0 and t_step != 0: 
                    loss.backward(retain_graph=True)
                    done = True
                    #loss = self.actor_critic.calc_loss(done)
                    #self.graph_loss.append(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    #for name, param in self.actor_critic.rnn.named_parameters():
                    #        print(name, param.data)
                    #for name, param in self.actor_critic.v.named_parameters():
                    #    print(name, param.data)
                    print(sum(self.actor_critic.actions))
                    #print("state dictionary:", self.actor_critic.state_dict())
                    self.actor_critic.h_t.detach()
                    self.actor_critic.c_t.detach()
                    self.actor_critic.h_t = (torch.zeros(self.actor_critic.x.size(0), self.actor_critic.hidden_size, dtype=torch.float32, device=device))
                    self.actor_critic.c_t = (torch.zeros(self.actor_critic.x.size(0), self.actor_critic.hidden_size, dtype=torch.float32, device=device))

                    if self.episode % 10 == 0:
                        self.action_history.append('Sum of actions taken:' + str(sum(self.actor_critic.actions)))
                    print("p_0 =", p_0, "p_1 =", p_1)
                    self.actor_critic.clear_memory()
                    self.actor_critic.hiddens = []
                    self.actor_critic.xs = []
                t_step += 1
            self.episode += 1
            print('Episode', self.episode, 'Reward %.1f' % score)
            self.graph_episode.append(self.episode)
            self.graph_reward.append(score/(max(p_0,p_1)))
                    
            if self.episode % 10 == 0: 
                first_string = 'episode ', str(self.episode), str('reward %.1f' % score)
                second_string = 'probabilities: ',str(p_0),str(p_1)
                self.terminal_memory.append(first_string+second_string)

        self.graph_loss = torch.tensor(self.graph_loss, dtype=torch.float, device=device)
        self.graph_loss.detach().numpy()
        #plt.plot(self.graph_episode,self.graph_loss)
        #plt.show()
        self.graph_reward_mean = []
        average = 0
        for n in range(1,len(self.graph_reward)+1): 
            average += self.graph_reward[n-1]
            if n % 100 == 0:
                self.graph_reward_mean.append(average/100)
                average = 0

        self.graph_episode_for_mean = np.arange(50,TOTAL_EPISODES+50,100)
        plt.plot(self.graph_episode, self.graph_reward)
        plt.plot(self.graph_episode_for_mean, self.graph_reward_mean)
        plt.show()

        self.graph_episode = []
        self.graph_reward = []

        FILE = "meta_lstm.pth"
        torch.save(actor_critic.state_dict(), FILE)

        for name, param in actor_critic.named_parameters():
            print(name, param.data)

        print("HOLDING WEIGHTS FIXED...")
        self.action_history.append("HOLDING WEIGHTS FIXED...")
        self.terminal_memory.append("HOLDING WEIGHTS FIXED...")

        t_step = 1
        
        while self.episode < (TOTAL_EPISODES + META_EPISODES):
            done = False
            score = 0 
            self.actor_critic.clear_memory()
            while not done:
                action,value = self.actor_critic.choose_action(self.actor_critic.x)
                reward, p_0, p_1 = self.bandits.calculate_reward_independent(action, t_step, T_MAX)
                score += reward 
                self.actor_critic.remember(action, reward, self.actor_critic.x, value)
                t_step_input = t_step % 100
                if t_step_input == 0:
                    t_step_input = 100
                self.actor_critic.form_x(action, reward, t_step_input)
                self.actor_critic.xs.append(self.actor_critic.x)
                
                if t_step % T_MAX == 0 and t_step != 0: 
                    self.actor_critic.h_t = (torch.zeros(self.actor_critic.x.size(0), self.actor_critic.hidden_size, dtype=torch.float32, device=device))
                    self.actor_critic.c_t = (torch.zeros(self.actor_critic.x.size(0), self.actor_critic.hidden_size, dtype=torch.float32, device=device))

                    done = True
                    #for name, param in self.actor_critic.rnn.named_parameters():
                    #    print(name, param.data)
                    print(sum(self.actor_critic.actions))
                    if self.episode % 10 == 0:
                        self.action_history.append('Sum of actions taken:' + str(sum(self.actor_critic.actions)))
                    print("p_0 =", p_0, "p_1 =", p_1)
                    self.actor_critic.clear_memory()
                    self.actor_critic.hiddens = []
                    self.actor_critic.xs = []
                t_step += 1
            self.episode += 1
            print('Episode', self.episode, 'Reward %.1f' % score)
            self.graph_episode.append(self.episode)
            self.graph_reward.append(score/(max(p_0,p_1))) 

        self.graph_episode_for_mean = np.arange(TOTAL_EPISODES+5,TOTAL_EPISODES+505,10)
        self.graph_reward_mean = []

        average = 0
        for n in range(1,len(self.graph_reward)+1): 
            average += self.graph_reward[n-1]
            if n % 10 == 0:
                self.graph_reward_mean.append(average/10)
                average = 0

        plt.plot(self.graph_episode,self.graph_reward)
        plt.plot(self.graph_episode_for_mean,self.graph_reward_mean)
        plt.show()

if __name__ == '__main__':
    lr = 7e-4 
    n_actions = 2 
    input_size = 4     
    T_MAX = 100
    TOTAL_EPISODES = 20000
    META_EPISODES = 500
    bandits = Bandits()
    actor_critic = A2C(input_size, n_actions, bandits)
    optimizer = optim.RMSprop(actor_critic.parameters(), lr=lr)
    ep = 0 
    worker = Agent(actor_critic, optimizer, input_size, n_actions, gamma=0.75, lr=lr, bandits = Bandits(), episode = ep)

    worker.run()
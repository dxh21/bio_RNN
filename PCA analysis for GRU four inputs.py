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
from matplotlib import colors
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import preprocessing
from mpl_toolkits import mplot3d

device = "cpu" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class A2C(nn.Module):
    def __init__(self, input_size, n_actions, bandits, hidden_size=96, num_layers=1, gamma = 0.75):
        super(A2C, self).__init__()

        self.num_layers = num_layers
        self.gamma = gamma
        self.hidden_size = hidden_size
        self.bandits = Bandits()

        self.rnn = nn.GRU(input_size, self.hidden_size)

        self.pi = nn.Linear(self.hidden_size, n_actions)
        self.v = nn.Linear(self.hidden_size, 1)

        self.actions = []
        self.rewards = []
        self.oracle_rewards = []
        self.dumb_rewards = []
        self.meta_rewards = []
        self.x = [[0, 1, 0, 1]]
        self.x = torch.tensor(self.x, dtype=torch.float, device=device)

        self.policies = []
        self.values = []
        
        self.h_t = (torch.zeros(self.x.size(0), self.hidden_size, dtype=torch.float32, device=device))

        self.hiddens = [self.h_t]
        self.xs = []

        zero_input = [[1, 1, 0, 1]]
        self.zero_input = torch.tensor(zero_input, dtype=torch.float, device=device)
        
    def remember(self, action, reward, policy, value, oracle, dumb):
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        self.oracle_rewards.append(oracle)
        self.dumb_rewards.append(dumb)
        self.meta_rewards.append(reward)

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
        lstm_output, self.h_t = self.rnn(x, self.h_t)
        self.hiddens.append(self.h_t)
        pre_softmax_policy = self.pi(lstm_output)
        value = self.v(lstm_output)

        policy = torch.softmax(pre_softmax_policy, dim=1)

        return pre_softmax_policy, value

    def hidden_path(self):
        lstm_output, self.h_t = self.rnn(self.zero_input, self.h_t)
        self.zero_input[0][3] += 1    

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
    
    def calculate_reward_25_75(self, action, T_MAX):
        p_0 = 0.1
        p_1 = 0.9
        reward = 0
        if action == 0:
            if random.random() <= p_0:
                reward += 1
        if action == 1:
            if random.random() <= p_1:
                reward += 1 
        return reward, p_0, p_1
    
    def calculate_reward_flip(self, action, T_MAX):
        p_0 = 0.9
        p_1 = 0.1
        reward = 0 
        if action == 0:
            if random.random() <= p_0:
                reward += 1
        if action == 1:
            if random.random() <= p_1:
                reward += 1 
        return reward, p_0, p_1

    def oracle(self, p_0, p_1):
        reward = 0
        if random.random() <= max(p_0,p_1):
            reward += 1
        return reward 

    def dumb(self, p_0, p_1):
        reward = 0
        if random.random() <= 0.5:
            dumb_decision = 0
        else:
            dumb_decision = 1
        if dumb_decision == 0:
            if random.random() <= p_0:
                reward += 1
        if dumb_decision == 1:
            if random.random() <= p_1:
                reward += 1
        return reward

class Agent():
    def __init__(self, actor_critic, input_size, n_actions, gamma, lr, bandits, episode):
        super(Agent, self).__init__() 
        self.actor_critic = actor_critic
        self.episode = episode  
        self.bandits = Bandits()
        self.actor_critic.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not
        next(self.actor_critic.parameters()).device

    def run(self):
        self.terminal_memory = []
        self.action_history = []
        self.graph_episode = []
        self.graph_reward = []
        self.graph_loss = []
        data = []
        hidden_state_matrix1 = []
        hidden_state_matrix2 = []

        t_step = 1

        self.actor_critic.h_t = (torch.zeros(self.actor_critic.x.size(0), self.actor_critic.hidden_size, dtype=torch.float32, device=device))
        
        while self.episode < (META_EPISODES):
            done = False
            score = 0 
            self.actor_critic.clear_memory()
            while not done:
                action, value = self.actor_critic.choose_action(self.actor_critic.x)
                reward, p_0, p_1 = self.bandits.calculate_reward_25_75(action, T_MAX)
                oracle_reward = self.bandits.oracle(p_0, p_1)
                dumb_reward = self.bandits.dumb(p_0, p_1)
                score += reward 
                self.actor_critic.remember(action, reward, self.actor_critic.x, value, oracle_reward, dumb_reward)
                t_step_input = t_step % 100
                if t_step_input == 0:
                    t_step_input = 100
                self.actor_critic.form_x(action, reward, t_step_input)
                self.actor_critic.xs.append(self.actor_critic.x)

                hidden_state_matrix1.append(self.actor_critic.h_t.tolist()[0]) 
                
                if t_step % T_MAX == 0 and t_step != 0: 
                    #self.actor_critic.h_t = (torch.zeros(self.actor_critic.x.size(0), self.actor_critic.hidden_size, dtype=torch.float32, device=device))

                    done = True
                    #for name, param in self.actor_critic.rnn.named_parameters():
                    #    print(name, param.data)
                    print(self.actor_critic.actions)
                    print(sum(self.actor_critic.actions))
                    data.append(self.actor_critic.actions)
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

        '''self.graph_episode_for_mean = np.arange(TOTAL_EPISODES+5,TOTAL_EPISODES+505,10)
        self.graph_reward_mean = []

        average = 0
        for n in range(1,len(self.graph_reward)+1): 
            average += self.graph_reward[n-1]
            if n % 10 == 0:
                self.graph_reward_mean.append(average/10)
                average = 0'''

        hidden_state_matrix1 = [list(i) for i in zip(*hidden_state_matrix1)]
        hidden_state_labels = ['Hidden dimension number ' + str(i) for i in range(1,actor_critic.hidden_size+1)]
        data1 = pd.DataFrame(hidden_state_matrix1, index=hidden_state_labels)
        scaled_data1 = preprocessing.scale(data1.T)

        pca1 = PCA()
        pca1.fit(scaled_data1)
        pca_data1 = pca1.transform(scaled_data1)

        per_var1 = np.round(pca1.explained_variance_ratio_* 100, decimals=1)
        labels1 = ['PC' + str(x) for x in range(1, len(per_var1)+1)]
        
        pca_df1 = pd.DataFrame(pca_data1, index=hidden_state_matrix1, columns=labels1)

        # starting second section 
        self.actor_critic.h_t = (torch.zeros(self.actor_critic.x.size(0), self.actor_critic.hidden_size, dtype=torch.float32, device=device))

        while self.episode < (META_EPISODES+META_EPISODES):
            done = False
            score = 0 
            self.actor_critic.clear_memory()
            while not done:
                action, value = self.actor_critic.choose_action(self.actor_critic.x)
                reward, p_0, p_1 = self.bandits.calculate_reward_flip(action, T_MAX)
                oracle_reward = self.bandits.oracle(p_0, p_1)
                dumb_reward = self.bandits.dumb(p_0, p_1)
                score += reward 
                self.actor_critic.remember(action, reward, self.actor_critic.x, value, oracle_reward, dumb_reward)
                t_step_input = t_step % 100
                if t_step_input == 0:
                    t_step_input = 100
                self.actor_critic.form_x(action, reward, t_step_input)
                self.actor_critic.xs.append(self.actor_critic.x)
                
                hidden_state_matrix2.append(self.actor_critic.h_t.tolist()[0])

                if t_step % T_MAX == 0 and t_step != 0: 
                    #self.actor_critic.h_t = (torch.zeros(self.actor_critic.x.size(0), self.actor_critic.hidden_size, dtype=torch.float32, device=device))

                    done = True
                    #for name, param in self.actor_critic.rnn.named_parameters():
                    #    print(name, param.data)
                    print(self.actor_critic.actions)
                    print(sum(self.actor_critic.actions))
                    data.append(self.actor_critic.actions)
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

        hidden_state_matrix2 = [list(i) for i in zip(*hidden_state_matrix2)]
        data2 = pd.DataFrame(hidden_state_matrix2, index=hidden_state_labels)
        scaled_data2 = preprocessing.scale(data2.T)

        pca2 = PCA()
        pca2.fit(scaled_data1)
        pca_data2 = pca2.transform(scaled_data2)

        per_var2 = np.round(pca2.explained_variance_ratio_* 100, decimals=1)
        labels2 = ['PC' + str(x) for x in range(1, len(per_var2)+1)]
        
        pca_df2 = pd.DataFrame(pca_data2, index=hidden_state_matrix2, columns=labels2)

        plt.scatter(pca_df1.PC1, pca_df1.PC2, color=['red'])
        plt.scatter(pca_df2.PC1, pca_df2.PC2, color=['blue'])
        plt.plot(pca_df1.PC1, pca_df1.PC2)
        plt.plot(pca_df2.PC1, pca_df2.PC2)
        plt.show()

        plt.plot(self.graph_episode,self.graph_reward)
        #plt.plot(self.graph_episode_for_mean,self.graph_reward_mean)
        plt.show()

        '''fig = plt.figure()
        ax = plt.axes(projection='3d')

        plt.plot(pca_df1.PC1, pca_df1.PC2, pca_df1.PC3, 'r')
        #plt.plot(pca_df2.PC1, pca_df2.PC2, 'b')
        plt.show()'''

        cmap = colors.ListedColormap(['dodgerblue','skyblue'])
        bounds = [0,1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(data, cmap=cmap, norm=norm)
        plt.show()

        print(sum(self.actor_critic.meta_rewards))
        print(sum(self.actor_critic.oracle_rewards))
        print(sum(self.actor_critic.dumb_rewards))

if __name__ == '__main__':
    lr = 7e-4 
    n_actions = 2 
    input_size = 4     
    T_MAX = 100
    TOTAL_EPISODES = 100
    META_EPISODES = 10
    bandits = Bandits()
    actor_critic = A2C(input_size, n_actions, bandits)
    actor_critic.load_state_dict(torch.load("meta_lstm_gru_96_hidden_size.pth"))
    ep = 0 
    worker = Agent(actor_critic, input_size, n_actions, gamma=0.75, lr=lr, bandits = Bandits(), episode = ep)

    #for name, param in actor_critic.named_parameters():
    #    print(name, param.data)
    
    hidden_state_matrix1= []
    hidden_state_matrix2 = []
    hidden_state_matrix3 = []
    hidden_state_matrix4 = []

    #print("hidden state", actor_critic.h_t)
    #actor_critic.hidden_path()
    #print("hidden state", actor_critic.h_t)
    #print("hidden state list", actor_critic.h_t.tolist()[0])

    # first

    h_t = [0] * actor_critic.hidden_size
    h_t = [h_t]
    actor_critic.h_t = torch.tensor(h_t, dtype=torch.float, device=device)

    for n in range(10):
        #print(actor_critic.zero_input) 
        actor_critic.hidden_path()  
        hidden_state_matrix1.append(actor_critic.h_t.tolist()[0]) 
    
    zero_input = [[0, 1, 0, 1]]
    actor_critic.zero_input = torch.tensor(zero_input, dtype=torch.float, device=device)

    for n in range(90):
        actor_critic.hidden_path()  
        hidden_state_matrix1.append(actor_critic.h_t.tolist()[0])

    hidden_state_matrix1 = [list(i) for i in zip(*hidden_state_matrix1)]
    #print("hidden_state_matrix", hidden_state_matrix)

    hidden_state_labels = ['Hidden dimension number ' + str(i) for i in range(1,actor_critic.hidden_size+1)]
    
    data1 = pd.DataFrame(hidden_state_matrix1, index=hidden_state_labels)

    scaled_data1 = preprocessing.scale(data1.T)

    pca1 = PCA()
    pca1.fit(scaled_data1)
    pca_data1 = pca1.transform(scaled_data1)

    per_var1 = np.round(pca1.explained_variance_ratio_* 100, decimals=1)
    labels1 = ['PC' + str(x) for x in range(1, len(per_var1)+1)]
    
    pca_df1 = pd.DataFrame(pca_data1, index=hidden_state_matrix1, columns=labels1)

    # second

    zero_input = [[0, 1, 0, 1]]
    actor_critic.zero_input = torch.tensor(zero_input, dtype=torch.float, device=device)
    h_t = [0] * actor_critic.hidden_size
    h_t = [h_t]
    actor_critic.h_t = torch.tensor(h_t, dtype=torch.float, device=device)

    for n in range(100):
        #print(actor_critic.zero_input) 
        actor_critic.hidden_path()  
        hidden_state_matrix2.append(actor_critic.h_t.tolist()[0])
    
    hidden_state_matrix2 = [list(i) for i in zip(*hidden_state_matrix2)]
    #print("hidden_state_matrix", hidden_state_matrix)

    hidden_state_labels = ['Hidden dimension number ' + str(i) for i in range(1,actor_critic.hidden_size+1)]
    
    data2 = pd.DataFrame(hidden_state_matrix2, index=hidden_state_labels)

    scaled_data2 = preprocessing.scale(data2.T)

    pca2 = PCA()
    pca2.fit(scaled_data2)
    pca_data2 = pca2.transform(scaled_data2)

    per_var2 = np.round(pca2.explained_variance_ratio_* 100, decimals=1)
    labels2 = ['PC' + str(x) for x in range(1, len(per_var1)+1)]
    
    pca_df2 = pd.DataFrame(pca_data2, index=hidden_state_matrix2, columns=labels2)
    # third

    zero_input = [[1, 0, 1, 1]]
    actor_critic.zero_input = torch.tensor(zero_input, dtype=torch.float, device=device)
    h_t = [0] * actor_critic.hidden_size
    h_t = [h_t]
    actor_critic.h_t = torch.tensor(h_t, dtype=torch.float, device=device)

    for n in range(100):
        #print(actor_critic.zero_input) 
        actor_critic.hidden_path()  
        hidden_state_matrix3.append(actor_critic.h_t.tolist()[0])
    
    hidden_state_matrix3 = [list(i) for i in zip(*hidden_state_matrix3)]
    #print("hidden_state_matrix", hidden_state_matrix)

    hidden_state_labels = ['Hidden dimension number ' + str(i) for i in range(1,actor_critic.hidden_size+1)]
    
    data3 = pd.DataFrame(hidden_state_matrix3, index=hidden_state_labels)

    scaled_data3 = preprocessing.scale(data3.T)

    pca3 = PCA()
    pca3.fit(scaled_data3)
    pca_data3 = pca3.transform(scaled_data3)

    per_var3 = np.round(pca3.explained_variance_ratio_* 100, decimals=1)
    labels3 = ['PC' + str(x) for x in range(1, len(per_var1)+1)]
    
    pca_df3 = pd.DataFrame(pca_data3, index=hidden_state_matrix3, columns=labels3)
    # fourth

    zero_input = [[0, 0, 1, 1]]
    actor_critic.zero_input = torch.tensor(zero_input, dtype=torch.float, device=device)
    h_t = [0] * actor_critic.hidden_size
    h_t = [h_t]
    actor_critic.h_t = torch.tensor(h_t, dtype=torch.float, device=device)

    for n in range(100):
        #print(actor_critic.zero_input) 
        actor_critic.hidden_path()  
        hidden_state_matrix4.append(actor_critic.h_t.tolist()[0])

    hidden_state_matrix4 = [list(i) for i in zip(*hidden_state_matrix4)]
    #print("hidden_state_matrix", hidden_state_matrix)

    hidden_state_labels = ['Hidden dimension number ' + str(i) for i in range(1,actor_critic.hidden_size+1)]
    
    data4 = pd.DataFrame(hidden_state_matrix4, index=hidden_state_labels)

    scaled_data4 = preprocessing.scale(data4.T)

    pca4 = PCA()
    pca4.fit(scaled_data4)
    pca_data4 = pca4.transform(scaled_data4)

    per_var4 = np.round(pca4.explained_variance_ratio_* 100, decimals=1)
    labels4 = ['PC' + str(x) for x in range(1, len(per_var4)+1)]
    
    pca_df4 = pd.DataFrame(pca_data4, index=hidden_state_matrix4, columns=labels4)
    #---------------------
    '''h_t = [2] * actor_critic.hidden_size
    h_t[3] = -2
    h_t[10] = -1
    h_t[20] = -1
    h_t[31] = -1
    h_t = [h_t]
    actor_critic.h_t = torch.tensor(h_t, dtype=torch.float, device=device)

    for n in range(1000):
        actor_critic.hidden_path()
        hidden_state_matrix.append(actor_critic.h_t.tolist()[0])'''
    
    #---------------------
    
    '''hidden_state_matrix = [list(i) for i in zip(*hidden_state_matrix)]
    #print("hidden_state_matrix", hidden_state_matrix)

    hidden_state_labels = ['Hidden dimension number ' + str(i) for i in range(1,actor_critic.hidden_size+1)]
    
    data = pd.DataFrame(hidden_state_matrix, index=hidden_state_labels)

    scaled_data = preprocessing.scale(data.T)

    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

    plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

    pca_df = pd.DataFrame(pca_data, index=hidden_state_matrix, columns=labels)'''

    #fig = plt.figure()
    #ax = plt.axes(projection='3d')

    plt.scatter(pca_df1.PC1, pca_df1.PC2, color=['red'])
    plt.scatter(pca_df2.PC1, pca_df2.PC2, color=['blue'])
    plt.scatter(pca_df3.PC1, pca_df3.PC2, color=['green'])
    plt.scatter(pca_df4.PC1, pca_df4.PC2, color=['yellow'])
    plt.plot(pca_df1.PC1, pca_df1.PC2, 'r')
    plt.plot(pca_df2.PC1, pca_df2.PC2, 'b')
    plt.plot(pca_df3.PC1, pca_df3.PC2, 'g')
    plt.plot(pca_df4.PC1, pca_df4.PC2, 'y')
    '''ax.scatter3D(pca_df1.PC1, pca_df1.PC2, pca_df1.PC3)
    ax.scatter3D(pca_df2.PC1, pca_df2.PC2, pca_df2.PC3)
    ax.scatter3D(pca_df3.PC1, pca_df3.PC2, pca_df3.PC3)
    ax.scatter3D(pca_df4.PC1, pca_df4.PC2, pca_df4.PC3)'''
    #ax.plot3D(pca_df.PC1[:len(pca_df.PC1)//2], pca_df.PC2[:len(pca_df.PC1)//2], pca_df.PC3[:len(pca_df.PC1)//2])
    #ax.plot3D(pca_df.PC1[len(pca_df.PC1)//2:], pca_df.PC2[len(pca_df.PC1)//2:], pca_df.PC3[len(pca_df.PC1)//2:])
    plt.title('Trajectory of hidden states in GRU')
    plt.legend(['Left arm always reward', 'Left arm never reward', 'Right arm always reward', 'Right arm never reward'], prop={'size': 20})
    #plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    #plt.ylabel('PC2 - {0}%'.format(per_var[1])) 
    #plt.zlabel('PC3 - {0}%'.format(per_var[2]))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

#----------------------------------------------------------------------------------------------
# Now doing PCA for all four dynamical systems together so the trajectories all start from 0 
    print(len(hidden_state_matrix1[0]))
    hidden_state_matrix_total = hidden_state_matrix1 + hidden_state_matrix2 + hidden_state_matrix3 + hidden_state_matrix4
    #print("hidden_state_matrix", hidden_state_matrix)

    hidden_state_matrix_total = [0] * 96
    
    for n in range(len(hidden_state_matrix_total)):
        hidden_state_matrix_total[n] = hidden_state_matrix1[n] + hidden_state_matrix2[n] + hidden_state_matrix3[n] + hidden_state_matrix4[n]

    hidden_state_labels = ['Hidden dimension number ' + str(i) for i in range(1,actor_critic.hidden_size+1)]
    
    data5 = pd.DataFrame(hidden_state_matrix_total, index=hidden_state_labels)

    scaled_data5 = preprocessing.scale(data5.T)

    pca5 = PCA()
    pca5.fit(scaled_data5)
    pca_data5 = pca5.transform(scaled_data5)

    per_var5 = np.round(pca5.explained_variance_ratio_* 100, decimals=1)
    labels5 = ['PC' + str(x) for x in range(1, len(per_var5)+1)]
    
    pca_df5 = pd.DataFrame(pca_data5, index=hidden_state_matrix_total, columns=labels5)
    print(len(pca_df5.PC1))
    plt.scatter(pca_df5.PC1[0:99], pca_df5.PC2[0:99], color=['red'])
    plt.scatter(pca_df5.PC1[100:199], pca_df5.PC2[100:199], color=['blue'])
    plt.scatter(pca_df5.PC1[200:299], pca_df5.PC2[200:299], color=['green'])
    plt.scatter(pca_df5.PC1[300:400], pca_df5.PC2[300:400], color=['yellow'])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(["left always reward", "left no reward", "right always reward", "right no reward"])
    plt.plot(pca_df5.PC1[0:99], pca_df5.PC2[0:99], 'r')
    plt.plot(pca_df5.PC1[100:199], pca_df5.PC2[100:199], 'b')
    plt.plot(pca_df5.PC1[200:299], pca_df5.PC2[200:299], 'g')
    plt.plot(pca_df5.PC1[300:400], pca_df5.PC2[300:400], 'y')
    #plt.legend("left always reward, left no reward, right always reward, right no reward ")
    plt.show()
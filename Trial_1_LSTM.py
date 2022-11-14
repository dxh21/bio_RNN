import numpy as np
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.multiprocessing as mp # to handle multiprocessing
import torch.nn.functional as F # to handle activation functions
from torch.distributions import Categorical # takes a probability output from a deep neural network and maps it to a distribution
import random
from statistics import mean
from scipy.stats import entropy

device = "cpu" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class A2C(nn.Module):
    def __init__(self, input_size, n_actions, hidden_size=48,num_layers=1, gamma=0.75):    # initialiser takes input_dims from environment, n_actions from agent, discount factor = 0.99
        super(A2C, self).__init__()
        self.num_layers = num_layers
        self.gamma = gamma
        self.hidden_size = hidden_size

        self.pi1 = nn.LSTM(*input_size, self.hidden_size)
        self.v1 = nn.LSTM(*input_size, self.hidden_size)

        self.pi = nn.Linear(self.hidden_size, n_actions)    #softmax layer for the policy
        self.v = nn.Linear(self.hidden_size, 1)     #linear layer for the value

        #network also has some basic memory 
        self.actions = []
        self.rewards = []
        self.xs = []
        self.x = [[0, 1, 0]]
        self.x = torch.tensor(self.x, dtype=torch.float, device=device)

    def remember(self, action, reward, x):
        self.actions.append(action)
        self.rewards.append(reward)
        self.xs.append(x)

    def clear_memory(self):
        self.actions = []
        self.rewards = []
        self.xs = []

    def form_x(self, action, reward):
        if action == 0:
            self.x = [[reward, 1, 0]]
        if action == 1:
            self.x = [[reward, 0, 1]]
        self.x = torch.tensor(self.x, dtype=torch.float, device=device)

        # the input consists of a scalar indicating the previous reward and a one-hot representation of the previous action
        # ie (1, [1,0]) for two armed bandit problem 
        # form the inputs into one vector ie [1, 1, 0]

        # the outputs consists of a scalar baseline(value function) and a real vector with length equal to the number of available actions
        # Actions were sampled from the softmax distribution defined by this vector

    def forward(self, x): 
        
        h_t = (torch.zeros(self.x.size(0), self.hidden_size, dtype=torch.float32, device=device))
        c_t = (torch.zeros(self.x.size(0), self.hidden_size, dtype=torch.float32, device=device))
        h_t2 = (torch.zeros(self.x.size(0), self.hidden_size, dtype=torch.float32, device=device))
        c_t2 = (torch.zeros(self.x.size(0), self.hidden_size, dtype=torch.float32, device=device))

        pi1,_ = self.pi1(self.x, (h_t,c_t))
        v1,_ = self.v1(self.x, (h_t2,c_t2))
        #self.pi1layer = nn.Parameter(pi1layer)
        #self.v1layer = nn.Parameter(v1layer)
        #return pi1    
        m = nn.Tanh()
        #pi1 = m(pi1)
        #v1 = m(v1)
        pi2 = self.pi(pi1)
        v2 = self.v(v1)
        
        '''pi1,_ = self.pi1(self.x)
        v1,_ = self.v1(self.x)
        pi = self.pi(pi1)
        v = self.v(v1)'''

        #output = self.softmax(output)

        return pi2, v2

    # next have the function that determines return(sum of rewards and discount factor) from action  
    def calc_R(self, done):
        xs = torch.tensor(self.xs, dtype=torch.float, device=device)
        _, v = self.forward(xs) # pass it through neural network 

        R = v[-1]*(1-int(done))    # when the episode is done 1-int(done) will be 0 therefore R becomes 0 

        # then handle the calculation of the returns at all the other timesteps
        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float, device=device)

        return batch_return

        # next handle calculation of loss function

    def calc_loss(self, done):
        xs = torch.tensor(self.xs, device=device)
        actions = torch.tensor(self.actions, dtype=torch.float, device=device)   # get the tensor representations of states and actions

        returns = self.calc_R(done)  # calculate returns

        # perform the update, pass states through actor critic network to get the new values as well as a distrition according to current values of neural network
        # use distribution to get the log probabilities of the actions the agent actually took at the time it took them 
        # the use those quantities for loss functions

        pi, values = self.forward(xs)
        values = values.squeeze()            # squeeze is needed to change shape of critic_loss and actor_loss from 5x5 to a 5x1 vector
        critic_loss = (returns-values)**2    # not sure why this is being squared but (returns-values) is the n-step return temporal-difference error that provides an estimate of the advantage function for actor-critic
                                                # wang paper doesn't have the **2 but the mnih paper does 
        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)            #creates a categorical distribution
        log_probs = dist.log_prob(actions)
        actor_loss = -log_probs*(returns-values) # from the paper 

        total_loss = (critic_loss + actor_loss).mean()    # we have to sum these two together because of the way backprop is handled by pytorch 
    
        return total_loss

    def calc_R_2018(self, done):
        #xs = torch.tensor(self.xs[0], device=device)
        xs = self.xs
        _, v = self.forward(xs) # pass it through neural network 

        first_reward = self.rewards[0]    # when the episode is done 1-int(done) will be 0 therefore R becomes 0 
        V_terminal = v[-1]
        # then handle the calculation of the returns at all the other timesteps
        batch_return = []
        for n in range(len(self.rewards)):
            R = self.rewards[n] 
            for k in range(len(self.rewards)-n-1):
                R += (self.gamma**(k+1))*self.rewards[n+k+1]
            R += (len(self.rewards)-1-n) * self.gamma**(len(self.rewards)-1-n) * V_terminal
            batch_return.append(R)
        batch_return = torch.tensor(batch_return, dtype=torch.float, device=device)

        return batch_return

    '''def calc_loss_2018(self, done):
        #xs = torch.tensor(self.xs, device=device)
        xs = self.xs
        beta_v = 0.05
        beta_e = 0.05

        actions = torch.tensor(self.actions, dtype=torch.float, device=device)
        returns = self.calc_R_2018(done)
        print("xs:",xs)
        pi_list = []
        values_list = []
        for x in xs:
            pi, values = self.forward(x)
            pi_list.append(pi)
            values_list.append(values)
            print(x , pi, values)
        print("pi_list:", pi_list)
        print("values_list:", values_list)
        values = values.squeeze()            # squeeze is needed to change shape of critic_loss and actor_loss from 5x5 to a 5x1 vector
        critic_loss = beta_v * (returns-values) *values    
        print("pi shape:",pi.shape)
        probs = torch.softmax(pi, dim=1)
        print("probs shape:",probs.shape)
        print("probs:",probs)
        dist = Categorical(probs)            #creates a categorical distribution
        log_probs = dist.log_prob(actions)
        print("log_probs:",log_probs.shape)
        print("log_probs:", log_probs)
        actor_loss = -log_probs*(returns-values) # from the paper 

        entropy_regularization = beta_e * dist.entropy()
        print(critic_loss.shape, actor_loss.shape, entropy_regularization.shape)
        print(actor_loss)
        total_loss = (critic_loss + actor_loss).mean()    # we have to sum these two together because of the way backprop is handled by pytorch 
    
        return total_loss'''

    def calc_loss_2018(self, done):
        #xs = torch.tensor(self.xs, device=device)
        xs = self.xs
        beta_v = 0.05
        beta_e = 0.05

        actions = torch.tensor(self.actions, dtype=torch.float, device=device)
        returns = self.calc_R_2018(done)
        pi, values = self.forward(xs)
        values = values.squeeze()            # squeeze is needed to change shape of critic_loss and actor_loss from 5x5 to a 5x1 vector
        critic_loss = beta_v * (returns-values) *values    
        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)            #creates a categorical distribution
        log_probs = dist.log_prob(actions)
        actor_loss = log_probs*(returns-values) # from the paper 
        entropy_regularization = beta_e * dist.entropy()
        entropy_regularization = [entropy_regularization]*100
        entropy_regularization = torch.tensor(entropy_regularization)
        total_loss = (-critic_loss - actor_loss - entropy_regularization).mean()    # we have to sum these two together because of the way backprop is handled by pytorch 
    
        return total_loss    

    def choose_action(self, x):
        #x = torch.tensor(x, dtype=torch.float)
        pi, v = self.forward(x)
        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        action = dist.sample().cpu().numpy()[0]
        return action

class Bandits():
    def calculate_reward(self,action):  
        p_0 = 0.3
        p_1 = 0.7
        reward = 0.
        if action == 0:
            if random.random() <= p_0:
                reward += 1
        if action == 1:
            if random.random() <= p_1:
                reward += 1
        return reward 

    def meta_calculate_reward(self,action):
        p_0 = 0.8
        p_1 = 0.1
        reward = 0 
        if action == 0:
            if random.random() <= p_0:
                reward += 1
        if action == 1:
            if random.random() <= p_1:
                reward += 1
        return reward

    def meta_calculate_reward2(self,action):
        p_0 = 0.1
        p_1 = 0.8
        reward = 0 
        if action == 0:
            if random.random() <= p_0:
                reward += 1
        if action == 1:
            if random.random() <= p_1:
                reward += 1
        return reward

    def store_arm_probs():
        self.p_0 = round(random.random(), 2)
        self.p_1 = round(random.random(), 2)

    def changing_blocks_calculate_reward(self, action, t_step, T_MAX, EPISODES_PER_BLOCK):

        if t_step % (T_MAX*EPISODES_PER_BLOCK) == 0: 
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

class Agent():     # can be called agent or worker
    def __init__(self, actor_critic, optimizer, input_size, n_actions,
                gamma, lr, bandits, global_ep_idx):
        super(Agent, self).__init__()
        self.local_actor_critic = actor_critic
        self.episode_idx = global_ep_idx     # episode index 
        self.optimizer = optimizer 
        self.bandits = Bandits()
        self.local_actor_critic.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not
        next(self.local_actor_critic.parameters()).device

    def run(self):
        t_step = 0     # local time step 
        self.terminal_memory = []
        self.action_history = []
        while self.episode_idx < (N_BLOCKS*EPISODES_PER_BLOCK):
            done = False 
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(self.local_actor_critic.x)         # choose action based on inputs into the neural network x, which is the previous action and reward
                #print(self.local_actor_critic.x)
                #print(action)
                #observation_, reward, done, info = self.env.step(action)  #this uses the environment to get a reward, I'm going to create a class that calculates the reward given action, then import an instance of the class within Agent, then just use e.g. self.local_bandits.calculate_reward(action)
                #reward = self.bandits.calculate_reward(action)
                reward,p_0,p_1 = self.bandits.changing_blocks_calculate_reward(action, t_step, T_MAX, EPISODES_PER_BLOCK)
                score += reward 
                self.local_actor_critic.remember(action, reward, self.local_actor_critic.x)
                self.local_actor_critic.form_x(action, reward)
                if t_step % T_MAX == 0:
                    done = True
                    #print("done")
                    #print(len(self.local_actor_critic.rewards))
                    loss = self.local_actor_critic.calc_loss_2018(done)
                    print("loss:",loss)
                    self.optimizer.zero_grad()                            # .zero_grad() sets the gradients of all optimized torch.Tensor to zero 
                    loss.backward() 
                    self.optimizer.step()                                 # All optimizers implement a step() method, that updates the parameters
                    print(sum(self.local_actor_critic.actions))
                    if self.episode_idx % 10 == 0:
                        self.action_history.append('sum of actions taken:' + str(sum(self.local_actor_critic.actions)))
                    print("p_0 =",p_0, "p_1 =",p_1)
                    self.local_actor_critic.clear_memory()
                    #done = True
                t_step += 1
                #if t_step % T_MAX == 0:
                    #done = True
            self.episode_idx += 1
            print('episode ', self.episode_idx, 'reward %.1f' % score)
        
            if self.episode_idx % 10 == 0: 
                first_string = 'episode ', str(self.episode_idx), str('reward %.1f' % score)
                second_string = 'probabilities: ',str(p_0),str(p_1)
                self.terminal_memory.append(first_string+second_string)

        print("STARTING META LEARNING...")
        self.action_history.append("STARTING META LEARNING...")
        self.terminal_memory.append("STARTING META LEARNING...")

        while self.episode_idx >= (N_BLOCKS*EPISODES_PER_BLOCK) and self.episode_idx < (N_BLOCKS*EPISODES_PER_BLOCK+META_GAMES):
            done = False 
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(self.local_actor_critic.x)         # choose action based on inputs into the neural network x, which is the previous action and reward
                #print(self.x)
                #print(action)
                #observation_, reward, done, info = self.env.step(action)  #this uses the environment to get a reward, I'm going to create a class that calculates the reward given action, then import an instance of the class within Agent, then just use e.g. self.local_bandits.calculate_reward(action)
                reward = self.bandits.meta_calculate_reward(action)
                score += reward 
                self.local_actor_critic.remember(action, reward, self.local_actor_critic.x)
                self.local_actor_critic.form_x(action, reward)
                if t_step % T_MAX == 0:
                    print(sum(self.local_actor_critic.actions))
                    if self.episode_idx % 10:
                        self.action_history.append('sum of actions taken:' + str(sum(self.local_actor_critic.actions)))
                    self.local_actor_critic.clear_memory()
                    done = True
                t_step += 1
                #if t_step % T_MAX == 0:
                    #done = True
            self.episode_idx += 1
            print('episode ', self.episode_idx, 'reward %.1f' % score)
            if self.episode_idx % 10 == 0: 
                first_string = 'episode ', str(self.episode_idx), str('reward %.1f' % score)
                second_string = 'probabilities: ',str(0.8),str(0.1)
                self.terminal_memory.append(first_string+second_string)

        while self.episode_idx >= (N_BLOCKS*EPISODES_PER_BLOCK+META_GAMES) and self.episode_idx < (N_BLOCKS*EPISODES_PER_BLOCK+2*META_GAMES):
            done = False 
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(self.local_actor_critic.x)         # choose action based on inputs into the neural network x, which is the previous action and reward
                #print(self.x)
                #print(action)
                #observation_, reward, done, info = self.env.step(action)  #this uses the environment to get a reward, I'm going to create a class that calculates the reward given action, then import an instance of the class within Agent, then just use e.g. self.local_bandits.calculate_reward(action)
                reward = self.bandits.meta_calculate_reward2(action)
                score += reward 
                self.local_actor_critic.remember(action, reward, self.local_actor_critic.x)
                self.local_actor_critic.form_x(action, reward)
                if t_step % T_MAX == 0:
                    print(sum(self.local_actor_critic.actions))
                    if self.episode_idx % 10:
                        self.action_history.append('sum of actions taken:' + str(sum(self.local_actor_critic.actions)))
                    self.local_actor_critic.clear_memory()
                    done = True
                t_step += 1
                #if t_step % T_MAX == 0:
                    #done = True
            self.episode_idx += 1
            print('episode ', self.episode_idx, 'reward %.1f' % score)
            if self.episode_idx % 10 == 0: 
                first_string = 'episode ', str(self.episode_idx), str('reward %.1f' % score)
                second_string = 'probabilities: ',str(0.1),str(0.8)
                self.terminal_memory.append(first_string+second_string)

        for n in zip(self.terminal_memory, self.action_history):
            print(n)

if __name__ == '__main__':
    lr = 7e-4
    n_actions = 2 
    input_size = [3]
    N_BLOCKS = 150
    T_MAX = 100   # comes from the paper
    EPISODES_PER_BLOCK = 100
    META_GAMES = 500
    actor_critic = A2C(input_size, n_actions)

    #actor_critic.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not
    #next(actor_critic.parameters()).device

    optimizer = optim.RMSprop(actor_critic.parameters(), lr=lr)
    global_ep = 0

    worker = Agent(actor_critic,
                    optimizer,
                    input_size,
                    n_actions,
                    gamma=0.75,
                    lr=lr,
                    bandits = Bandits(),
                    global_ep_idx=global_ep)

    worker.run()


# 1) run again and see if meta learns 
# 2) add timestep into input 
# 3) change loss function to the same as in 2018 paper 
# 4) change probabilities to the same as in 2018 paper 
# essentialy try and simulate 2016 paper and then 2018 paper and then get to what yashar wants (serial change of probabilities)
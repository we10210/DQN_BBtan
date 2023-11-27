from BBTan import BBTan
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import random
import collections
import pickle
from tensorboardX import SummaryWriter

# from torchviz import make_dot

import torch 
from torch import nn
# from torchviz import make_dot, make_dot_from_trace

#tensorboard
# from torch.utils.tensorboard import SummaryWriter
# from model.SUNet import SUNet_model

# 超參數
BATCH_SIZE = 32
LR = 0.1                # learning rate 0.0025
EPSILON = 0.99              # greedy policy # 0.1
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 50    # target update frequency # 10
MEMORY_CAPACITY = 2000
N_ACTIONS = 20 # 角度, range在(- 0.1 ~ -1* (0.9*pi+0.1)), 分為20等分
N_STATES = 57 # 球的位置, cloest brick x, cloest brick y (最靠近ball的row的狀態)

ENV_A_SHAPE = 0 #if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


# class Net(nn.Module):
#     def __init__(self, ):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(N_STATES, 50)
#         self.fc1.weight.data.normal_(0, 0.1)   # initialization
#         self.fc2 = nn.Linear(50, N_ACTIONS)
#         self.fc2.weight.data.normal_(0, 0.1)   # initialization

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.fc2(x)
#         actions_value = F.softmax(x, dim=-1)  # Apply softmax to the last layer
#         return actions_value

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES , 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x) # x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net() #two q model
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0    
        self.memory = collections.deque(maxlen = MEMORY_CAPACITY)                                     # for storing memory
        # self.memory = np.empty((MEMORY_CAPACITY, ), dtype=object)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) # 優化器
        self.loss_func = nn.MSELoss() # loss function 
        self.learning_counter = 0 

    def choose_action(self, x):
        # print("x",x)
        # print("x.shape",x.shape)
        # print("torch.FloatTensor(x)",torch.FloatTensor(x)) 
        # 
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            # print("actions_value",actions_value)
            action_index = torch.max(actions_value, 1)[1].data.numpy()
            # actions_value_pi_angle = -1 * ( F.sigmoid(actions_value) * 0.9 * torch.pi +0.1)
            # actions_value_pi_angle = -1 * ( ((action_index+1) / 10) * 0.9 * torch.pi +0.1)
            # action = actions_value_pi_angle#.data.numpy()
            action = action_index[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
            # print("action",action)
                          
        else:   # random
            # action = np.random.rand(N_ACTIONS) * 0.9 * np.pi * -1 - 0.1
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            # print("action",action)

        return action

    def store_transition(self, s, a, r, s_):
        transition = (s, a, r, s_)  
        self.memory.append(transition)
        self.memory_counter += 1


    def save_transition(self):
        with open("latest_repaly_buffer.pkl", "wb") as f:
            pickle.dump(self.memory, f)
        np.save("memory_counter.npy", self.memory_counter)

    def load_transition(self):
        if os.path.isfile("latest_repaly_buffer.pkl"):
            with open("latest_repaly_buffer.pkl", "rb") as f:
                self.memory = pickle.load(f)
            self.memory_counter = np.load("memory_counter.npy")
        else:
            self.memory_counter = 0                                         
            self.memory = collections.deque(maxlen = MEMORY_CAPACITY)  # initialize memory

    def save_model(self):
        torch.save(self.eval_net.state_dict(), "latest_eval.pth")
        torch.save(self.target_net.state_dict(), "latest_target.pth")

    def load_model(self):
        if os.path.isfile("latest_eval.pth"): 
            self.eval_net, self.target_net = Net(), Net()
            self.eval_net.load_state_dict(torch.load("latest_eval.pth"))
            self.eval_net.train()
            self.target_net.load_state_dict(torch.load("latest_target.pth"))
            self.target_net.train()
        else:
            self.eval_net, self.target_net = Net(), Net()

    def learn(self, cnt):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        self.optimizer.zero_grad()
        # sample batch transitions
        sample_indices = random.sample(range(len(self.memory)), BATCH_SIZE)
        batch = [self.memory[index] for index in sample_indices]
        b_s, b_a, b_r, b_s_ = zip(*batch)
        b_s = torch.tensor(np.array(b_s), dtype=torch.float32)
        b_a = torch.tensor(np.array(b_a), dtype=torch.int64).unsqueeze(1)
        b_r = torch.tensor(np.array(b_r), dtype=torch.int64).unsqueeze(1)
        b_s_ = torch.tensor(np.array(b_s_), dtype=torch.float32)
        # b_s, b_a, b_r, b_s_ = zip(*batch)
        # b_s = torch.tensor(b_s, dtype=torch.float32)
        # b_a = torch.tensor(b_a, dtype=torch.int64).unsqueeze(1)  # Adding an extra dimension for gather
        # b_r = torch.tensor(b_r, dtype=torch.int64).unsqueeze(1)  # Adding an extra dimension for Q learning computation
        # b_s_ = torch.tensor(b_s_, dtype=torch.float32)
        # print('b_s',b_s[0])
        # print('b_a',b_a[0])
        # print('b_r',b_r[0])
        # print('b_s_',b_s_[0])



        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)
        writer.add_scalar("loss", loss, cnt)
        
        loss.backward()
        self.optimizer.step() 
        

if __name__ == "__main__":
    dqn = DQN() # AI
    # dqn.load_transition()
    # dqn.load_model()
    
    writer = SummaryWriter(comment="dqn")
    
    # 印出模型結構
    # print(dqn.eval_net)  

    # 模型可視化
    # sample_input = torch.randn(1, N_STATES)
    # sample_output = dqn.eval_net(sample_input)
    
    print('\nCollecting experience...')

    for i_episode in range(2000):  # 訓練AI和環境玩的次數
        env = BBTan()

        # 從環境（env）中獲取初始狀態（Initial State）
        s = env.get_init_state() 
        # print("s",s)
        # print("s.shape",s.shape)
        # print("here~")
        # stop()


        #total reward in this game
        ep_r = 0 
        step_count = 0
        
        while True:
            step_count += 1
            # print("Step count", step_count, "dqn.memory_counter", dqn.memory_counter)

            a = dqn.choose_action(s) # 選動作

            # take action
            done, r, s_ = env.step(a)
            # done, r, s_ = env.step(a, True) if i_episode == 254 or i_episode == 510 or i_episode == 766 else env.step(a)
            dqn.store_transition(s, a, r, s_) #存進Replay Buffer 
            # print('final_r', r)

            ep_r += r
            
            if done:
                print('Ep: ', i_episode , '| Ep_r: ', ep_r)
                writer.add_scalar("reward", ep_r, i_episode)
                # print('memory_counter', dqn.memory_counter)
                
                if dqn.memory_counter > MEMORY_CAPACITY: #Replay Buffer達到一定的量   

                    dqn.learning_counter += 1  # 在主循環中增加

                    if dqn.learning_counter % 1 == 0:  #每1進行一次學習
                        print("_____!!!!!dqn learn!!!!!_____")
                        dqn.learn(i_episode)
                        EPSILON *= 0.998
                        print("Current exploration rate (EPSILON): {:.4f}".format(EPSILON))
                env.close_game()   
                break

            s = s_

    dqn.save_transition()
    dqn.save_model()


def learn(self, cnt):
    # sample batch transitions
    sample_indices = random.sample(range(len(self.memory)), BATCH_SIZE)
    batch = [self.memory[index] for index in sample_indices]
    b_s, b_a, b_r, b_s_ = zip(*batch)

    # 將狀態、動作、獎勵和下一狀態轉換為Tensors
    b_s = torch.FloatTensor(b_s)  # 將狀態轉換為Tensor
    b_a = torch.LongTensor(b_a)  # 將動作轉換為Tensor
    b_r = torch.FloatTensor(b_r)  # 將獎勵轉換為Tensor
    b_s_ = torch.FloatTensor(b_s_)  # 將下一狀態轉換為Tensor

    # 確保動作的Tensor是正確的形狀
    b_a = b_a.unsqueeze(1)  # 將動作的形狀從 [batch_size] 轉換為 [batch_size, 1]
    b_r = b_r.unsqueeze(1)  # 將獎勵的形狀從 [batch_size] 轉換為 [batch_size, 1]

    q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
    q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
    q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
    loss = self.loss_func(q_eval, q_target)

    writer.add_scalar("loss", loss, cnt)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # target parameter update
    self.learn_step_counter += 1
    if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
        self.target_net.load_state_dict(self.eval_net.state_dict())

from BBTan import BBTan
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 10    # target update frequency
MEMORY_CAPACITY = 2000
N_ACTIONS = 20 # 角度, range在(- 0.1 ~ -1* (0.9*pi+0.1)), 分為20等分
N_STATES = 57 # 球的位置, cloest brick x, cloest brick y (最靠近ball的row的狀態)
print("N_ACTIONsS",N_ACTIONS,"N_STATES",N_STATES)

ENV_A_SHAPE = 0 #if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES , 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net() #two q model
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) # 優化器
        self.loss_func = nn.MSELoss() # loss function 

    def choose_action(self, x):
        # print("x",x)
        # print("x.shape",x.shape)
        # print("torch.FloatTensor(x)",torch.FloatTensor(x))
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
        transition = np.hstack((s, [a, r], s_)) #八維的data 2000 * 8 __ (3+1+3)
        transition = np.array[s, a, r, s_]

        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self, cnt):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        self.optimizer.zero_grad()
        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :] # 32 * 8 array
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)

        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)
        writer.add_scalar("loss", loss, cnt)
        
        loss.backward()
        self.optimizer.step() 

if __name__ == "__main__":
    dqn = DQN() # AI
    
    writer = SummaryWriter(comment="dqn")
    
    # 印出模型結構
    # print(dqn.eval_net)  

    # 模型可視化
    sample_input = torch.randn(1, N_STATES)
    sample_output = dqn.eval_net(sample_input)
    
    print('\nCollecting experience...')

    for i_episode in range(750):  # 訓練AI和環境玩的次數
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
            done, r, s_ = env.step(a) # if i_episode< 254 else env.step(a, True)
            dqn.store_transition(s, a, r, s_) #存進Replay Buffer 

            ep_r += r
            
            if done:
                print('Ep: ', i_episode, '| Ep_r: ', ep_r)
                writer.add_scalar("reward", ep_r, i_episode)
            
                if dqn.memory_counter > MEMORY_CAPACITY: #Replay Buffer達到一定的量 再做訓練 
                    # print("dqn learn")
                    dqn.learn(i_episode)
                    # if done:
                    #     print('Ep: ', i_episode, '| Ep_r: ', round(ep_r, 2))
                break
            s = s_

# 導入需要的函式庫
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

# 超參數
BATCH_SIZE = 32
LR = 0.1   # learning rate 0.0025
EPSILON = 1        # greedy policy # 0.1
GAMMA = 0.9               # reward discount
TARGET_REPLACE_ITER = 100    # target update frequency # 10
MEMORY_CAPACITY = 3200
N_ACTIONS = 20 # 角度, range在(- 0.1 ~ -1* (0.9*pi+0.15)), 分為20等分
N_STATES = 57 # 球的位置, 8*7 array 

# 在BBTan.py中 有註解標示出state action reward是在哪裡

ENV_A_SHAPE = 0 # if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape

# class Net(nn.Module):
#     def __init__(self, ):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(N_STATES , 50)
#         self.fc1.weight.data.normal_(0, 0.1)   # initialization
#         self.out = nn.Linear(50, N_ACTIONS)
#         self.out.weight.data.normal_(0, 0.1)   # initialization

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.relu(x) # x = F.relu(x)
#         actions_value = self.out(x)
#         return actions_value

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(N_STATES, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, N_ACTIONS)

        # He initialization
        # nn.init.kaiming_normal_(self.layer1.weight, nonlinearity='relu')
        # nn.init.kaiming_normal_(self.layer2.weight, nonlinearity='relu')
        # For the output layer, can either use the default initialization or apply a different one if needed

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net() #two q model
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0    
        self.memory = collections.deque(maxlen = MEMORY_CAPACITY)   # for storing memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR) # 優化器
        self.loss_func = nn.MSELoss() # loss function 
        self.learning_counter = 0 

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)

        # random
        if np.random.uniform() < EPSILON:   # greedy
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
            # print("RANDOM action",action)
                          
        else:   
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
            print("action",action)
                          
        return action

    def store_transition(self, s, a, r, s_):
        transition = (s, a, r, s_)
        self.memory.append(transition)
        self.memory_counter += 1
        # print('state', s)
        # print('action', a)
        # print('reward', r)
        # print('nest_state', s_)


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

        self.learn_step_counter += 1
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer.zero_grad() # 位置是否正確

        # sample batch transitions
        sample_indices = random.sample(range(len(self.memory)), BATCH_SIZE)
        batch = [self.memory[index] for index in sample_indices]
        b_s, b_a, b_r, b_s_ = zip(*batch)
        b_s = torch.FloatTensor(b_s)
        b_a = torch.LongTensor(b_a).unsqueeze(1)
        b_r = torch.LongTensor(b_r).unsqueeze(1)
        b_s_ = torch.FloatTensor(b_s_)
        # print('b_s',b_s[0])
        # print('b_a',b_a[0])
        # print('b_r',b_r[0])
        # print('b_s_',b_s_[0])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        print('q_eval', q_eval)
        print('q_rext', q_next)
        print('q_target', q_target)

        writer.add_scalar("loss", loss, cnt)
        loss.backward()
        self.optimizer.step()


        


    # def learn(self, cnt):
    #         # 確保回放記憶庫中有足夠的樣本
    #         if len(self.memory) < BATCH_SIZE:
    #             return
    #         # 更新目標網絡
    #         self.learn_step_counter += 1
    #         if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
    #             self.target_net.load_state_dict(self.eval_net.state_dict())

    #         # 隨機抽取一批樣本
    #         sample_indices = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
    #         batch_memory = np.array(self.memory)[sample_indices]
            
    #         self.optimizer.zero_grad()

    #         # 分解批次記憶
    #         b_s = torch.FloatTensor(batch_memory[:, 0].tolist())
    #         b_a = torch.LongTensor(batch_memory[:, 1].astype(int)).unsqueeze(1)
    #         b_r = torch.FloatTensor(batch_memory[:, 2].tolist()).unsqueeze(1)
    #         b_s_ = torch.FloatTensor(batch_memory[:, 3].tolist())
    #         # print('b_s',b_s[0])
    #         # print('b_a',b_a[0])
    #         # print('b_r',b_r[0])
    #         # print('b_s_',b_s_[0])

    #         # 計算 Q 值
    #         q_eval = self.eval_net(b_s).gather(1, b_a)
    #         q_next = self.target_net(b_s_).detach()

    #         # 計算目標 Q 值
    #         q_target = b_r + GAMMA * q_next.max(1)[0].unsqueeze(1)
    #         print('q_eval', q_eval)
    #         print('q_rext', q_next)
    #         print('q_target', q_target)

    #         # 計算損失
    #         loss = self.loss_func(q_eval, q_target)

    #         # 反向傳播和優化

    #         loss.backward()
    #         self.optimizer.step()


    #         # 更新 TensorBoard
    #         writer.add_scalar("loss", loss.item(), cnt)
            

if __name__ == "__main__":
    # env = BBTan()
    dqn = DQN() # AI
    # dqn.load_transition()
    # dqn.load_model()
    
    writer = SummaryWriter(comment="dqn")
    print('\nCollecting experience...')

    for i_episode in range(1500):  # 訓練AI和環境玩的次數
        env = BBTan()

        # 從環境（env）中獲取初始狀態（Initial State）
        s = env.get_init_state() 

        #total reward in this game
        ep_r = 0 
        step_count = 0
        
        while True:
            step_count += 1
            # print("Step count", step_count, "dqn.memory_counter", dqn.memory_counter)

            a = dqn.choose_action(s) # 選動作

            # take action
            done, r, s_ = env.step(a)
            dqn.store_transition(s, a, r, s_) #存進Replay Buffer 
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
                        EPSILON = max(0.05, EPSILON * 0.995)
                        print("Current exploration rate (EPSILON): {:.4f}".format(EPSILON))
                env.close_game()   
                # env.reset_game()
                break

            s = s_

    dqn.save_transition()
    dqn.save_model()
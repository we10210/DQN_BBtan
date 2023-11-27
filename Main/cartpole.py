import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from BBTan import BBTan
import os

env = BBTan()
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32 # 訓練樣本批次量
n_episodes = 1000 # 訓練回合次數次數

# from google.colab import drive
# drive.mount("/content/drive")
# output_dir = "/content/drive/MyDrive/CartPole"
# if not os.path.exists(output_dir):
#   os.makedirs(output_dir)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen = 2000) # 儲存每一時步的資訊，保留最新的2000筆資料
        self.gamma = 0.95 # 折扣係數
        self.epsilon = 1.0 # 探索率，不依照累積經驗行動而是隨機做出行為。一開始訓練為為1.0，因為毫無經驗可循，等有經驗後才開始調低。
        self.epsilon_decay = 0.995 # 每次降緩探索率的程度
        self.epsilon_min = 0.01 # 探索率下降的最小值
        self.learning_rate = 0.001 # 調整權重的學習率
        self.model = self._build_model() # 建構模型建構模型 method前加入_為private
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation="relu", input_dim = self.state_size))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear")) # 線性
        model.compile(loss="mse", optimizer = Adam(learning_rate=self.learning_rate))
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward # 若該局結束
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state, verbose = 0)[0])) # 貝爾曼方程、動態規劃方程？
            
            target_f = self.model.predict(state, verbose = 0) 
            target_f[0][action] = target # target是基於代理人的實際經驗與下一步狀態所算出來的，比模型預測還較有可信度
            self.model.fit(state, target_f, epochs = 1, verbose = 0) # 訓練

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose = 0)
        return np.argmax(act_values[0])
    
    def save(self, name):
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.load_weights(name)

agent = DQNAgent(state_size, action_size)

for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    time = 0
    while not done:
        # env.render()
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        if not done:
            reward = reward
        else:
            reward = -10

        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: ", (e + 1), "/", n_episodes, "\tscore: ", time, "\te: %.2f" % agent.epsilon, sep = "")
        
        time += 1
    if (len(agent.memory) % (batch_size / 2) == 0) and len(agent.memory) >= (batch_size * 10):
        agent.train(batch_size)
    
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    
    if e % 50 == 0:
        agent.save("model_py/weights_" + "{:04d}".format(e) + ".hdf5")
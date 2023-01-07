from utils_DQN import *
from tqdm import tqdm

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class DQN:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.policy_net = self.build_model()

        self.criterion = torch.nn.MSELoss()  
        self.optimizer = optim.RMSprop(params=self.policy_net.parameters(), lr=0.001)

        self.policy_net.to(self.device)
        self.memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BATCH_SIZE, input_dims=8)
        self.update_target_net_counter = 0

    def build_model(self):

        class SimpleNet(nn.Module):
            def __init__(self):
                super(SimpleNet, self).__init__()
                self.layer1 = nn.Linear(8, 512)
                self.layer2 = nn.Linear(512, 512)
                self.layer3 = nn.Linear(512, 4)
            
            def forward(self, x):
                x = self.layer1(x)
                x = torch.relu(x)

                x = self.layer2(x)
                x = torch.relu(x)

                x = self.layer3(x)
                return x               

        return SimpleNet()

    def memorize(self, state, action, reward, next_state, terminated, truncated):
        self.memory.memorize(state, action, reward, next_state, terminated, truncated)

    def sample(self):
        return self.memory.sample()

    def train(self):
        if len(self.memory) < BATCH_SIZE:
            return
        self.policy_net.train()

        states, actions, rewards, next_states, dones = self.sample()
        
        states, rewards, next_states, dones = \
                torch.Tensor(states).to(self.device), \
                torch.Tensor(rewards).to(self.device), \
                torch.Tensor(next_states).to(self.device), \
                torch.Tensor(dones).type(torch.bool).to(self.device) 
        self.optimizer.zero_grad()

        all_indexes = np.arange(BATCH_SIZE, dtype=np.int32)
        output = self.policy_net(states)[all_indexes, actions] # get q-values of chosen actions for the whole batch
        target = self.policy_net(next_states)
        target[dones] = 0.0
        target = rewards + DISCOUNT * torch.max(target, dim=1)[0]

        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()

    def get_q(self, state):
        state = torch.Tensor(state).to(self.device)
        return self.policy_net(state).cpu().detach().numpy()

    def save_model(self, epoch, ep_reward, highest_reward):
        if ep_reward < highest_reward:
            return highest_reward

        path = "./" + str(ep_reward)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.criterion,
            'reward': ep_reward
            }, path)
        
        with open('cur_highest.txt', 'w') as f:
            f.write(str(ep_reward))
        
        return ep_reward
         

    def load_model(self, best_model_path=None, highest_reward_path=None):

        if (best_model_path != None and highest_reward_path != None):
            raise Exception("Can only load one model at a time.")

        if highest_reward_path != None:
            path = highest_reward_path
        else:
            path = best_model_path
    
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        self.criterion = checkpoint['loss']
        self.policy_net.eval()
        return epoch

    def act(self, env, action):
        return env.step(action)



def learn(env, agent, epsilon):
    try:
        with open('cur_highest.txt') as f:
            highest_reward = float(f.readline())
    except:
        with open('cur_highest.txt', 'w') as f:
            f.write("-1000")
            highest_reward = -1000
    

    for i in tqdm(range(1, EPISODES), ascii=True, unit="episodes"):
        ep_reward = 0
        state, _ = env.reset()

        done = False

        while not done:
            
            if np.random.random() > epsilon:
                action = np.argmax(agent.get_q(state))
            else:
                action = np.random.randint(low=0, high=4)
            
            new_state, reward, terminated, truncated, info = agent.act(env, action)

            done = terminated or truncated
            ep_reward += reward

            agent.memorize(state, action, reward, new_state, terminated, truncated)
            agent.train()

            state = new_state

            if done: break

        if epsilon > MIN_EPSILON:
            epsilon -= EPSILON_DECAY
            epsilon = max(MIN_EPSILON, epsilon)

        highest_reward = agent.save_model(i, ep_reward, highest_reward)


def test(env, agent):

    avg_score = list()

    with torch.no_grad():
        for _ in range(10):
            state, _ = env.reset()
            done = False
            action, reward, info = None, None, None
            ep_reward = 0
            local_reward = list()
            for _ in range(3):
                counter = 0
                while not done:
                    counter += 1
                    action = np.argmax(agent.get_q(state))
                    state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_reward += reward

                    if done: 
                        break

                local_reward.append(ep_reward)

            avg_score.append(sum(local_reward)/len(local_reward))

        return sum(avg_score)/len(avg_score)

def learn_loop():
    epsilon = 1
    learn_env = gym.make("LunarLander-v2")
    agent = DQN()

    try:
        agent.load_model(best_model_path='./best_model')
    except:
        pass
    learn(learn_env, agent, epsilon)

    learn_env.close()

def test_loop():

    test_env = gym.make("LunarLander-v2")
    agent = DQN()
    with open('cur_highest.txt') as f:
        highest_reward_path = f.readline()
    agent.load_model(highest_reward_path=highest_reward_path)
    highest_reward = test(test_env, agent)

    # test best model
    try:
        agent = DQN()
        agent.load_model(best_model_path="./best_model")
        best_model = test(test_env, agent)
        
        if highest_reward > best_model:
            torch.save(torch.load(str(highest_reward_path)), "./best_model")
            print("Found new best model!")
    except:
        torch.save(torch.load(str(highest_reward_path)), "./best_model")
         
    test_env.close()

# test human
def test_human():
    test_env =gym.make("LunarLander-v2", render_mode='human')

    # test best model
    agent = DQN()
    agent.load_model(best_model_path="./best_model")
    best_model = test(test_env, agent)
    test_env.close()

def main():
    for _ in range(5):
        learn_loop()
        test_loop()
    test_human()

MEMORY_SIZE = 5000
BATCH_SIZE = 64
EPISODES = 1000
DISCOUNT = 0.95 # gamma
EPSILON_DECAY = 0.001
MIN_EPSILON = 0.01

main()




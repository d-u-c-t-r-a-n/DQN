import numpy as np

class ReplayMemory:
    def __init__(self, size, batch_size, input_dims) -> None:
        self.length = 0
        self.size = size
        self.batch_size = batch_size
        self.states = np.zeros((self.size, input_dims), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.next_states = np.zeros((self.size, input_dims), dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.bool8)
    
    def memorize(self, state, action, reward, next_state, terminated, truncated):
        index = self.length % self.size
        self.length += 1

        self.states[index] = state
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_states[index] = next_state
        self.dones[index] = (terminated or truncated)

    def sample(self):
        sample_length = min(self.length, self.size)
        batch = np.random.choice(sample_length, self.batch_size, replace=False)
        # print(self.dones[batch].dtype)
        return  self.states[batch], \
                self.actions[batch], \
                self.rewards[batch], \
                self.next_states[batch], \
                self.dones[batch]
    
    def __len__(self):
        return self.length


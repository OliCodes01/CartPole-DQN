from collections import deque
import random


class ReplayMemory:
    def __init__(self, length):
        self.memory = deque([], length)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def append(self, experience):
        self.memory.append(experience)

    def __len__(self):
        return len(self.memory)

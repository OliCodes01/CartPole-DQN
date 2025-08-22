import random

import gymnasium as gym
import torch.nn as nn
import yaml
from util import tensor_dict, replaymemory as rm
from util.DQN import DQN
import numpy as np
import torch


class Training:
    def __init__(self, params):
        # Env
        self.env_name = params['env_name']
        self.env = gym.make(self.env_name)

        # Replay memory
        self.replay_memory = rm.ReplayMemory(params['replay_memory'])

        # Networks
        self.policy_network = DQN(self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_network = DQN(self.env.observation_space.shape[0], self.env.action_space.n)

        # DQN data
        self.learning_rate = params['learning_rate']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.epsilon_end = params['epsilon_end']
        self.batch_size = params['batch_size']
        self.tensor_dict = tensor_dict.TENSOR_DICT

        # Back propagation
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)

        # Run requirements
        self.n_episodes = params['n_episodes']

        # Training data
        self.total_actions = 0
        self.reward_list = []

    def run_training(self):
        for x in range(self.n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Epsilon greedy logic
                if self.epsilon > random.uniform(0, 1):
                    action = self.env.action_space.sample()
                else:
                    state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
                    action = torch.argmax(self.policy_network(state_tensor).detach()).item()

                new_state, reward, terminated, truncated, _ = self.env.step(action)
                done = truncated or terminated

                self.replay_memory.append([state, action, reward, new_state, done])
                self.total_actions += 1
                episode_reward += reward

                state = new_state

                # Copy policy networks weights and biases onto target network every 2000 steps
                if self.total_actions % 2000 == 0:
                    self.target_network.load_state_dict(self.policy_network.state_dict())

                # If replay memory's experiences are equal or above 500, optimize the network every 4 steps
                if len(self.replay_memory) >= 500 and self.total_actions % 4 == 0:
                    self.optimize_network()

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            self.reward_list.append(episode_reward)
            if (x + 1) % 100 == 0:
                print(f"Episode {x + 1}: Reward = {episode_reward}")
                print(f"Episode {x - 100} -> {x}: Reward = {sum(self.reward_list[x - 100:x]) / 100}")
                print(f"Epsilon: {self.epsilon}")
        torch.save(self.policy_network.state_dict(), "networks/policy_network.pth")
        torch.save(self.target_network.state_dict(), "networks/target_network.pth")

    def optimize_network(self):
        # Check if replay memory has enough experiences
        if len(self.replay_memory) < self.batch_size:
            return

        # Get all batches in tensor format
        batch_tensors = self.get_batch_tensors()

        # Get Q-values for the state batch (this will be both actions)
        all_q_values = self.policy_network(batch_tensors['state_batch'])

        # Get only the Q-values for the actions taken
        action_taken_q_values = all_q_values.gather(1, batch_tensors['action_batch'].unsqueeze(1)).squeeze(1)

        # Predict target Q-values with the target network (1 Q-value per next state)
        target_q_values = self.get_target_q_values(batch_tensors)

        # Compute loss
        loss = self.loss(action_taken_q_values, target_q_values)

        # Clear the gradients
        self.optimizer.zero_grad()

        # Back propagation to calculate the gradients
        loss.backward()

        # Update the weights and biases using the calculated gradients
        self.optimizer.step()

    def get_target_q_values(self, batch_tensors):

        with torch.no_grad():
            # Get the target network's prediction for the best action in the new state based on the highest Q-value
            # (stores only the max Q-value)
            next_q_values = self.target_network(batch_tensors['new_state_batch']).max(1)[0].detach()
            # Bellman's equation to calculate the target Q-values
            target_q_values = batch_tensors['reward_batch'] + self.gamma * next_q_values * (
                    1 - batch_tensors['done_batch'])

        return target_q_values

    def get_batch_tensors(self):
        # Returns list of np arrays
        batch_arrays = self.get_batch_arrays()

        tensors = {}
        # Convert to tensors
        for key, arr in batch_arrays.items():
            tensors[key] = torch.tensor(arr, dtype=self.tensor_dict[key])
        return tensors

    def get_batch_arrays(self):
        batch = self.replay_memory.sample(self.batch_size)
        state, action, reward, new_state, done = map(np.vstack, zip(*batch))

        batch_arrays = {
            "state_batch": state,
            "action_batch": action.squeeze(1),
            "reward_batch": reward.squeeze(1),
            "new_state_batch": new_state,
            "done_batch": done.squeeze(1),
        }

        return batch_arrays

    def show_trained_network_in_action(self):
        self.env = gym.make(self.env_name, render_mode='human')
        for x in range(20):
            state, _ = self.env.reset()
            done = False
            while not done:
                state_tensor = torch.tensor(state[np.newaxis, :], dtype=torch.float32)
                action = torch.argmax(self.policy_network(state_tensor).detach()).item()

                new_state, reward, terminated, truncated, _ = self.env.step(action)
                done = truncated or terminated

                state = new_state


if __name__ == '__main__':
    with open('util/param_config.yml', 'r') as file:
        params_config = yaml.safe_load(file)['cartpole1']
    experiment = Training(params_config)
    experiment.run_training()
    experiment.show_trained_network_in_action()

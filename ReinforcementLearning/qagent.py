import numpy as np
import pandas as pd

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.Q_table = np.zeros((state_size, action_size))

    def map_state_to_index(self, state):
        # Map state to index
        # Assuming state is a list or a tuple
        state_hash = hash(tuple(state))
        return state_hash % self.state_size

    def map_action_to_index(self, action):
        # Map action to index
        # Assuming action is an integer
        return int(action) % self.action_size

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # Explore: choose a random action
            return np.random.randint(self.action_size)
        else:
            # Exploit: choose the action with the highest Q-value for the current state
            state_index = self.map_state_to_index(state)
            return np.argmax(self.Q_table[state_index])

    def update_q_table(self, state, action, reward, next_state):
        # Map state and action to indices
        state_index = self.map_state_to_index(state)
        action_index = self.map_action_to_index(action)

        # Q-learning update rule
        next_state_index = self.map_state_to_index(next_state)
        td_target = reward + self.discount_factor * np.max(self.Q_table[next_state_index])
        td_error = td_target - self.Q_table[state_index, action_index]
        self.Q_table[state_index, action_index] += self.learning_rate * td_error

    def get_action(self, state):
        state_index = self.map_state_to_index(state)
        return np.argmax(self.Q_table[state_index])


    def train(self, state, action, reward, next_state):
        self.update_q_table(state, action, reward, next_state)

    def get_q_table(self):
        return self.Q_table

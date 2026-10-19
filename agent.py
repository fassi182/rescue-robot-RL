import numpy as np

class QLearningAgent:
    """
    Tabular Q-learning algorithm implementation from scratch.
    """
    def __init__(self, state_shape, action_size, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
        self.state_shape = state_shape
        self.action_size = action_size
        self.lr = lr                       # Learning rate (alpha)[cite: 2]
        self.gamma = gamma                 # Discount factor (gamma)[cite: 2]
        self.epsilon = epsilon             # Exploration rate[cite: 2]
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        
        # Initialize Q-Table with zeros[cite: 1, 2]
        self.q_table = np.zeros((*state_shape, action_size))

    def state_to_index(self, state):
        return tuple(state)

    def choose_action(self, state):
        """Epsilon-greedy policy decision[cite: 2]."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        
        state_idx = self.state_to_index(state)
        return np.argmax(self.q_table[state_idx])

    def update(self, state, action, reward, next_state, done):
        """Q-value Bellman equation update[cite: 2]."""
        s_idx = self.state_to_index(state)
        ns_idx = self.state_to_index(next_state)
        
        best_next_action = np.argmax(self.q_table[ns_idx])
        td_target = reward + (0 if done else self.gamma * self.q_table[ns_idx][best_next_action])
        td_error = td_target - self.q_table[s_idx][action]
        
        # Q(s,a) update
        self.q_table[s_idx][action] += self.lr * td_error

    def decay_epsilon(self):
        """Decay exploration rate after each episode[cite: 2]."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save Q-table weights to .npy file[cite: 1, 2]."""
        np.save(filepath, self.q_table)

    def load(self, filepath):
        """Load Q-table weights."""
        self.q_table = np.load(filepath)
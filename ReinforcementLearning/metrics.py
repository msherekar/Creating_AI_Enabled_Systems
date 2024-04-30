import numpy as np
import matplotlib.pyplot as plt


# The Metrics class provides a way to calculate and visualize different metrics related to the rewards obtained during
# training. It calculates cumulative rewards, average reward, and discounted rewards over time. The class also includes
# methods to plot these metrics for visualization purposes. The example demonstrates how to use the Metrics class to
# analyze the rewards obtained during training and visualize the results.

class Metrics:
    def __init__(self, rewards, discount_factor):
        self.rewards = rewards
        self.discount_factor = discount_factor
        self.cumulative_rewards = self.calculate_cumulative_rewards()
        self.average_reward = np.mean(rewards)
        self.discounted_rewards = self.calculate_discounted_rewards()

    def calculate_cumulative_rewards(self):
        cumulative_rewards = np.cumsum(self.rewards)
        return cumulative_rewards

    def calculate_discounted_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards, dtype=np.float64)
        for t in range(len(self.rewards)):
            discounted_reward = 0
            discount = 1
            for k in range(t, len(self.rewards)):
                discounted_reward += self.rewards[k] * discount
                discount *= self.discount_factor
            discounted_rewards[t] = discounted_reward
        return discounted_rewards

    def calculate_average_reward(self):
        return np.mean(self.rewards)

    def plot_cumulative_rewards(self):
        plt.plot(self.cumulative_rewards)
        plt.title('Cumulative Rewards Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.savefig('cumulative_rewards.png')
        plt.show()
        # save

    def plot_discounted_rewards(self):
        plt.plot(self.discounted_rewards)
        plt.title('Discounted Rewards Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Discounted Reward')
        plt.savefig('discounted_rewards.png')
        plt.show()
        # save

    def plot_average_reward(self):
        plt.plot(self.calculate_average_reward())
        plt.title('Average Reward Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.savefig('average_reward.png')
        plt.show()
        # save


if __name__ == '__main__':
    # Example usage of the Metrics class
    # Generate random rewards for demonstration purposes
    rewards = np.random.randint(0, 10, 100)

    # Create an instance of the Metrics class
    metrics = Metrics(rewards, discount_factor=0.9)

    # Print and plot the metrics
    print("Cumulative Rewards:", metrics.cumulative_rewards[-1])
    print("Average Reward:", metrics.average_reward)
    print("Discounted Reward:", metrics.discounted_rewards[0])
    metrics.plot_cumulative_rewards()
    metrics.plot_discounted_rewards()

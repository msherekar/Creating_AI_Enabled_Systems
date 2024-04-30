from preprocess import load_data, merge_data, calculate_rewards, preprocess_data
from sar import generate_states, generate_actions, get_reward, calculate_reward
from qagent import QLearningAgent
from metrics import Metrics
from train import train_agent
from report import save_report
import numpy as np
import datetime
import pickle

# Load data
userbase_file = 'userbase.csv'
sent_file = 'sent_emails.csv'
responded_file = 'responded.csv'

# Setup dataframes
userbase, sent, responded = load_data(userbase_file, sent_file, responded_file)

# Merge data
important_features = ['Gender', 'Type', 'Age', 'Tenure']
merged_data = merge_data(sent, userbase, responded, important_features)

# Calculate rewards
merged_data = calculate_rewards(merged_data)

# STATES
states = generate_states(merged_data)

# Initialize a dictionary to store rewards for each state
state_rewards = {state: get_reward(state, merged_data, states) for state in states}

# ACTIONS
actions = generate_actions(merged_data)

# Parameters
state_size = len(states)
action_size = len(actions)

# Initialize the Q-learning agent
agent = QLearningAgent(state_size, action_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)

# Training
training_rewards, episodes_to_convergence = train_agent(agent, states, merged_data)

# Calculate discounted rewards
discounted_rewards = [reward * (0.9 ** i) for i, reward in enumerate(training_rewards)]

# Metrics
metrics = Metrics(training_rewards, discount_factor=0.9)

# Save the trained Q-table

Q_table = agent.get_q_table()

# DOWNLOAD THE AGENT
with open('q_table.pkl', 'wb') as f:
    pickle.dump(Q_table, f)

# **************************** RESULTS ****************************

# Print and plot the metrics
print("Cumulative Rewards:", metrics.cumulative_rewards[-1])
print("Average Reward:", metrics.average_reward)
print("Discounted Reward:", metrics.discounted_rewards[0])
metrics.plot_cumulative_rewards()
metrics.plot_discounted_rewards()

if episodes_to_convergence is not None:
    print("Episodes to Convergence:", episodes_to_convergence)
else:
    print("Convergence not reached within the maximum number of episodes.")

# Report
# Define the report file name
report_file = datetime.datetime.now().strftime("training_%Y-%m-%d_%H-%M-%S.txt")

# Define the training metrics
training_metrics = {
    "Cumulative Rewards": metrics.cumulative_rewards[-1],
    "Average Reward": metrics.average_reward,
    "Discounted Reward":  metrics.discounted_rewards[0],
    "Episodes to converge": episodes_to_convergence if episodes_to_convergence is not None else "Not converged within "
                                                                                                "the maximum number "
                                                                                                "of episodes"
}
# Define the details to be included in the report
report_details = {

    "Name of files": [userbase_file, sent_file, responded_file],
    "Features used": important_features,
    "Length of states": len(states),
    "Learning Rate": 0.1,
    "Discount factor": 0.9,
    "Epsilon": 0.1
}
# Call the function to save the report
save_report(report_file, report_details, training_metrics)



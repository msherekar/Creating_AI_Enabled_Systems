# LIBRARIES
import pandas as pd
import numpy as np
import pickle
from itertools import product
from qagent import QLearningAgent

# DATA
userbase = pd.read_csv('userbase.csv')
sent = pd.read_csv('sent_emails.csv')
responded = pd.read_csv('responded.csv')
responded = responded.drop_duplicates()

# MERGE DATA
# sent + userbase
merged_data = pd.merge(sent, userbase, on='Customer_ID', how='left')

# Fill in additional columns based on userbase information
merged_data['Gender'] = merged_data['Gender'].fillna(userbase.set_index('Customer_ID')['Gender'])
merged_data['Type'] = merged_data['Type'].fillna(userbase.set_index('Customer_ID')['Type'])
merged_data['Email_Address'] = merged_data['Email_Address'].fillna(userbase.set_index('Customer_ID')['Email_Address'])
merged_data['Age'] = merged_data['Age'].fillna(userbase.set_index('Customer_ID')['Age'])
merged_data['Tenure'] = merged_data['Tenure'].fillna(userbase.set_index('Customer_ID')['Tenure'])

# Drop rows with missing values
merged_data = merged_data.dropna() # CHECK AGAIN LATER
# add responded
merged_data = pd.merge(merged_data, responded, on=['Customer_ID'], how='left') # meger on responded date as well.

# REWARDS
merged_data['Reward'] = np.where(merged_data['Sent_Date'] == merged_data['Responded_Date'], 1, 0)

# Fix missing dates
placeholder_date = pd.to_datetime('1900-01-01') # placeholder date
merged_data['Responded_Date'].fillna(placeholder_date, inplace=True)

# Renaming columns
merged_data.rename(columns={'SubjectLine_ID_x': 'SubLine_Sent'}, inplace=True)
merged_data.rename(columns={'SubjectLine_ID_y': 'SubLine_Responded'}, inplace=True)
merged_data['SubLine_Responded'].fillna(-1, inplace=True)

# STATES
states=list(product(merged_data['Gender'].unique(), merged_data['Type'].unique(),merged_data['Age'].unique(), merged_data['Tenure'].unique()))
# Initialize a dictionary to store rewards for each state
state_rewards = {}

# Iterate over each state
for state in states:
    # Calculate the hash value of the state tuple and get the corresponding index
    state_index = hash(tuple(state)) % len(states)

    # Retrieve the reward from the merged_data DataFrame based on the state index
    reward = merged_data.loc[state_index, 'Reward']

    # Store the reward in the state_rewards dictionary
    state_rewards[state] = reward

# ACTIONS
actions = merged_data['SubLine_Sent'].unique()

# TRAINING
state_size = len(states)  # Assuming states is a list of states
action_size = len(actions)  # Assuming actions is a list of unique actions

# Initialize the Q-learning agent
agent = QLearningAgent(state_size, action_size, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)

# Define your training loop
num_episodes = 1000  # Number of training episodes

for episode in range(num_episodes):
    # Iterate over state-action pairs
    for i in range(len(states) - 1):
        # Get current state and action
        state = states[i]
        state_index = hash(tuple(state)) % len(states)
        action = merged_data.loc[state_index, 'SubLine_Sent']
        reward = merged_data.loc[state_index, 'Reward']

        # Get next state
        next_state = states[i + 1]

        # Train the agent
        agent.train(state, action, reward, next_state)

# Get the learned Q-table
Q_table = agent.get_q_table()

# DOWNLOAD THE AGENT
with open('q_table.pkl', 'wb') as f:
    pickle.dump(Q_table, f)





# State, Action, Report
from itertools import product


def generate_states(data):
    """
    Generate states based on unique combinations of Gender, Type, Age, and Tenure from the given data.
    """
    states = list(
        product(data['Gender'].unique(), data['Type'].unique(), data['Age'].unique(), data['Tenure'].unique()))
    return states


def generate_actions(data):
    """
    Generate actions based on unique SubLine_Sent values from the given data.
    """
    actions = data['SubLine_Sent'].unique()
    return actions


def get_reward(state, merged_data, states):
    """
    Get the reward for the given state from the merged_data DataFrame.
    For adding reward feature
    """
    state_index = hash(tuple(state)) % len(states)
    reward = merged_data.loc[state_index, 'Reward']
    return reward


# Define your reward function
def calculate_reward(state):
    # for calculating reward during training
    return merged_data.loc[hash(tuple(state)) % len(states), 'Reward']


# Get subjectline for a given state
def get_subject_line(state, merged_data, states):
    """
    Get the suggested subject line for the given state from the merged_data DataFrame.
    """
    state_index = hash(tuple(state)) % len(states)
    subject_line = merged_data.loc[state_index, 'SubLine_Sent']
    return subject_line

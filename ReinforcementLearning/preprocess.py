# class to preprocess a .csv file and obtain states to be fed into the Q-table.

import pandas as pd
import numpy as np


def preprocess_data(file_path):
    # Load the data
    data = pd.read_csv(file_path)

    # Drop rows with missing values
    data = data.dropna()

    # Define states
    states = []

    # Iterate over rows
    for index, row in data.iterrows():
        # Extract features
        gender = row['Gender']
        user_type = row['Type']
        age = row['Age']
        tenure = row['Tenure']

        # Define state
        state = (gender, user_type, age, tenure)

        # Append state to list of states
        states.append(state)

    return states


if __name__ == '__main__':
    file_path = 'userbase.csv'
    states = preprocess_data(file_path)
    print(states)

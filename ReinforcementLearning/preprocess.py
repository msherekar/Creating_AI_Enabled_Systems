# class to preprocess a .csv file and obtain states to be fed into the Q-table.

import pandas as pd
import numpy as np


def load_data(userbase_file, sent_file, responded_file):
    """
    Load the userbase, sent emails, and responded emails data from CSV files.
    """
    userbase = pd.read_csv(userbase_file)
    sent = pd.read_csv(sent_file)
    responded = pd.read_csv(responded_file).drop_duplicates()
    return userbase, sent, responded


def merge_data(sent, userbase, responded, important_features):
    """
    Merge the sent emails data with the userbase and responded emails data,
    keeping only the important features along with Customer ID.
    """
    # Select important features along with Customer ID
    userbase_selected = userbase[['Customer_ID'] + important_features]

    # Merge the selected userbase data with sent emails
    merged_data = pd.merge(sent, userbase_selected, on='Customer_ID', how='left')

    # Fill missing values with corresponding values from userbase
    for feature in important_features:
        merged_data[feature] = merged_data[feature].fillna(userbase_selected.set_index('Customer_ID')[feature])

    # Merge with responded emails
    merged_data = pd.merge(merged_data, responded, on=['Customer_ID'], how='left')

    return merged_data



def calculate_rewards(data):
    """
    Calculate rewards based on whether a customer responded to an email or not.
    """
    data['Reward'] = np.where(data['Sent_Date'] == data['Responded_Date'], 1, 0)
    data['Responded_Date'].fillna(pd.to_datetime('1900-01-01'), inplace=True)
    data.rename(columns={'SubjectLine_ID_x': 'SubLine_Sent', 'SubjectLine_ID_y': 'SubLine_Responded'}, inplace=True)
    data['SubLine_Responded'].fillna(-1, inplace=True)
    return data


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




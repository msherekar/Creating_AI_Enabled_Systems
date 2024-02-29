import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.tree import DecisionTreeClassifier
import datetime


class Fraud_Dataset:
    def __init__(self, data_path, k_folds=5, random_state=42):
        """
        Initialize the Fraud_Dataset class.

        Parameters:
        - data_path (str): Path to the dataset file.
        - k_folds (int): Number of folds for k-fold cross-validation.
        - random_state (int): Random state for reproducibility.
        """
        self.data_path = data_path
        self.k_folds = k_folds
        self.random_state = random_state
        self.data = None
        self.split_indices = None
        self.groups = None
        self.reports = []

    def load_data(self):
        """
        Load dataset from the given data path.
        """
        self.data = pd.read_csv(self.data_path)

    def balance_data(self, n=1000):
        """
        Balance the dataset by oversampling the minority class (fraud) to match the majority class (non-fraud).
        """
        # Separate the fraud and non-fraud rows
        fraud_df = self.data[self.data['is_fraud'] == 1]
        non_fraud_df = self.data[self.data['is_fraud'] == 0]

        # Sample non-fraud rows to match the number of fraud rows
        sampled_non_fraud_df = non_fraud_df.sample(n=n, random_state=self.random_state) # soft code value for 5000

        # Sample fraud rows to match the number of fraud rows
        sampled_fraud_df = fraud_df.sample(n=n, random_state=self.random_state) # soft code value for 5000


        # Concatenate the fraud and sampled non-fraud rows
        self.data = pd.concat([sampled_fraud_df, sampled_non_fraud_df])

        # Shuffle the concatenated DataFrame
        self.data = self.data.sample(frac=1, random_state=self.random_state)

    def split_data(self):
        """
        Generate k-fold splits.
        """
        # Load data if not already loaded
        if self.data is None:
            self.load_data()

        # Assuming 'cc_num' is the column representing the unique group
        self.groups = self.data['cc_num']

        # Initialize GroupKFold with the number of splits
        group_kfold = GroupKFold(n_splits=self.k_folds)

        # Generate splits
        self.split_indices = list(group_kfold.split(self.data, groups=self.groups))

    def get_training_dataset(self, fold):
        """
        Return the training dataset for a specific fold.
        """
        if self.data is None or self.split_indices is None:
            self.load_data()
            self.balance_data()
            self.split_data()
        train_indices, _ = self.split_indices[fold]
        return self.data.iloc[train_indices]

    def get_validation_dataset(self, fold):
        """
        Return the validation dataset for a specific fold.
        """
        if self.data is None or self.split_indices is None:
            self.load_data()
            self.balance_data()
            self.split_data()
        _, val_indices = self.split_indices[fold]
        return self.data.iloc[val_indices]

    def get_testing_dataset(self):
        """
        Return the entire dataset as the testing dataset.
        """
        if self.data is None:
            self.load_data()
            self.balance_data()
        return self.data
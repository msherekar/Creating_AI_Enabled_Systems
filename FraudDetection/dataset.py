import pandas as pd
from sklearn.model_selection import GroupKFold

class Fraud_Dataset:
    def __init__(self, data_path: str, k_folds: int = 5, random_state: int = 42, balance_samples: int = 1000):
        """
        Initialize the Fraud_Dataset class.

        Parameters:
        - data_path (str): Path to the dataset file.
        - k_folds (int): Number of folds for k-fold cross-validation.
        - random_state (int): Random state for reproducibility.
        - balance_samples (int): Number of samples to balance the classes.
        """
        self.data_path = data_path
        self.k_folds = k_folds
        self.random_state = random_state
        self.balance_samples = balance_samples
        self.data = None
        self.split_indices = None
        self.groups = None

    def load_data(self):
        """
        Load dataset from the given data path.
        """
        self.data = pd.read_csv(self.data_path)

    def balance_data(self):
        """
        Balance the dataset by oversampling the minority class (fraud) to match the majority class (non-fraud).
        """
        if self.data is None:
            self.load_data()

        fraud_df = self.data[self.data['is_fraud'] == 1]
        non_fraud_df = self.data[self.data['is_fraud'] == 0]

        sampled_non_fraud_df = non_fraud_df.sample(n=self.balance_samples, random_state=self.random_state)
        sampled_fraud_df = fraud_df.sample(n=self.balance_samples, random_state=self.random_state)

        self.data = pd.concat([sampled_fraud_df, sampled_non_fraud_df]).sample(frac=1, random_state=self.random_state)

    def split_data(self):
        """
        Generate k-fold splits.
        """
        if self.data is None:
            self.load_data()

        self.groups = self.data['cc_num']
        group_kfold = GroupKFold(n_splits=self.k_folds)
        self.split_indices = list(group_kfold.split(self.data, groups=self.groups))

    def get_training_validation_datasets(self, fold):
        """
        Return the training and validation datasets for a specific fold.
        """
        if self.data is None or self.split_indices is None:
            self.load_data()
            self.balance_data()
            self.split_data()

        train_indices, val_indices = self.split_indices[fold]
        train_dataset = self.data.iloc[train_indices]
        val_dataset = self.data.iloc[val_indices]
        
        return train_dataset, val_dataset

    def get_testing_dataset(self):
        """
        Return the entire dataset as the testing dataset.
        """
        if self.data is None:
            self.load_data()
            self.balance_data()
        return self.data

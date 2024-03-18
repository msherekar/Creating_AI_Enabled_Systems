import numpy as np

class Object_Detection_Dataset:
    def __init__(self, data):
        """
        Initialize the Object_Detection_Dataset class with the provided data.

        Args:
            data (numpy.ndarray): The dataset to be split.
        """
        self.data = data

    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split the dataset into training, validation, and testing sets.

        Args:
            train_ratio (float): The ratio of training data. Default is 0.7.
            val_ratio (float): The ratio of validation data. Default is 0.15.
            test_ratio (float): The ratio of testing data. Default is 0.15.

        Returns:
            tuple: A tuple containing training, validation, and testing datasets.
        """
        data_size = len(self.data)
        indices = np.arange(data_size)
        np.random.shuffle(indices)

        train_end = int(data_size * train_ratio)
        val_end = int(data_size * (train_ratio + val_ratio))

        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        train_data = self.data[train_indices]
        val_data = self.data[val_indices]
        test_data = self.data[test_indices]

        return train_data, val_data, test_data

    def k_fold_split(self, k):
        """
        Split the dataset into k folds.

        Args:
            k (int): The number of folds for cross-validation.

        Returns:
            list: A list of tuples containing indices for each fold.
        """
        data_size = len(self.data)
        fold_size = data_size // k
        remainder = data_size % k

        indices = np.arange(data_size)
        np.random.shuffle(indices)

        fold_indices = []
        start = 0
        for i in range(k):
            end = start + fold_size
            if i < remainder:
                end += 1
            fold_indices.append(indices[start:end])
            start = end

        return fold_indices

    def get_training_dataset(self, k_fold=False, k=5, fold_index=None):
        """
        Get the training dataset.

        Args:
            k_fold (bool): Whether to perform k-fold cross validation. Default is False.
            k (int): The number of folds for cross validation. Default is 5.
            fold_index (int): The index of the fold to retrieve. Default is None.

        Returns:
            numpy.ndarray: The training dataset.
        """
        if k_fold:
            # Perform k-fold cross validation
            if fold_index is None:
                raise ValueError("fold_index must be provided for k-fold cross-validation.")
            fold_indices = self.k_fold_split(k)
            if fold_index < 0 or fold_index >= k:
                raise ValueError("fold_index must be within the range [0, k-1].")
            training_indices = np.concatenate([fold_indices[i] for i in range(k) if i != fold_index])
            return self.data[training_indices]
        else:
            # Return the entire training dataset
            return self.data

    def get_validation_dataset(self, k_fold=False, k=5, fold_index=None):
        """
        Get the validation dataset.

        Args:
            k_fold (bool): Whether to perform k-fold cross validation. Default is False.
            k (int): The number of folds for cross validation. Default is 5.
            fold_index (int): The index of the fold to retrieve. Default is None.

        Returns:
            numpy.ndarray: The validation dataset.
        """
        if k_fold:
            # Perform k-fold cross validation
            if fold_index is None:
                raise ValueError("fold_index must be provided for k-fold cross-validation.")
            fold_indices = self.k_fold_split(k)
            if fold_index < 0 or fold_index >= k:
                raise ValueError("fold_index must be within the range [0, k-1].")
            return self.data[fold_indices[fold_index]]
        else:
            # Return the entire validation dataset
            return self.data

    def get_testing_dataset(self):
        """
        Get the testing dataset.

        Returns:
            numpy.ndarray: The testing dataset.
        """
        return self.data

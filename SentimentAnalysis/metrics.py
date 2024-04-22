# ********************* Code for various metrics for sentiment analysis ***********************
# This code will generate accuracy score, classification report, confusion matrix, precision, recall, and F1 score.


# Import necessary libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Metrics():

    def __init__(self):
        pass

    def accuracy(self, y_true, y_pred):
        """
        Calculate the accuracy of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - accuracy (float): Accuracy of the model.
        """
        return accuracy_score(y_true, y_pred)

    def precision(self, y_true, y_pred):
        """
        Calculate the precision of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - precision (float): Precision of the model.
        """
        return precision_score(y_true, y_pred)

    def recall(self, y_true, y_pred):
        """
        Calculate the recall of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - recall (float): Recall of the model.
        """
        return recall_score(y_true, y_pred)

    def f1(self, y_true, y_pred):
        """
        Calculate the F1 score of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - f1 (float): F1 score of the model.
        """
        return f1_score(y_true, y_pred)
    def accuracy(y_true, y_pred):
        """
        Calculate the accuracy of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - accuracy (float): Accuracy of the model.
        """
        return accuracy_score(y_true, y_pred)


    def precision(y_true, y_pred):
        """
        Calculate the precision of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - precision (float): Precision of the model.
        """
        return precision_score(y_true, y_pred)


    def recall(y_true, y_pred):
        """
        Calculate the recall of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - recall (float): Recall of the model.
        """
        return recall_score(y_true, y_pred)


    def f1(y_true, y_pred):
        """
        Calculate the F1 score of the model.

        Parameters:
        - y_true (array-like): True target values.
        - y_pred (array-like): Predicted target values.

        Returns:
        - f1 (float): F1 score of the model.
        """
        return f1_score(y_true, y_pred)



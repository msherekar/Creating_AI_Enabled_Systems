import numpy as np

import matplotlib.pyplot as plt

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
import os
import pickle


class Fraud_Detector_Model:
            
    def __init__(self, model_dir=None):
        
        self.model = None
        self.models = {}
        
        if model_dir is None:
            self.model_dir = os.path.dirname(__file__)  # Directory of the current Python file
        else:
            self.model_dir = model_dir

    def train(self, train_df=None, train_datafile=None, model_type=None):
        """
        Train the fraud detection model.

        Parameters:
        - train_df (DataFrame): DataFrame containing both input features and target labels for training (optional if train_datafile is provided).
        - train_datafile (str): Path to the CSV file containing training data (optional if train_df is provided).
        - model_type (str): Type of model to train. Options: 'random_forest', 'logistic_regression', 'ensemble'.

        Returns:
        - trained_model: Trained and validated model object.
        """
        if train_datafile is not None:
            try:
                train_dataset = pd.read_csv(train_datafile)
                X_train = train_dataset.drop(columns=['is_fraud'])  # Assuming 'is_fraud' is the name of the target column
                y_train = train_dataset['is_fraud']
            except Exception as e:
                raise ValueError(f"Error reading CSV file: {str(e)}")
        elif train_df is not None:
            # Assuming 'is_fraud' is the name of the target column
            X_train = train_df.drop(columns=['is_fraud'])  
            y_train = train_df['is_fraud']
        else:
            raise ValueError("Either train_df or train_datafile must be provided.")

        if model_type == 'random_forest':
            self.model = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=0.95)), ('predict', RandomForestClassifier(n_estimators=100, random_state=42))])

        elif model_type == 'logistic_regression':
            self.model = Pipeline([('scaler', StandardScaler()),('pca', PCA(n_components=0.95)),("predict", LogisticRegression(class_weight='balanced'))])

        elif model_type == 'ensemble':
            self.model = Pipeline([('scaler', StandardScaler()),('pca', PCA(n_components=0.95)),("predict", StackingClassifier([('lda', LDA()), ('dt', DecisionTreeClassifier(max_depth=7, min_samples_leaf=10, random_state=42))]))])

        else:
            raise ValueError("Invalid model type. Choose from 'random_forest', 'logistic_regression', or 'ensemble'.")

        # Train the model
        self.model.fit(X_train, y_train)

        return self.model
    
    def validate(self, val_df=None, val_datafile=None, model_type=None, num_folds=5):
        """
        Perform cross-validation and return the best model.

        Parameters:
        - X_val (DataFrame): Input features for validation (optional if val_datafile is provided).
        - y_val (Series): Target labels for validation (optional if val_datafile is provided).
        - val_datafile (str): Path to the CSV file containing validation data (optional if X_val and y_val are provided).
        - model_type (str): Type of model to validate. Options: 'random_forest', 'logistic_regression', 'ensemble'.
        - num_folds (int): Number of folds for cross-validation.

        Returns:
        - best_model: Best model based on validation results.
        """
        if val_datafile is not None:
            try:
                val_dataset = pd.read_csv(val_datafile)
                X_val = val_dataset.drop(columns=['is_fraud'])  # Assuming 'is_fraud' is the name of the target column
                y_val = val_dataset['is_fraud']
            except Exception as e:
                raise ValueError(f"Error reading CSV file: {str(e)}")
        elif val_df is not None:
            # Assuming 'is_fraud' is the name of the target column
            X_val = val_df.drop(columns=['is_fraud'])  
            y_val = val_df['is_fraud']
        else:
            raise ValueError("Either val_df or val_datafile must be provided.")

        
        # Perform cross-validation
        scores = cross_validate(self.model, X_val, y_val, scoring="accuracy", cv=num_folds, return_estimator=True)
        best_model_idx = np.argmax(scores['test_score'])
        best_model = scores['estimator'][best_model_idx]

        # Save the best model
        model_name = f"{model_type}_best_model.pkl"
        model_path = os.path.join(self.model_dir, model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        self.models[model_type] = model_path

        return best_model


    def test(self, test_df=None, test_datafile=None, model_type=None):
        """
        Test the fraud detection model.

        Parameters:
        - X_test (DataFrame): Input features for testing (optional if test_datafile is provided).
        - y_test (Series): Target labels for testing (optional if test_datafile is provided).
        - test_datafile (str): Path to the CSV file containing test data (optional if X_test and y_test are provided).
        - model_type (str): Type of model to test. Options: 'random_forest', 'logistic_regression', 'ensemble'.

        Returns:
        - test_accuracy: Accuracy of the model on the test data.
        """
        if test_datafile is not None:
            try:
                test_dataset = pd.read_csv(test_datafile)
                X_test = test_dataset.drop(columns=['is_fraud'])  # Assuming 'is_fraud' is the name of the target column
                y_test = test_dataset['is_fraud']
            except Exception as e:
                raise ValueError(f"Error reading CSV file: {str(e)}")
        elif test_df is not None:
            # Assuming 'is_fraud' is the name of the target column
            X_test = test_df.drop(columns=['is_fraud'])  
            y_test = test_df['is_fraud']
        else:
            raise ValueError("Either test_df or test_datafile must be provided.")
        
        # Load the specified model
        model = self.load_model(model_type)

        # Perform prediction on the test data
        y_pred = model.predict(X_test)

        # Calculate accuracy
        test_accuracy = accuracy_score(y_test, y_pred)

        return y_test, y_pred, test_accuracy

    
    def load_model(self, model_type):
        """
        Load a trained model of a specific type.

        Parameters:
        - model_type (str): Type of model to load.

        Returns:
        - loaded_model: Loaded model object.
        """
        # Construct full path to the model file
        full_model_path = os.path.join(self.model_dir, f"{model_type}_best_model.pkl")

        # Check if the model file exists
        if not os.path.exists(full_model_path):
            raise ValueError(f"No model of type {model_type} found.")

        # Load the model from the specified path
        with open(full_model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        return loaded_model
    
    def infer_model(self, csv_file, model_type):

            # Check if csv_file is None
            if csv_file is None:
                return "Error: No input_transaction provided."

            
            # Read the CSV file
            try:
                transaction_data = pd.read_csv(csv_file)
            except Exception as e:
                return f"Error reading CSV file: {str(e)}"
            
            # Load the specified model
            model = self.load_model(model_type)


            # Perform inference on the transaction data
            prediction = model.predict(transaction_data)

            # Determine the result based on the prediction
            if prediction == 1:
                determination = 'Yes, It is a fraudulent transaction, take immediate action'
            else:
                determination = 'No, It is a legitimate transaction'

            return determination
    
    
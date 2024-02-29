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

    def train(self, X_train, y_train, model_type='random_forest'):
        """
        Train the fraud detection model.

        Parameters:
        - X_train (DataFrame): Input features for training.
        - y_train (Series): Target labels for training.
        - model_type (str): Type of model to train. Options: 'random_forest', 'logistic_regression', 'ensemble'.

        Returns:
        - trained_model: Trained and validated model object.
        """
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

    

    def validate(self, X_val, y_val,model_type, num_folds=5):
        """
        Perform cross-validation and return the best model.

        Parameters:
        - X_val (DataFrame): Input features for validation.
        - y_val (Series): Target labels for validation.
        - num_folds (int): Number of folds for cross-validation.

        Returns:
        - best_model: Best model based on validation results.
        """
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
    

    
    def model_infer(self,df,model_type):

        # Load the specified model
        model = self.load_model(model_type)
        
        # Perform inference on the transaction data
        prediction = model.predict(df)

        # Determine the result based on the prediction
        if prediction == 1:
            determination = 'Yes, It is a fraudulent transaction, take immediate action'
        else:
            determination = 'No, It is a legitimate transaction'

        return determination

    
    def model_infer2(self, csv_file, model_type):
        # Check if csv_file is None
        if csv_file is None:
            return "Error: No input_transaction provided."

        # Load the specified model
        model = self.load_model(model_type)

        # Read the CSV file
        try:
            transaction_data = pd.read_csv(csv_file)
        except Exception as e:
            return f"Error reading CSV file: {str(e)}"

        # Perform inference on the transaction data
        prediction = model.predict(transaction_data)

        # Determine the result based on the prediction
        if prediction == 1:
            determination = 'Yes, It is a fraudulent transaction, take immediate action'
        else:
            determination = 'No, It is a legitimate transaction'

        return determination






































        

        


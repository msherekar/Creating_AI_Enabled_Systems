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




class ETL_Pipeline:

    @staticmethod
    def extract(filename: str) -> pd.DataFrame:
        """
        Extract data from a CSV file.
        
        Args:
            filename (str): The path to the CSV file.
            chunk_size (int, optional): The number of rows to read in each iteration. 
                If provided, data will be read in chunks to handle large files. 
                Defaults to None.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted data.
        """
        # Check if the file is a CSV file.
        if not filename.endswith('.csv'):
            raise ValueError("File is not a CSV")

        # Read the CSV file into a pandas DataFrame.

        dataframe = pd.read_csv(filename)

        return dataframe

 

    @staticmethod
    def categorize_time(transactions_df: pd.DataFrame, night_start_hour: int = 22, night_end_hour: int = 4) -> pd.DataFrame:
        """
        Categorize time-related features in a DataFrame of transactions.

        Args:
            transactions_df (pd.DataFrame): DataFrame containing transaction data.
            night_start_hour (int, optional): The start hour of night time. Defaults to 22.
            night_end_hour (int, optional): The end hour of night time. Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame with time-related features added.
        """
        # Convert Unix timestamp to datetime object
        transactions_df['datetime'] = pd.to_datetime(transactions_df['unix_time'], unit='s')

        # Extract hour component from datetime
        transactions_df['hour'] = transactions_df['datetime'].dt.hour

        # Create a new binary feature indicating whether a transaction occurred between specified night hours
        transactions_df['is_night'] = ((transactions_df['hour'] >= night_start_hour) | 
                                        (transactions_df['hour'] < night_end_hour)).astype(int)

        return transactions_df.copy()  # Return a copy of the DataFrame to avoid modifying the original


    @staticmethod
    def categorize_year(transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize transactions based on different periods within a year.

        Args:
            transactions_df (pd.DataFrame): DataFrame containing transaction data with 'unix_time' column.

        Returns:
            pd.DataFrame: DataFrame with categorical feature 'time_period' indicating the time period for each transaction.
        """
        # Convert Unix timestamp to datetime object
        transactions_df['datetime'] = pd.to_datetime(transactions_df['unix_time'], unit='s')

        # Define boolean masks for each period
        holidays_mask = ((transactions_df['datetime'].dt.month == 12) & (transactions_df['datetime'].dt.day >= 24)) | \
                        ((transactions_df['datetime'].dt.month == 1) & (transactions_df['datetime'].dt.day <= 1))
        post_holidays_mask = (transactions_df['datetime'].dt.month == 1) | (transactions_df['datetime'].dt.month == 2)
        summer_mask = (transactions_df['datetime'].dt.month >= 5) & (transactions_df['datetime'].dt.month <= 9)

        # Create one-hot encoded features for each period
        transactions_df['is_holidays'] = holidays_mask.astype(int)
        transactions_df['is_post_holidays'] = post_holidays_mask.astype(int)
        transactions_df['is_summer'] = summer_mask.astype(int)

        return transactions_df.copy()  # Return a copy of the DataFrame to avoid modifying the original


    @staticmethod
    def remove_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove unnecessary columns from the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: DataFrame with only important columns.
        """
        imp_columns = ['amt', 'cc_num', 'is_night', 'is_holidays', 'is_post_holidays', 'is_summer', 'is_fraud']
        df_selected = df.loc[:, imp_columns]
        return df_selected

    @staticmethod
    def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data.

        Args:
            dataframe (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        processed_dataframe = ETL_Pipeline.categorize_time(dataframe)
        processed_dataframe = ETL_Pipeline.categorize_year(processed_dataframe)
        processed_dataframe = ETL_Pipeline.remove_columns(processed_dataframe)
        return processed_dataframe
    
    @staticmethod
    def preprocess_new(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data.

        Args:
            dataframe (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        processed_dataframe = ETL_Pipeline.categorize_time(dataframe)
        processed_dataframe = ETL_Pipeline.categorize_year(processed_dataframe)
        processed_dataframe = ETL_Pipeline.remove_columns_new(processed_dataframe)
        return processed_dataframe

    @staticmethod
    def transform(csv_file: str) -> str:
        """
        Transform raw transaction data.

        Args:
            csv_file (str): Path to the CSV file containing raw transaction data.

        Returns:
            str: Filename of the transformed CSV file.
        """
        etl_pipeline = ETL_Pipeline()

        # Read the given .csv file into a DataFrame
        transactions_df = etl_pipeline.extract(csv_file)

        # Apply preprocessing steps
        transactions_df = etl_pipeline.preprocess(transactions_df)

        # Save the transformed DataFrame to a CSV file
        transformed_filename = etl_pipeline.load(transactions_df)

        return transformed_filename
    
    @staticmethod
    def transform_new(csv_file: str) -> str:
        """
        Transform raw transaction data.

        Args:
            csv_file (str): Path to the CSV file containing raw transaction data.

        Returns:
            str: Filename of the transformed CSV file.
        """
        etl_pipeline = ETL_Pipeline()

        # Read the given .csv file into a DataFrame
        transactions_df = etl_pipeline.extract(csv_file)

        # Apply preprocessing steps
        transactions_df = etl_pipeline.preprocess_new(transactions_df)

        # Save the transformed DataFrame to a CSV file
        transformed_filename = etl_pipeline.load(transactions_df)

        return transformed_filename

    
    @staticmethod
    def load(transformed_dataframe: pd.DataFrame) -> str:
        """
        Export the transformed DataFrame to a CSV file.

        Args:
            transformed_dataframe (pd.DataFrame): Transformed DataFrame.

        Returns:
            str: Filename of the exported CSV file.
        """
        transformed_filename = 'transformed_data.csv'
        transformed_dataframe.to_csv(transformed_filename, index=False)
        return transformed_filename


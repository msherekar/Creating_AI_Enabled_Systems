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


class ETL_Pipeline:



    @staticmethod
    def extract(filename: str) -> pd.DataFrame:
        """
        Reads a CSV file into a pandas DataFrame.

        Args:
            filename: The path to the CSV file.

        Returns:
            A pandas DataFrame containing the data from the CSV file.

        Raises:
            ValueError: If the file is not a CSV file.
        """

        # Check if the file is a CSV file.
        if not filename.endswith('.csv'):
            raise ValueError("File is not a CSV")

        # Read the CSV file into a pandas DataFrame.
        dataframe = pd.read_csv(filename, index_col=0)

        return dataframe

    @staticmethod
    def extract_json(filename: str) -> pd.DataFrame:
        """
        Reads a JSON file into a pandas DataFrame.

        Args:
            filename: The path to the JSON file.

        Returns:
            A pandas DataFrame containing the data from the JSON file.
        """

        # Read the JSON file into a pandas DataFrame.
        dataframe = pd.read_json(filename, orient='records')

        return dataframe

        

    @staticmethod

    def categorize_time(transactions_df):
      # Convert Unix timestamp to datetime object
      transactions_df['datetime'] = pd.to_datetime(transactions_df['unix_time'], unit='s')

      # Extract hour component from datetime
      transactions_df['hour'] = transactions_df['datetime'].dt.hour

      # Create a new binary feature indicating whether a transaction occurred between 10 PM and 4 AM
      transactions_df['is_night'] = ((transactions_df['hour'] >= 22) | (transactions_df['hour'] < 4)).astype(int)

      return transactions_df

    @staticmethod

    def categorize_year(transactions_df):
      # Convert Unix timestamp to datetime object
      transactions_df['datetime'] = pd.to_datetime(transactions_df['unix_time'], unit='s')

      # Define boolean masks for each period
      holidays_mask = ((transactions_df['datetime'].dt.month == 12) & (transactions_df['datetime'].dt.day >= 30)) | \
                      ((transactions_df['datetime'].dt.month == 1) & (transactions_df['datetime'].dt.day <= 31))
      post_holidays_mask = (transactions_df['datetime'].dt.month == 1) | (transactions_df['datetime'].dt.month == 2)
      summer_mask = (transactions_df['datetime'].dt.month >= 5) & (transactions_df['datetime'].dt.month <= 9) & \
                    ((transactions_df['datetime'].dt.month == 5) & (transactions_df['datetime'].dt.day >= 24) |
                    (transactions_df['datetime'].dt.month == 9) & (transactions_df['datetime'].dt.day <= 1))

      # Create a new categorical feature indicating the time period for each transaction
      transactions_df['time_period'] = 'rest_of_year'  # Initialize with 'rest_of_year'
      transactions_df.loc[holidays_mask, 'time_period'] = 'holidays'
      transactions_df.loc[post_holidays_mask, 'time_period'] = 'post_holidays'
      transactions_df.loc[summer_mask, 'time_period'] = 'summer'

      return transactions_df




    @staticmethod
    
    def calculate_avg_transactions(transactions):
      # Step 1: Calculate age

      transactions['dob'] = pd.to_datetime(transactions['dob'])  # Convert dob column to datetime
      transactions['age'] = pd.Timestamp.now().year - transactions['dob'].dt.year  # Calculate age in years

      # Step 2: Define age groups
      age_bins = [0, 14, 18, 23, 30, 40, 50, 60, 70, 80, 90, 100]  # Define age bins
      age_labels = ['0-14','14-18', '18-23', '23-30', '31-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']  # Labels for age groups
      transactions['age_group'] = pd.cut(transactions['age'], bins=age_bins, labels=age_labels, right=False)

      # Step 3: Count the number of frauds for each age group
      frauds_by_age_group = transactions[transactions['is_fraud'] == 1].groupby('age_group').size()

      # Filter to include only fraudulent transactions
      fraud = transactions[transactions['is_fraud'] == 1]

      # Group by victim and date of transaction, count the number of transactions per day
      transactions_per_day = fraud.groupby(['cc_num', fraud['datetime'].dt.date]).size()

      # Calculate the average number of transactions per day for each victim
      average_transactions_per_day = transactions_per_day.groupby('cc_num').mean()


      # Convert trans_date_trans_time to datetime
      transactions['trans_date_trans_time'] = pd.to_datetime(transactions['trans_date_trans_time'])

      # Step 1: Group by dob; each dob signifies each unique victim
      grouped_by_unique_victims = transactions.groupby('cc_num')

      # Step 2: Calculate the number of transactions for each victim
      num_transactions_per_victim = grouped_by_unique_victims['trans_num'].count()

      # Step 3: Calculate the time span (in days) between the earliest and latest transactions for each victim
      time_span_per_victim = grouped_by_unique_victims['trans_date_trans_time'].apply(lambda x: (x.max() - x.min()) / pd.Timedelta(days=1))

      # Step 4: Calculate the average number of transactions per day for each victim
      average_transactions_per_day = num_transactions_per_victim / time_span_per_victim

      ## Step 5: Add the new column 'average_transactions_per_day' to the original DataFrame
      transactions['average_transactions_per_day'] = grouped_by_unique_victims['trans_num'].transform(lambda x: x.count()) /       grouped_by_unique_victims['trans_date_trans_time'].transform(lambda x: (x.max() - x.min()) / pd.Timedelta(days=1))

      # Binning strategy
      bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

      # Assign bin values based on 'average_transactions_per_day'
      transactions['bin'] = pd.cut(transactions['average_transactions_per_day'], bins=bins, labels=range(1, len(bins)))

      average_transactions = pd.DataFrame({'average_transactions_per_day': average_transactions_per_day})

      # Bin the average transactions per day
      average_transactions['bin'] = pd.cut(average_transactions['average_transactions_per_day'], bins=bins)
        
      # Count how many average transactions fall into each bin
      transactions_per_bin = average_transactions['bin'].value_counts().sort_index()

      average_transactions_per_day = transactions_per_day.groupby('cc_num').mean()

      # Define a condition for low average transactions per day
      low_avg_condition = (transactions['average_transactions_per_day'] >= 1) & (transactions['average_transactions_per_day'] <= 6)

      # Create a new feature 'Low_avg_transactions_per_day'
      transactions['Low_avg_transactions_per_day'] = 0  # Initialize with 0
      transactions.loc[low_avg_condition, 'Low_avg_transactions_per_day'] = 1  # Set to 1 if the condition is met

      return transactions

    # 3. PREPARE
    # Remove unneccessary columns

    @staticmethod
    def remove_columns(df):

      imp_columns = ['amt', 'cc_num', 'Low_avg_transactions_per_day', 'average_transactions_per_day', 'is_night','is_fraud']
      #columns_drop = ['merchant', 'category','trans_date_trans_time','first', 'last', 'sex', 'street', 'city', 'state', 'zip', 'lat', 'long','city_pop',    #'job','trans_num', 'merch_lat','merch_long']
      #maybe = ['bin','hour','unix_time']

      df_selected = df.loc[:, imp_columns]

      return df_selected



    @staticmethod
    def load(transformed_dataframe):

      """ This method performs preprocessing &
          Export to a .csv"""
      transformed_filename = 'transformed_data.csv'  # provide the filename here

      transformed_dataframe.to_csv(transformed_filename, index=False)

      return transformed_filename

    

    @staticmethod
    def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses the data.

        Args:
            dataframe: The input pandas DataFrame.

        Returns:
            A pandas DataFrame containing the preprocessed data.
        """

        # Apply preprocessing steps here
        processed_dataframe = ETL_Pipeline.categorize_time(dataframe)
        processed_dataframe = ETL_Pipeline.categorize_year(processed_dataframe)
        processed_dataframe = ETL_Pipeline.calculate_avg_transactions(processed_dataframe)
        processed_dataframe = ETL_Pipeline.remove_columns(processed_dataframe)

        return processed_dataframe
    
    @staticmethod
    def transform(csv_file):
        
        '''This function can only be used for raw transactions.
        It is just a demo function to process raw transactions file
        for demonstrating system is working'''
        
        etl_pipeline = ETL_Pipeline()

        # read the given .csv file into a df
        transactions_df = etl_pipeline.extract(csv_file)

        # add features to signify time(hour) of transactions
        transactions_df = etl_pipeline.categorize_time(transactions_df)

        # add features to signify part of year for transactions
        transactions_df = etl_pipeline.categorize_year(transactions_df)

        # add average transactions as a feature
        transactions_df = etl_pipeline.calculate_avg_transactions(transactions_df)

        # finally, remove all the unnecessary columns
        transactions_df = etl_pipeline.remove_columns_demo(transactions_df)
        


        # save as a transformed csv file
        transactions_df = etl_pipeline.load(transactions_df)

        return transactions_df
    
    @staticmethod
    def remove_columns_demo(df):
        '''This function can only be used for raw transactions.
        It is just a demo function to process raw transactions file
        for demonstrating system is working'''

        imp_columns = ['amt', 'cc_num', 'Low_avg_transactions_per_day', 'average_transactions_per_day', 'is_night']
        df_selected = df.loc[:, imp_columns]
        return df_selected







    

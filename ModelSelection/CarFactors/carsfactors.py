from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import joblib
import pickle

class carsfactors:
    def __init__(self):
        self.modelLearn = False
        self.stats = 0

    
    def model_learn(self):
        # Importing the dataset into a pandas dataframe
        cars = pd.read_csv('cars.csv')  # Change this to your file path

        # Define useful and unused columns
        useful_columns = ['transmission', 'color', 'odometer_value', 'year_produced', 'price_usd', 'duration_listed'] # change this code
        unused_cols = [col for col in cars.columns if col not in useful_columns]

        # Remove unwanted columns
        cars = cars.drop(columns=unused_cols)

        # One hot enconding for transmission and color
        one_hot_transmission = pd.get_dummies(cars['transmission'], prefix='transmission')
        one_hot_color = pd.get_dummies(cars['color'], prefix='color')

        # Concatenate one-hot encoded columns with the original DataFrame
        cars_encoded = pd.concat([cars, one_hot_transmission, one_hot_color], axis=1)

        # Drop the original 'transmission' and 'color' columns
        cars_encoded.drop(['transmission', 'color'], axis=1, inplace=True)


        
        
        # Saving the encoders
        with open('transmission_encoder.pkl', 'wb') as f:
            pickle.dump(one_hot_transmission, f)

        with open('color_encoder.pkl', 'wb') as f:
            pickle.dump(one_hot_color, f)

        # Feature Scaling
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cars_encoded.drop('duration_listed', axis=1))  # Excluding the target variable

        # Save the scaler for future use
        joblib.dump(scaler, 'feature_scaler.pkl')

        # Concatenate the scaled features with the target variable
        scaled_cars = pd.concat([pd.DataFrame(scaled_features, columns=cars_encoded.drop('duration_listed', axis=1).columns), cars_encoded['duration_listed']], axis=1)

        # Spliltting into X and y values
        # Splitting into training and testing
        X = scaled_cars.drop('duration_listed', axis=1)  # Features
        y = scaled_cars['duration_listed']  # Target variable

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Select useful model to deal with regression (it is not categorical for the number of days can vary quite a bit)
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        self.stats = self.model.score(X_test, y_test)
        self.modelLearn = True

    
    def model_infer(self, transmission, color, odometer_value, year_produced, price_usd):
      if not self.modelLearn:
          self.model_learn()

      # Load the encoders and scaler
      # Loading the encoders
      with open('transmission_encoder.pkl', 'rb') as f:
          loaded_transmission_encoder = pickle.load(f)

      with open('color_encoder.pkl', 'rb') as f:
          loaded_color_encoder = pickle.load(f)
      
      scaler = joblib.load('feature_scaler.pkl')

      # Convert the transmission and color into one-hot encoded format
      transmission_encoded = pd.get_dummies(pd.Series(transmission), prefix='transmission')
      color_encoded = pd.get_dummies(pd.Series(color), prefix='color')

      # Align one-hot encoded columns with the original encoding
      transmission_encoded = transmission_encoded.reindex(columns=loaded_transmission_encoder.columns, fill_value=0)
      color_encoded = color_encoded.reindex(columns=loaded_color_encoder.columns, fill_value=0)

      # Prepare other columns
      other_columns = np.array([[odometer_value, year_produced, price_usd]])

      # Concatenate all features
      total = np.concatenate((transmission_encoded.values, color_encoded.values, other_columns), axis=1)

      # Scale the features
      scaled_total = scaler.transform(total)

      # Predict the duration listed
      y_pred = self.model.predict(scaled_total)

      return str(y_pred[0])  # Return the prediction



    def model_stats(self):
        if not self.modelLearn:
            self.model_learn()
        return str(self.stats)


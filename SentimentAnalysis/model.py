# ******************* Code for models for sentiment analysis *******************

# Import necessary libraries
import joblib
from data_pipeline import Pipeline

class Model():
    def __init__(self):
        self.model = joblib.load('rf_basic.pkl')


    def predict_sentiment(self, embeddings):
        prediction = self.model.predict(embeddings)
        return prediction


if __name__ == '__main__':
    # Test the model
    model = Model()
    pipeline = Pipeline('data.csv')
    processed_data = pipeline.new_X
    print(model.predict_sentiment(processed_data))

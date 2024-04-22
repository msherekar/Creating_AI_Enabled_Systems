# *********************** Code for establishing the data pipeline ************************

# This code demonstrates how to establish a data pipeline for processing text for sentiment analysis.
# The pipeline consists of the following steps: 1) loading the data, 2) preprocessing the data, 3) normalizing the data,
# 4) tokenizing the data, 5) embedding the data

# Import necessary libraries
import pandas as pd
import numpy as np
from preprocess import normalization
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import joblib


class Pipeline:
    def __init__(self, csv_file_path):
        # Load data from the provided CSV file
        self.new_data = pd.read_csv(csv_file_path)
        # Assuming the text column is named 'text'
        self.new_reviews = self.new_data[['text']]  # the name of review text column can be changed

        # Assuming normalization() and word_tokenize() are defined elsewhere
        self.new_reviews['normalized_text'] = self.new_reviews['text'].apply(lambda x: normalization(x))
        self.new_reviews['tokens'] = self.new_reviews['normalized_text'].apply(lambda x: word_tokenize(x))

        # Initialize max_length and model_wv
        max_length = 100  # Define your desired max_length
        model_wv = Word2Vec.load("model_wv.bin")

        # Generate embeddings for the new data
        self.new_embeddings = [model_wv.wv[tokens_list] for tokens_list in self.new_reviews['tokens']]

        # Apply padding or truncation and flatten the embeddings
        self.new_X = []
        for emb in self.new_embeddings:
            if len(emb) >= max_length:
                emb = emb[:max_length]
            else:
                emb = np.pad(emb, ((0, max_length - len(emb)), (0, 0)), mode='constant')
            self.new_X.append(emb.flatten())
        self.new_X = np.array(self.new_X)


# Example usage:
if __name__ == '__main__':
    # initialize the pipeline
    test_csv_file_path = 'data.csv'
    pipeline = Pipeline(test_csv_file_path)
    processed_data = pipeline.new_X
    print(processed_data)

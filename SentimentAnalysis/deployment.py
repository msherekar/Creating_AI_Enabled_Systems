# *********** This code is for deploying the model and making predictions ***********
# This code will accept new text review(as .csv file) via post return the sentiment of the review.

from flask import Flask, request
import os
from data_pipeline import Pipeline
from model import Model

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_post():
    # Get the text review from the request
    text = request.files.get('text', '')
    model = Model()
    pipeline = Pipeline(text)
    processed_data = pipeline.new_X
    prediction = model.predict_sentiment(processed_data)
    return str(prediction)



if __name__ == "__main__":
    flaskPort = 8786
    print('starting server...')
    app.run(host='0.0.0.0', port=flaskPort, debug=True)

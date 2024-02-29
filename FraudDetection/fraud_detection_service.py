from flask import Flask, request
import os
import pandas as pd
from model import Fraud_Detector_Model

fdm = Fraud_Detector_Model()
app = Flask(__name__)

# paste this link in the browser to run
#http://localhost:8786/infer?input_transaction=input_transaction.csv

@app.route('/', methods=['GET'])
def getInfer():
    # Retrieve query parameters from the request URL
    args = request.args
    
    # Get the value of the 'input_transaction' parameter
    csv_file = args.get('input_transaction')
    
    # Assuming `random_forest` is the model type (e.g., "random_forest")
    model_type = "random_forest"

    # Call the model_infer2 method to perform inference
    # pass a processed csv file or df as argument
    
    # this version of model_infer takes in a csv file
    determination = fdm.model_infer2(csv_file, model_type)
    
    # Return the determination result as JSON
    return determination


def hellopost():    
    
    # set up the request; this datafile variable should be the name of key in postmaster
    datafile = request.files.get('datafile', '')
    
    print("Data: ", datafile.filename)
    
    datafile.save('data.csv')
    
    return 'File Received - Thank you'



if __name__ == "__main__":
    
    fdm = Fraud_Detector_Model()
    flaskPort = 8786
    print('starting server...')
    app.run(host='0.0.0.0', port=flaskPort, debug=True)
    
#datafile.save('Users/mukulsherekar/Documents/CRAISYS/workspace/705.603Portfolio/FraudDetection/datafile.csv')
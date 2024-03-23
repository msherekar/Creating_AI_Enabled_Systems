from flask import Flask, request
import os
import pandas as pd
from model import Fraud_Detector_Model


fdm = Fraud_Detector_Model(model_dir=None)
app = Flask(__name__)


@app.route('/detect', methods=['POST'])
def hellopost():    
    
    # set up the request; this datafile variable should be the name of key in postmaster
    input_transaction = request.files.get('input_transaction', '')        
    return (fdm.infer_model(input_transaction, 'logistic_regression'))
    
@app.route('/infer', methods=['GET'])
def getInfer():
    
    # Retrieve query parameters from the request URL
    args = request.args
    
    # Get the value of the 'input_transaction' parameter
    csv_file = args.get('check_transaction')
    return (fdm.infer_model(csv_file, 'logistic_regression'))
    

if __name__ == "__main__":
    
    flaskPort = 8786
    print('starting server...')
    app.run(host='0.0.0.0', port=flaskPort, debug=True)
    

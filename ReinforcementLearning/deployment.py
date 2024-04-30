from flask import Flask, request, jsonify
import numpy as np
import pickle
from preprocess import preprocess_data
from metrics import Metrics

app = Flask(__name__)

#Load the trained Q-table
with open('q_table.pkl', 'rb') as f:
    Q_table = pickle.load(f)


@app.route('/suggest_subject_lines', methods=['POST'])
def suggest_subject_lines():
    file = request.files['new_state']  # Get uploaded .csv file

    # Preprocess the .csv file and obtain states to be fed into the Q-table
    states = preprocess_data(file)

    suggested_subject_lines = {}

    for new_state in states:
        # Map state to index and obtain suggested action
        new_state_index = hash(new_state) % Q_table.shape[0]  # Assuming Q-table is pre-loaded
        q_values = Q_table[new_state_index]
        suggested_action_index = np.argmax(q_values)

        # Define subject lines corresponding to actions
        subject_lines = ['Subject line 1', 'Subject line 2', 'Subject line 3']
        suggested_subject_line = subject_lines[suggested_action_index]

        # Store suggested subject line for this new state
        suggested_subject_lines[str(new_state)] = suggested_subject_line

    return jsonify(suggested_subject_lines)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

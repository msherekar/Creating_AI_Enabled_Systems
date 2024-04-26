from flask import Flask, request, jsonify
import numpy as np
import pickle
from preprocess import preprocess_data

app = Flask(__name__)

# Load the trained Q-table
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



# @app.route('/suggest_subject_lines', methods=['POST'])
# def suggest_subject_lines():
#     file = request.files['new_state']  # Get uploaded JSON file
#     data = file.read().decode('utf-8')  # Read file contents as bytes and decode to string
#     json_data = json.loads(data)  # Parse JSON data from string
#
#     suggested_subject_lines = {}
#
#     for key, new_state in json_data.items():
#         # Map state to index and obtain suggested action
#         new_state_index = hash(tuple(new_state)) % Q_table.shape[0]  # Assuming Q-table is pre-loaded
#         q_values = Q_table[new_state_index]
#         suggested_action_index = np.argmax(q_values)
#
#         # Define subject lines corresponding to actions
#         subject_lines = ['Subject line 1', 'Subject line 2', 'Subject line 3']
#         suggested_subject_line = subject_lines[suggested_action_index]
#
#         # Store suggested subject line for this new state
#         suggested_subject_lines[key] = suggested_subject_line
#
#     return jsonify(suggested_subject_lines)



# @app.route('/suggest_subject_line', methods=['POST'])
# def suggest_subject_line():
#
#     new_state = request.files.get('new_state', '')  # Get new state from request
#     new_state = tuple(new_state)  # Convert to tuple if necessary
#
#     # Map state to index and obtain suggested action
#     new_state_index = hash(new_state) % Q_table.shape[0]  # Assuming Q-table is pre-loaded
#     q_values = Q_table[new_state_index]
#     suggested_action_index = np.argmax(q_values)
#
#     # Define subject lines corresponding to actions
#     subject_lines = ['Subject line 1', 'Subject line 2', 'Subject line 3']
#     suggested_subject_line = subject_lines[suggested_action_index]
#
#     return jsonify({'suggested_subject_line': suggested_subject_line})




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

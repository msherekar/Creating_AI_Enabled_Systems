import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime


def save_report(report_file, report_details, training_metrics):
    """
    Save a training report with the given details and metrics.

    Parameters:
        report_file (str): The file path for the report.
        report_details (dict): Details to be included in the report.
        training_metrics (dict): Training metrics to be included in the report.
        discounted_rewards (list): List of discounted rewards.
        episodes_to_convergence (int): Number of episodes to convergence, if applicable.
    """
    # Write the report content to the file
    with open(report_file, "w") as file:
        # Write the report heading
        #file.write(report_details["Heading"] + "\n\n")

        # Write details
        file.write("Details:\n")
        for key, value in report_details.items():
            if isinstance(value, list):
                file.write("{}: {}\n".format(key, ", ".join(value)))
            else:
                file.write("{}: {}\n".format(key, value))
        file.write("\n")

        # Write training metrics
        file.write("Training Metrics:\n")
        for key, value in training_metrics.items():
            file.write("{}: {}\n".format(key, value))
        file.write("\n")



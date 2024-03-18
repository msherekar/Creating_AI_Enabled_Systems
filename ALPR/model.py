from metrics import Metrics

class Object_Detection_Model:
    def __init__(self, model):
        """
        Initialize the Object_Detection_Model class with the provided object detection model.

        Args:
            model: The object detection model (e.g., YOLO, SSD, etc.).
        """
        self.model = model
        self.metrics = Metrics()

    def predict(self, input_data):
        """
        Perform inference on the input data and return predictions.

        Args:
            input_data: Raw input data for object detection.

        Returns:
            predictions: Predicted bounding boxes or masks.
        """
        predictions = self.model.predict(input_data)
        return predictions

    def test(self, y_prediction, y_label):
        """
        Call the Metrics class to generate a report.

        Args:
            y_prediction: Predicted bounding boxes or masks.
            y_label: Ground truth bounding boxes or masks.
        """
        self.metrics.generate_report(y_prediction, y_label, "report.txt")

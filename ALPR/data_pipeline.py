#   ************************************** DATA ENGINEERING ***********************************************

import numpy as np
import cv2 as cv
import ffmpeg
import subprocess
import os
import json
import pytesseract
from PIL import Image


class Pipeline():
    
    @staticmethod
    def stream_to_frame(input_url, width, height):        
        process1 = (
            ffmpeg
            .input(input_url)
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        while True:
            in_bytes = process1.stdout.read(width * height * 3)
            if not in_bytes:
                break
            in_frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            yield in_frame  # Yield the frame instead of storing it in a list
    
 

    @staticmethod
    def license_plate(frame):
        """
        plate refer to license plate
        Argument -> image in form of a numpy array
        Returns -> image(s) with license plates detected in them
        Note: license plates detected with a confidence >0.9

        """

        # load yolo
        net = cv.dnn.readNet("lpr-yolov3-tiny.weights", "lpr-yolov3-tiny.cfg")
        #net = cv.dnn.readNet("lpr-yolov3.weights", "lpr-yolov3.cfg")

        ln = net.getLayerNames()

        classes = []
        with open("coco.names", 'r') as f: # changes to coco.names?
            classes = [line.strip() for line in f.readlines()]

        layer_name = net.getLayerNames()
        output_layer = [layer_name[i - 1] for i in net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        height, width, channel = frame.shape

        # Define parameters
        scale_factor = 1/255.0  # Scale factor to normalize pixel values to [0, 1]
        size = (416, 416)       # Standard size for YOLO input
        mean = (0, 0, 0)        # Mean subtraction (zero mean)
        swapRB = True           # Swap Red and Blue channels

        # Create blob from image
        blob = cv.dnn.blobFromImage(frame, scale_factor, size, mean, swapRB)

        # Detect objects
        net.setInput(blob)
        outs = net.forward(output_layer)

        # Showing Information on the screen
        class_ids = [] # can we d=chage this to just licence plates; food for thought
        confidences = []
        boxes = []
        detection_found = False
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.9: # Threshold
                    detection_found = True
                    # Object detection
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width*1.5) # changes to increase area of bb
                    h = int(detection[3] * height*1.5)# changes to increase area of bb
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS)
        indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.1, 0.2) 

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                #return (x,y,w,h),label,confidences[i]
                license = frame[y:y+h, x:x+w] # image of cropped license plate
                return license



    @staticmethod
    def read_cropped_image(cropped_image):
        """
        Reads text from a cropped image using Tesseract OCR.
        """
        #pytesseract.pytesseract.tesseract_cmd = r'/opt/local/bin/tesseract'
        
        # Convert the cropped image array to PIL Image format
        pil_image = Image.fromarray(cropped_image)

        # Perform OCR on the cropped image
        read_text = pytesseract.image_to_string(pil_image, lang='eng')

        # Check if pytesseract fails to read or if nothing is returned
        if not read_text:
            return "Nothing Detected"

        # Strip any trailing whitespace or special characters
        read_text = read_text.strip()

        # Remove trailing '|' if present
        if read_text.endswith('|'):
            read_text = read_text[:-1]

        # Check if the plate number contains only A-Z and 0-9 characters
        if not read_text.isalnum():
            return "Invalid Characters"

        # Check if the length of the plate number is within the expected range
        if len(read_text) != 7:
            return "Invalid Length"

        # Additional conditions can be added here based on specific state rules or formats

        return read_text

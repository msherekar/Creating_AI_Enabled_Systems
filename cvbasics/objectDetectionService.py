from flask import Flask
from flask import request
import os

from GraphicDataProcessing import ObjectDetection
#image_path = "Pictures/bermuda.jpg"
#reader = ObjectDetection(image_path)

app = Flask(__name__)

# Use postman to generate the post with a graphic of your choice

@app.route('/detect', methods=['POST'])
def detection():
    args = request.args
    
    #name = args.get('name')
    #location = args.get('description') # two argument coming in with post request
    
    imagefile = request.files.get('imagefile', '')
    print("Image: ", imagefile.filename)
    
    
    
    #imagefile.save('LOCAL DIRECTORY')
    
    # The file is now downloaded and available to use with your detection class
    #imagepath = 'LOCAL DIRECTORY' ??
    
    
    ot = ObjectDetection(imagepath)
    findings = ot.detect()
    
    # covert to useful string
    findingsString = findings[0]
    return findingsString

if __name__ == "__main__":
    flaskPort = 8786
    ot = ObjectDetection()
    print('starting server...')
    app.run(host = '0.0.0.0', port = flaskPort)


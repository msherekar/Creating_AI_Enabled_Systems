# AUTOMATED LICENSE PLATE RECOGNITION SYSTEM
## SECOND CLASS PROJECT FOR CREATING AI ENABLED SYSTEMS (EN.705.603.81)
 

**Problem Statement**: To create a basic object recognition model for license plates. 

**Value Proposition**: Am automated system will improve traffic flow, reduce congestion, and eliminate the need for manual toll collection.

# CONTENTS OF THIS REPOSITORY

## Readme

## Systems Planning & Requirements

## Jupyter Notebooks
- ** Exploratory Data Analysis**
- ** Model Performance Analysis**


## Python Files
- **data_pipeline.py** : For data engineering
- **dataset.py**: For partitioning data 
- **metrics.py**: For calculating various metrics
- **model.py** : For model training
- **deployment.py**: The systems level file that executes the whole code

## Model Files
- yolov3-tiny weights and configuration file
- coco.names file
- coco annotations json file

## Text Files
- Sample output text file

## Miscellaneous Files
- Various .csv files for testing
- original data file (transactions.csv)

## [Docker Image](docker pull msherekar/705.603spring24:ALPR)

# INSTUCTIONS TO RUN
- Download the container and execute these two commands on two separate command lines:
- ffmpeg -i LicensePlateReaderSample_4K.mov -vcodec mpeg4 -f mpegts udp://127.0.0.1:23002
- docker run -it -p 23002:23002/udp msherekar/705.603spring24:ALPR
- Note the port to be used is 23002



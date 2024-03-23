# **Motivation (WHY's)**

## **Why are we solving this problem?**

**Problem Statement**: Tax collection at toll booths on highways needs to be modernized.

 **Value Proposition**:  A modern machine learning based system will improve traffic flow, reduce congestion, and eliminate the need for manual toll collection

 **Why is our solution a viable one**?

 A ML based solution is viable because it will accurately and efficiently recognize license plates from various states at high speeds, day and night, under different weather conditions.

We can tolerate some degree of risk in the sense that there will be an auditor in the loop. Even if there are some mis-recognitions by the model, human in loop can fix it. Also, by presence of human, model will be improving all the time. Mis-classified

This solution is a feasible one because we have the data and compute power. Also, there is enough good quality data and powerful trained models ready to be deployed.

# **Requirements (WHAT's)**

## **Scope**

### What are our goals?
- **Organizational Goals:** Modernize the toll collection
- **System Goals:** Capture images of moving vehicles at toll booths and automatically charge tolls to vehicles' registered owners by accurately recognizing and recording the license plates. 
- **User Goals:** Help auditors improve the model by capturing mis labels
- **Customer:** Ensure smooth experience for drivers at the toll booths with out any traffic congestion and payment hassles.
- **Company:** Improve traffic, increase toll collection, develop an accurate model 
- **Model Goals:** Maintain high accuracy to ensure correct billing and customer satisfaction.

### What are the success criteria?
Our success criteria is to have more than 95% accuracy

## **Requirements**

### **What are our(system) Assumptions?**
We assume we have the compute power (hardware), cyber security set up (hardware and software), data storage capabilities, availability of state of the art ML softwares, visualization products, project management solutions and talented hardworking man power.

### **What are our (system) Requirements?**
First and foremost - clean data ready to be fed into models. Secondly, ability to create and analyze new models. Thirdly, ability to quickly fix any mistakes made by model. Finally, ability to change or update model with new incoming data.

#### **identify Stakeholders**: Regulatory Bodies, Financial Institutions, Customers, AI systems designers, Legal & Compliance Teams

#### **Collect Information (data)**
- **License Number Plate Data:** Collect data in various environmental conditions, of obscured number plates and variety of cars.

#### **Negotiate Conflicts** 
- **Risk Tolerance vs. False Positives:** Balancing the need to detect fraud with the risk of flagging legitimate transactions as false positives.
- **Resource Allocation:** Address conflicts regarding budget allocation and resource prioritization for AI model development and deployment.
- **Interpretation of Results:** Resolve disagreements on the interpretation of AI model outputs and the appropriate action to take in response to flagged transactions.
- **Ethical Considerations:** Navigate ethical dilemmas related to the use of AI in license plate detection by making sure faces of drivers are blurred.

#### **Document Requirements**
- **Functional Requirements:** Define the features and capabilities of the license plate detection AI model, including real-time monitoring, anomaly detection, and predictive analytics.
- **Non-functional Requirements:** Specify performance criteria such as detection accuracy, latency, scalability, and reliability.
- **Compliance Requirements:** Document regulatory requirements for license plate detection systems
- **Data Handling Policies:** Establish policies for data acquisition, storage, and processing, outlining procedures for data encryption, access controls and audit trails.
- **Model Validation Procedures:** Document methodologies for validating the AI model, including testing procedures, validation metrics, and acceptance criteria.

#### **Evaluate**
- **Detection Accuracy:** Measure the AI model's effectiveness in accurately identifying license plates.
- **False Positive Rate:** Evaluate the frequency of false positives and their impact on customer experience and operational efficiency.
- **Model Performance Over Time:** Monitor the AI model's performance and adaptability to evolving fraud patterns and attack techniques.
- **Feedback from Stakeholders:** Gather feedback from financial institutions, regulatory bodies, and customers to assess the AI model's ####effectiveness and identify areas for improvement.
- **Continuous Monitoring and Improvement:** Implement mechanisms for ongoing evaluation and refinement of the fraud detection AI model based on performance metrics and feedback.

#### **Iterate**
- **Feedback Loop Integration:** Incorporate feedback from stakeholders and evaluation results into the model development process to enhance detection capabilities and reduce false positives.
- **Feature Engineering:** Iterate on feature selection and engineering techniques to improve the AI model's ability to detect subtle patterns indicative of fraudulent activity.
- **Algorithm Optimization:** Explore and experiment with different machine learning algorithms and techniques to optimize the fraud detection model's performance.
- **Adaptive Learning:** Implement mechanisms for the AI model to learn from new data and adapt its detection strategies to emerging fraud trends.
- **Regular Updates and Maintenance:** Schedule regular updates and maintenance activities to address software vulnerabilities, data drift, and changing regulatory requirements.

## **Risk & Uncertainties**

### **What are possible harms?**
Possible harms for a ML based fraud detection system include - mis-identifications and inconvenience to the drivers if they are sent incorrect invoices. Then, secondly missing on toll collection all together causing monetary losses to the Department of Transportation. Overall, this can add an overhead expenditure to recover and rectify transactions missed by the ML systems.

### **What are causes of mistakes?**
Mistakes can arise from stale data or incorrect data. Also, faulty pre-processing can lead to mistakes. Absence of variety in data will not able the model to generalize or learn holistically. 

# **Implementation (HOW's)**

## **Development**

### **Methodology:** 

- **ETL Pipeline:** Develop an Extract, Transform, Load (ETL) pipeline to preprocess and clean the data, including steps such as data normalization, feature engineering, and outlier detection to improve model performance. 

- **Exploratory Data Analysis**: 
1. Sizes/dimensions of bounding boxes: 
2. Locations of these boxes relative to the image space

- **Data Engineering**: 
1. The live feed video was cropped into frames.
2. Frames were fed into the yolov3 or yolov3-tiny model to detect number plates. These trained models were provided to us for license plate detection
3. Once license plates were detected, they were cropped based on their bounding boxes
4. Cropped license plates were fed into PyTesseract optical character recognition model
5. Finally, Non Maximal Suppression (NMS) model was used for post processing of detected plates.

- **Dataset Partitioning:** Although training was not required for this project but dataset partitioning module has following capabilities-
1. To change percentage of data going into training 
2. 5-fold cross validation

- **Metrics and Advanced Metrics:**  
1. Accuracy will be used to get an understanding of basic performance of the model i.e to check if indeed number plate was detected on a car, truck etc
2. Jaccard Index will be used to measure the accuracy of bounding box or segmentation mask predictions compared to ground truth annotations.


- **Model Selection:** 
1.One Stage Model YOLO (you look only once) will be used
2. It is a perfect model for real time detection focussing on speed rather than accuracy
3. Various versions of YOLO exist. For this project, yolov3 and yolov3-tiny were used.

 - **Deployment Strategy**

The deployment strategy for the License Plate Detection system involves packaging the machine learning model into a Dockerized system architecture. 
1. The system will run a main.py type of file. It will use various modules for realtime detection and print out results on the terminal as well as in a text file.
2. It will import modules and classes from data_pipeline module.
3. data_pipeline module has Pipeline class with following functions to-
4. Receive videos via udp streaming, 
5. license plate detection via yolov models 
6. Read license plate numbers via Tesseract OCR

### **High-level System Design**

The systems will comprise of three main modules namely Detection, Rectification and User Interface
1. Detection module will have  sub-modules for preprocessing videos, objection detection, non maximal suppression, OCR and master storage
2. Rectification module will have sub modules for annotation (by auditor), error and training image storage, image augmentation and retraining model
3. The interface module will have all the tools and dashboards for internal and external stake holders.

### **Development Workflow**
- **Agile Development:** Adopt an agile development methodology to facilitate iterative development, allowing for frequent feedback cycles and rapid iteration based on stakeholder inputs and evolving requirements.
- **Version Control:** Utilize version control systems such as Git to manage codebase changes and facilitate collaboration among development team members.
- **Continuous Integration and Testing:** Implement continuous integration and automated testing processes to ensure code quality, reliability, and consistency across different development stages.

## **Policy**

### **Human-Machine Interfaces**
- **User Interface Design:** Design intuitive user interfaces for analysts and investigators to interact with the license plate detection system, providing them with actionable insights and decision support tools to efficiently investigate flagged transactions.
- **Training and Education:** Provide training programs and educational resources to equip employees with the necessary skills and knowledge to effectively utilize the AI-powered  license plate detection system.

### **Regulations:**
- **Compliance:** Ensure that the  license plate detection system complies with relevant regulations and industry standards governing data privacy, security, and financial transactions, such as GDPR, PCI DSS, and regulations set forth by financial regulatory bodies.

## **Operations**

### **Continuous Deployment**
- **Automated Deployment Pipelines:** Implement automated deployment pipelines to streamline the deployment process, enabling rapid and consistent deployment of model updates and enhancements.
- **Rollback Mechanisms:** Establish rollback mechanisms to revert to previous model versions in case of deployment failures or performance issues, minimizing downtime and service disruptions.

### **Post-deployment Monitoring**
- **Performance Monitoring:** Continuously monitor the performance of the deployed fraud detection model in production, tracking key metrics such as Jaccard Index.
- **Anomaly Detection:** Implement anomaly detection techniques to identify deviations from expected model behavior or performance indicators, enabling proactive intervention to address potential issues or anomalies.

### **Maintenance**
- **Model Retraining:** Schedule periodic model retraining cycles using updated data to ensure that the fraud detection model remains effective and adaptive to evolving fraud patterns and trends.
- **System Updates and Patches:** Apply regular updates and patches to the fraud detection system to address security vulnerabilities, software bugs, and compatibility issues, minimizing risks of system downtime or breaches.

### **Quality Assurance**
- **Testing and Validation:** Conduct comprehensive testing and validation procedures across different stages of the development lifecycle, including unit testing, integration testing, and end-to-end testing, to verify the correctness, robustness, and reliability of the fraud detection system.
- **Auditing and Compliance Checks:** Perform regular audits and compliance checks to ensure that the fraud detection system adheres to established quality standards, regulatory requirements, and internal policies.


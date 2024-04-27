# **Motivation (WHY's)**

## **Why are we solving this problem?**

**Problem Statement**: To improve response rates to campaign marketing emails. 

 **Value Proposition**: Changing subject lines dynamically might result in better response rates.

 **Why is our solution a viable one**?

We are proposing a reinforcement learning based solution. Idea is to capture dynamic nature of responses to emails. Then, change the subject line based on responses.

# **Requirements (WHAT's)**

## **Scope**

### What are our goals?
- **Organizational Goals:** To improve response rates to emails in order to derive actions from potenial customers
- **System Goals:** Develop a Reinforcement Learning based system to change email subject lines to improve response rates

- **User Goals:**
- **Customer:** Provide a meaningful email subject line to respond.
- **Company:** Obtain better response rates from customers
- **Model Goals:** Build a Q Table based upon the conversion rate given the state and actions.

### What are the success criteria?
Our success criteria is to double the response rate/success rate

## **Requirements**

### **What are our(system) Assumptions?**
We assume we have the compute power (hardware), cyber security set up (hardware and software), data storage capabilities, availability of state of the art ML softwares, visualization products, project management solutions and talented hardworking man power.

### **What are our (system) Requirements?**
First and foremost - clean data ready to be fed into models. Secondly, ability to create and analyze new models. Thirdly, ability to change or update model with new incoming data.

# **Implementation (HOW's)**

### **Methodology:** 

- **Feature Engineering**: The three given datasets were cleaned up and then merged. The idea was to create a new feature called Reward. Reward was set to 1 if the customer responded on the same day as the email was sent. Otherwise reward was set to 0. For the given dataset, gender, age (of customer), Tenure and Type were chosen to create a state.

- **Dataset Partitioning:**: All the states were used to create the Q-table.

- **Metrics and Advanced Metrics:** Cumulatiive Rewards, Average Rewards, Discounted Reward, Convergence Rate were used to evaluate the model

- **Model Selection:** Q-learning based Reinforcement Learning model was chosen to capture response dynamics. 

- ** Deployment Strategy:**The deployment strategy for the Email Marketing System involves packaging the model into a Dockerized system architecture. The system will utilize a class named QLearningAgent defined in the qagent.py module to construct and train the agent . The system will be capable of receiving input data  via requests sent through Postman  which should be in the format of a .csv. The output can be seen on the Postman. The final system will be packaged into a Docker image and published on DockerHub for ease of deployment.



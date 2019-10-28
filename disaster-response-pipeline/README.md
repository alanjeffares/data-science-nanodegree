# Disaster Response Pipeline

## Introduction
This project takes two datasets containing real messages that were sent during disaster events. I create a ETL pipeline followed by a machine learning pipeline to categorize these messages so that they can be connected to an appropriate disaster relief agency. Finally, I include a Flask web application that applies this pipeline to user input and includes some descriptive plots of the data. 

<img src="https://raw.githubusercontent.com/alanjeffares/elements-of-statistical-learning/master/chapter-2/images/MSE_vs_Dimension_1.png"  width="400"> <img src="https://github.com/alanjeffares/data-science-nanodegree/blob/master/disaster-response-pipeline/app_screenshot.png"  width="400">

## Structure 
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  
|- disaster_messages.csv  
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 
|- model_results.csv
```

## Instructions:
1. (optional) Run the following commands in the project's root directory to set up the database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py DisasterResponse.db classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

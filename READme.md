# Disaster Pipeline Project

This project is one of the [Udacity's Data Science Nanodegree Program](https://www.udacity.com/course/data-scientist-nanodegree--nd025). The purpose of this project is to analyze disaster data from [Figure Eight](https://appen.com/) to build a model for an API that classifies disaster messages. 

### 1. Project Components
This project divided in the following section:
1. ETL Pipeline 
   In a Python script, `process_data.py`, write a data cleaning pipeline that:
   * Loads the `messages` and `categories` datasets
   * Merges two datasets
   * Cleans the data
   * Stores it in a SQLite database
   
   
2. ML Pipeline
   In a Python script, `train_classifier.py`, write a machine learning pipeline that:
   * Loads data from the SQLite database
   * Splits the dataset into training and test sets
   * Builds a text processing and machine learning pipeline
   * Trains and tunes a model using GridSearchCV
   * Outputs results on the test set
   * Exports the final model as pickel file
   
   
3. Flask Web App
   * Modify file paths for database and model as needed
   * Add data visualizations using Ploty in the web app

### 2. Files

- app
  * template
       * master.html - main page of web app
       * go.html - classification result page of the web app
  * run.py - Flask file that runs app
  
- data
  * disaster_categories.csv - categories dataset
  * disaster_messages.csv - messages dataset
  * process_data.py - ETL pipeline
  * DisasterResponse.db - clean database
  
- models
  * train_classifier.py - ML pipeline
  * classifier.pkl - GridSearchCV model
  
- README.md

### 3. Program Executing

1. Run the ETL Pipeline 

   *python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db*

2. Run the ML pipeline 

   *python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl*

3. Run the web app

   *python app/run.py*
   
4. Go to https://view6914b2f4-3001.udacity-student-workspaces.com/ 

### 4. Main Page of the Web app


### 5. Software Requirement

This project uses **Python 3** and the necessary libraries to complete. 


```python

```

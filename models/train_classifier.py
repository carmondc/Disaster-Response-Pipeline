import sys
import nltk 
nltk.download (['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import sqlite3
from sqlalchemy import create_engine

import pickle

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier


def load_data(database_filepath):
    """
    load data from database_filepath
    
    Args:
    database_filepath -> The path of database
    
    Return:
    X (Dataframe) -> messages feature in dataframe
    Y (Dataframe) -> variables of dataframe
    """
    #load dataset
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_table', engine)
    
    #define X and Y
    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis = 1)
    print("successful load_data")
    
    return X, Y


def tokenize(text):
    """
    split text into the original root form 
    
    Args:
    text -> the message
    
    return:
    tokens -> list of root form of the messages word
    """
    # normalization text
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    # word tokenization
    tokens = word_tokenize(text)
    # Lemmatization and remove stop word
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return tokens


def build_model():
    """
    build model to classifying the messages
    
    return:
    cv -> GridSearch classification model
    """
    
    #create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    
    #create parameters
    parameters= {
    'clf__estimator__leaf_size': [10,20]
    }
    
    #create cv
    cv = GridSearchCV(pipeline, param_grid = parameters)
    
    print("succesful build model")
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    evaluating model and print f1 score, precision and reccall for each output category of the dataset
    
    Args:
    model -> the Gridsearch classification model
    X_test -> test messages
    Y_test -> test target
   
    """
    #Prediction
    y_pred = model.predict(X_test)
    
    #print the f1 score, precision and recall for each output category of the dataset
    i = 0
    for column in Y_test:
        print('labels {}:{}'.format(i+1, column))
        print(classification_report(Y_test[column], y_pred[:, i]))
        i = i + 1
    accuracy = (y_pred == Y_test).mean()
    print("Accuracy:", accuracy)
    print("successful evaluate model")



def save_model(model, model_filepath):
    """
    save the model to a pickle file
    
    Args:
    model -> the GridSearch Classification model
    model_filepath(str) -> the path of pickle file
    """
    filename = model_filepath
    pickle.dump(model, open(filename, 'rb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    load data from dataset
    
    Args: 
    messages_filepath -> filepath of messages csv
    categories_filepath -> filepath of categories csv
    
    return:
    df(dataframe) -> dataframe of messages and categories
   
    """
    
    #load dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge datasets
    df = pd.merge(messages, categories, on = 'id')
    
    print("successful load dataset")
    return df


def clean_data(df):
    """
    clean dataset 
    
    Args:
    df(Dataframe) -> dataframe that contain two dataset
    
    return:
    df(Dataframe) -> clean dataframe
    """
    #split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    #extract a list of new column names for categories
    row = categories.head(1)
    category_colnames = row.apply(lambda x: x.str[:-2]).values.tolist()
    
    #rename columns name
    categories.columns = category_colnames
    
    #convert category values to numbers 0 to 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    print("successful split categories into separate category column")
    #Replace categories column in df with new category columns
    df.drop(['categories'], axis = 1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1, join = 'inner', sort = False)
    print("successful merge new categories dataframe")
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    print("succesful drop duplicates")
    
    return df


def save_data(df, database_filename):
    """
    save the dataframe in a database
    
    Args???
    df(DataFrame) -> A dataframe of messages and categories
    dataframe_filename(str) -> The filename of the database
    
    """
    
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages_table', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
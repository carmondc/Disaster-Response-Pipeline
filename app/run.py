import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import sqlite3


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    print('tokenize done')
    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('messages_table', engine)
print('successful load data')

# load model
model = joblib.load("../models/classifier.pkl")
print('successful load model')


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    category_names = df.iloc[:, 4:].columns
    category_counts = (df.iloc[:, 4:] != 0).sum().values
    top_category = category_counts.sort_values(ascending = False)[1]
    top_category_name = list(top_category.index)
                       
    
                     
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
       
        
        { 
            'data': [
                Bar(
                    x= category_names, 
                    y= category_counts
                )
            ],
            'layout': { 
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Genre'
                }
            }
        }, 
       
        { 
            'data': [
                Bar(
                    x = top_category_name,
                    y = top_category
                )
            ],
            'layout': {
                'title': 'Top 10 Categories',
                'yaxis': {
                    'title' :'Count'
                },
                'xaxis': {
                    'title' : 'Categories'
                }
            }
        }
                          
        
    ]
   
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    print('encode done')
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    print('predict classification done')

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
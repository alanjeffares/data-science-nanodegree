import json
import plotly
import pandas as pd
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objects import Bar
from plotly.graph_objects import Table

import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin



app = Flask(__name__)


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def parse_colnames(x: pd.Series) -> str:
    x_list = list(x[x>0].index)
    x_str = ', '.join(x_list).replace('_', ' ')
    return x_str

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('tweets', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    random_idx = np.random.randint(0, df.shape[0], 10)
    random_tweets = df['message'][random_idx].to_list()
    categories = df.loc[random_idx, df.columns[4:]]
    categories = categories.apply(parse_colnames, axis=1)
    categories = categories.tolist()
    
    cats = df.iloc[:,4:].sum()
    cats = cats.sort_values(ascending=False)
    values = cats.tolist()
    names = cats.index.tolist()
    names = [name.replace('_', ' ') for name in names]
    

    # create visuals
    graphs = [
                {
            'data': [
                Table(header=dict(values=['<b>Tweet<b>', '<b>Categories<b>'],
                                 fill_color='cornflowerblue',
                                 line_color='darkslategray',
                                 font=dict(color='white', size=16),
                                 height=40),
                         cells=dict(values=[random_tweets, categories],
                                   align=['left', 'center'],
                                   line_color='darkslategray',
                                   fill_color='whitesmoke'))
            ],

            'layout': {
                'title': 'Sample of Input Data',
                'annotations': [{
                    'text': "Refresh page for more",
                      'font': {
                      'size': 13
                    },
                    'showarrow': False,
                    'align': 'center',
                    'x': 0.5,
                    'y': 1.15,
                    'xref': 'paper',
                    'yref': 'paper',
                  }]
            }
        },       
        {
            'data': [
                Bar(
                    y=names,
                    x=values,
                    orientation="h"
            )],

            'layout': {
                'title': 'Category Frequencies',
                'height': 800,
                'yaxis': {
                    'tickangle': 0
                },
                'margin': {
                    'l': 200
                }
        }},
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Tweet Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
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
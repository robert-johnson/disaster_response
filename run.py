import json
import plotly
import pickle
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

app = Flask(__name__, template_folder='templates')


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///data/disaster_messages.db')
df = pd.read_sql_table('messages', engine)

# load model
file = open('models/message_classifier.pkl', 'rb')
model = pickle.load(file)
file.close()
# model = pickle.load("models/message_classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Main entry point for the home page of the web applicaiton
    :return: json representing charts to display
    """

    # let's see the number of requests versus the number of offers for help
    requests = df[df['request'] == 1].count()['message']
    offers = df[df['offer'] == 1].count()['message']
    labels = ['requests', 'offers']

    # of the offers, what types of aid is being offered?
    offer_types = df[(df['offer'] == 1)]
    # offer_types = offer_types[offer_types['food', 'water', 'shelter', 'transport']]
    offer_types = offer_types[offer_types.columns.intersection(['food', 'water', 'shelter', 'transport', 'other_aid'])].sum()
    offer_types = offer_types.sort_values(ascending=False)
    offer_types_labels = list(offer_types.index)

    # create graphs
    graphs = [
        {
            'data': [
                Bar(
                    x=labels,
                    y=[requests, offers]
                    # ,
                    # marker_color=['red', 'green']

                )
            ],

            'layout': {
                'title': 'Requests for Aid versus Offers to Help',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Message Type"
                },
                'barmode' : 'group'
            }
        }
        ,
        {
            'data': [
                Bar(
                    x=offer_types_labels,
                    y=offer_types
                    # ,
                    # marker_color=['blue', 'red', 'green', 'orange', 'yellow']
                )
            ],

            'layout': {
                'title': 'Help Offer Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Offer Type"
                },
                'barmode': 'group'
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
    """
    gets the query entered into the classifier and predicts which labels are relevant
    :return: informaiton on the predicted classification
    """
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

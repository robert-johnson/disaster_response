# Import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

import warnings

warnings.simplefilter('ignore')


def load_data(database_filepath):
    """Load cleaned data from the database, split into X, y sets

    Args:
    database_filename: string. Filename for SQLite database containing cleaned message data.

    Returns:
    X: dataframe. Dataframe containing features dataset.
    Y: dataframe. Dataframe containing labels dataset.
    category_names: list of strings. List containing category names.
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)

    # Create X and Y datasets
    X = df['message']
    # drop unneeded columns
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # Create list containing all category names
    category_names = list(y.columns.values)

    return X, y, category_names


def tokenize(text):
    """Normalize, tokenize and stem text string

    Args:
    text: string. String containing message for processing

    Returns:
    stemmed: list of strings. List containing normalized and stemmed word tokens
    """
    # Convert text to lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # Tokenize words
    tokens = word_tokenize(text)

    # Stem word tokens and remove stop words
    stemmer = PorterStemmer()
    stop_words = stopwords.words("english")

    # loop through the tokens, stem and remove any stop words
    processed = [stemmer.stem(word) for word in tokens if word not in stop_words]

    #  return the processed words
    return processed


# Define performance metric for use in grid search scoring object
def performance_metric(y_true, y_pred):
    """Calculate median F1 score for all of the output classifiers

        Args:
        y_true: array. Array containing actual labels.
        y_pred: array. Array containing predicted labels.

        Returns:
        score: float. Median F1 score for all of the output classifiers
        """
    f1_list = []
    for i in range(np.shape(y_pred)[1]):
        f1 = f1_score(np.array(y_true)[:, i], y_pred[:, i])
        f1_list.append(f1)

    score = np.median(f1_list)
    return score


def build_model():
    """Build a machine learning pipeline

    Args:
    None

    Returns:
    cv: gridsearchcv object. Gridsearchcv object that transforms the data, creates the
    model object and finds the optimal model parameters.
    """
    # Create pipeline with the initial values
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, min_df=5)),
        ('tfidf', TfidfTransformer(use_idf=True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10,
                                                             min_samples_split=10)))
    ])

    # Create parameters dictionary-for the grid search
    parameters = {'vect__min_df': [5, 10],
                  'tfidf__use_idf': [True, False],
                  'clf__estimator__n_estimators': [10, 15],
                  'clf__estimator__min_samples_split': [5, 10]}

    # Create scorer
    scorer = make_scorer(performance_metric)

    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer, verbose=10)
    return cv


def get_metrics(actual, predicted, columns):
    """Calculate metrics model

    Args:
    actual: Array containing actual labels.
    predicted: Array containing predicted labels.
    columns: list of strings. List containing names for each of the predicted fields.

    Returns:
    metrics_df: dataframe. Dataframe containing the accuracy, precision, recall
    and f1 score for a given set of actual and predicted labels.
    """
    metrics = []

    # Calculate evaluation metrics for each set of labels
    for i in range(len(columns)):
        accuracy = accuracy_score(actual[:, i], predicted[:, i])
        precision = precision_score(actual[:, i], predicted[:, i], average='macro')
        recall = recall_score(actual[:, i], predicted[:, i], average='macro')
        f1 = f1_score(actual[:, i], predicted[:, i], average='macro')

        metrics.append([accuracy, precision, recall, f1])

    # Create dataframe containing metrics
    metrics = np.array(metrics)
    metrics_df = pd.DataFrame(data=metrics, index=columns, columns=['Accuracy', 'Precision', 'Recall', 'F1'])

    return metrics_df


def evaluate_model(model, X_test, y_test, category_names):
    """Returns test accuracy, precision, recall and F1 score for fitted model

    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.

    Returns:
    None
    """
    # Predict labels for test dataset
    y_pred = model.predict(X_test)

    # Calculate and print evaluation metrics
    eval_metrics = get_metrics(np.array(y_test), y_pred, category_names)
    print(eval_metrics)


def save_model(model, model_filepath):
    """Saves the model in pickle format

    Args:
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved

    Returns:
    None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        print('Loading data')

        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model')
        model = build_model()

        print('Training model')
        model.fit(X_train, y_train)

        print('Evaluating model')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model')
        save_model(model, model_filepath)

    else:
        print('Please provide the following arguments:\n\n'
              '(1) filepath to the messages db\n'
              '(2) filepath to the output model\n\n'
              'Example: python train_classifier.py data/disaster_messages.db models/message_classifier.pkl')


if __name__ == '__main__':
    main()

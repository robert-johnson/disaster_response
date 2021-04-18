# Disaster Response Project

## Installation
The following Python packages are required to successfully run this project: json, plotly, pandas, nltk, flask, sklearn, sqlalchemy, sys, numpy, re, pickle, warnings.

## Project Overview
This repository contains a web app that can be used during a disaster to classify received messages into several categories, so that the message can be directed to the proper resources, or filtered as unimportant. 

The app uses a model to classify any new messages received though the web UI, and also contains the code used to prepare the data and train the model doing the classification.

## File Descriptions
* **process_data.py**: takes csv files and created a sqlite db with the messages and category columns
* **train_classifier.py**: takes the information from the db and performs the NLP processing, then uses the data to train the model
* **ETL Pipeline Preparation.ipynb**: The code and analysis contained in this Jupyter notebook was used in the development of process_data.py. process_data.py effectively automates this notebook.
* **ML Pipeline Preparation.ipynb**: The code and analysis contained in this Jupyter notebook was used in the development of train_classifier.py. In particular, it contains the analysis used to tune the ML model and determine which algorithm to use. train_classifier.py effectively automates the model fitting process contained in this notebook.
* **data**: contains source csv, output db
* **data**: contains saved model files
* **app**: contains files for the web app

## Running Instructions
### ***Run process_data.py***
1. Save the data folder in the current working directory.
2. From the current working directory, run the following command:
`python process_data.py data/messages.csv data/categories.csv data/disaster_messages.db`

### ***Run train_classifier.py***
1. In the current working directory, create a folder called 'models'.
2. From the current working directory, run the following command:
`python train_classifier.py data/disaster_messages.db models/message_classifier.pkl`

### ***Run the web app***
1. Save the app folder in the current working directory.
2. Run the following command in the app directory:
    `python run.py`
3. Go to http://0.0.0.0:3001/

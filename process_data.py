# Import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data():
    """Load and merge messages and categories datasets

    Args:
    None

    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """

    # Load messages dataset
    messages = pd.read_csv('data/messages.csv')

    # Load categories dataset
    categories = pd.read_csv('data/categories.csv')

    # Merge datasets
    df = messages.merge(categories, how='left', on=['id'])

    return df


def clean_data(df):
    """Clean dataframe transforming values and adding columns.

    Args:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.

    Returns:
    df: dataframe. Dataframe containing cleaned version of input dataframe.
    """

    # split the categories into individual columns
    categories = df['categories'].str.split(';', expand=True)

    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # apply a lambda function to parse off the last 2 characters
    category_colnames = row.transform(lambda x: x[:-2]).tolist()

    # Rename the columns to the column names
    categories.columns = category_colnames

    # Convert  category values to numeric values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop the original categories column from `df`, because it's not needed
    df.drop('categories', axis=1, inplace=True)

    # Concatenate the original dataframe with the new categories columns
    df = pd.concat([df, categories], axis=1)

    # Drop any duplicates
    df.drop_duplicates(inplace=True)

    # Remove rows with a related value of 2 from the dataset
    # df = df[df['related'] != 2]

    return df


def save_data(df):
    """Save cleaned data into an SQLite database.

    Args:
    df: dataframe. Dataframe containing cleaned version of merged message and
    categories data.
    database_filename: string. Filename for output database.

    Returns:
    None
    """
    engine = create_engine('sqlite:///data/DisasterMessages.db')
    df.to_sql('messages', engine, index=False)


def main():
    """
    Main entry point to the application
    """

    # load the data from the messages and categories csv files
    df = load_data()

    # clean the loaded data
    df = clean_data(df)

    # save the data to the database for future processing
    save_data(df)


if __name__ == '__main__':
    main()
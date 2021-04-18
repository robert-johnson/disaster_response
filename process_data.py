# Import packages
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_path, categories_path):
    """Load and merge messages and categories datasets

    Args:
    None

    Returns:
    df: dataframe. Dataframe containing merged content of messages and categories datasets.
    """

    # Load messages dataset
    messages = pd.read_csv(messages_path)

    # Load categories dataset
    categories = pd.read_csv(categories_path)

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
    category_columns = row.transform(lambda x: x[:-2]).tolist()

    # Rename the columns to the column names
    categories.columns = category_columns

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


def save_data(df, database_path):
    """Save cleaned data into an SQLite database.

    Args:
    df: dataframe. Dataframe containing cleaned version of merged message and
    categories data.
    database_filename: string. Filename for output database.

    Returns:
    None
    """
    engine = create_engine(f'sqlite:///' + database_path)
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    """
    Main entry point to the application
    """

    if len(sys.argv) == 4:

        messages_path, categories_path, database_path = sys.argv[1:]

        print('Merging messages/categories')
        df = load_data(messages_path, categories_path)

        print('Cleaning data')
        df = clean_data(df)

        print('Saving data')
        save_data(df, database_path)

    else:
        print('Please provide the following arguments:\n\n'
              '(1) filepath to the messages.csv\n'
              '(2) filepath to the categories.csv\n'
              '(3) filepath to the output database\n\n'
              'Example: python process_data.py data/messages.csv data/categories.csv '
              'data/DisasterResponse.db')


if __name__ == '__main__':
    main()

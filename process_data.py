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
    df = messages.merge(categories, how = 'left', on = ['id'])
    
    return df

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(db_path:str):
    '''Load "vendor invoice" data from the SQLite database.'''
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM vendor_invoice"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def prepare_features_and_target_variables(df:pd.DataFrame):
    ''' Prepare features and target variable for modeling.'''
    x = df[["Dollars"]]
    y = df[["Freight"]]
    return x, y

def split_data(x, y, test_size=0.2, random_state=42):
    '''Split the data into training and testing sets.'''
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
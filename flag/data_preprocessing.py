import sqlite3
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(db_path):
    '''Load "purchases", and "vendor invoice" data from the SQLite database.'''
    conn = sqlite3.connect(db_path)
    query = """
    WITH purchase_agg AS (
    SELECT
    p.PONumber,
    COUNT(DISTINCT p.Brand) AS TotalBrands,
    SUM(p.Quantity) AS TotalQuantity,
    SUM(p.Dollars) AS TotalDollars,
    AVG(julianday(p.ReceivingDate) - julianday(p.PODate)) AS AverageReceivingDelay
    FROM purchases p
    GROUP BY p.PONumber
    )
    SELECT
    vi.PONumber,
    vi.Quantity AS InvoiceQuantity,
    vi.Dollars AS InvoiceDollars,
    vi.Freight,
    (julianday(vi.InvoiceDate) - julianday(vi.PODate)) AS InvoicingDelay,
    (julianday(vi.PayDate) - julianday(vi.InvoiceDate)) AS PayingDelay,
    pa.TotalBrands,
    pa.TotalQuantity,
    pa.TotalDollars,
    pa.AverageReceivingDelay
    FROM vendor_invoice vi
    LEFT JOIN purchase_agg pa
    ON vi.PONumber = pa.PONumber
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def label_risks(row):
    '''Consider large discrepancies between invoice and purchase order as a risk factor.'''
    if (abs(row["InvoiceDollars"] - row["TotalDollars"]) > 5):
        return 1
    '''Consider delays as a risk factor.'''
    if row["AverageReceivingDelay"] > 10:
        return 1
    return 0

def label(df):
    '''Label the data with a binary flag indicating potential risks.'''
    df["FlagInvoice"] = df.apply(label_risks, axis=1)
    return df

def prepare_features_and_target_variables(df, features, target_variables):
    ''' Prepare features and target variable for modeling.'''
    x = df[features]
    y = df[target_variables]
    return x, y

def split_data(x, y, test_size=0.2, random_state=42):
    '''Split the data into training and testing sets.'''
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

def scale_features(x_train, x_test):
    '''Scale features using StandardScaler.'''
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    joblib.dump(scaler, "models/scaler.pkl")
    return x_train_scaled, x_test_scaled
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def load_model(model_path):
    """Load the pre-trained model from the specified path."""
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    return model

INVOICE_FLAGGER_MODEL_PATH = BASE_DIR / "flag" / "models" / "invoice_flagger.pkl"

flagger_model = load_model(INVOICE_FLAGGER_MODEL_PATH)

SCALER_MODEL_PATH = BASE_DIR / "flag" / "models" / "scaler.pkl"

scaler_model = load_model(SCALER_MODEL_PATH)

def flag_invoice(input_data):
    '''Flag based on input data using the pre-trained model.'''
    input_df = pd.DataFrame(input_data)
    required_columns = ["InvoiceQuantity", "InvoiceDollars", "Freight", "TotalQuantity", "TotalDollars"]
    if not all(col in input_df.columns for col in required_columns):
        raise ValueError(f"Input data must contain the following columns: {required_columns}")
    input_df = input_df[required_columns]
    scaled_df = scaler_model.transform(input_df)
    input_df["Flagged"] = flagger_model.predict(scaled_df).round()
    return input_df

if __name__ == "__main__":
    '''Example input data for flagging.'''
    example_input = {
        "InvoiceQuantity": [90, 500, 6000],
        "InvoiceDollars": [1000, 5000, 45000],
        "Freight": [6, 30, 300],
        "TotalQuantity": [90, 500, 6000],
        "TotalDollars": [1000, 5000, 45000]
    }
    flagged = flag_invoice(example_input)
    print(flagged)
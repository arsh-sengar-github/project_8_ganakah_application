import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FREIGHT_COST_PREDICTOR_MODEL_PATH = BASE_DIR / "predict" / "models" / "freight_cost_predictor.pkl"

def load_model(model_path=FREIGHT_COST_PREDICTOR_MODEL_PATH):
    """Load the pre-trained model from the specified path."""
    with open(model_path, "rb") as file:
        model = joblib.load(file)
    return model

predictor_model = load_model()

def predict_freight_cost(input_data):
    '''Predict freight cost based on input data using the pre-trained model.'''
    input_df = pd.DataFrame(input_data)
    required_columns = ["Dollars"]
    if not all(col in input_df.columns for col in required_columns):
        raise ValueError(f"Input data must contain the following columns: {required_columns}")
    input_df = input_df[required_columns]
    input_df["PredictedFreight"] = predictor_model.predict(input_df).round()
    return input_df

if __name__ == "__main__":
    '''Example input data for prediction.'''
    example_input = {
        "Dollars": [4000, 8000, 11000],
    }
    predicted = predict_freight_cost(example_input)
    print(predicted)
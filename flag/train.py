import joblib
from pathlib import Path
from data_preprocessing import load_data, label, prepare_features_and_target_variables, split_data, scale_features
from model_evaluation import train_random_forest, evaluate_model

FEATURES = ["InvoiceQuantity", "InvoiceDollars", "Freight", "TotalQuantity", "TotalDollars"]
TARGET_VARIABLES = "FlagInvoice"

def main():
    db_path = Path(__file__).parent.parent / "data" / "inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    '''Load "purchases", and "vendor invoice" data from the SQLite database.'''
    df = load_data(db_path)
    '''Label the data with a binary flag indicating potential risks.'''
    df = label(df)
    ''' Prepare features and target variable for modeling.'''
    x, y = prepare_features_and_target_variables(df, FEATURES, TARGET_VARIABLES)
    '''Split the data into training and testing sets.'''
    x_train, x_test, y_train, y_test = split_data(x, y, test_size=0.2, random_state=42)
    '''Scale features using StandardScaler.'''
    x_train_scaled, x_test_scaled = scale_features(x_train, x_test)
    '''Train a Random Forest Classifier model.'''
    grid_search = train_random_forest(x_train_scaled, y_train)
    '''Evaluate the model and print performance metrics.'''
    evaluate_model(grid_search.best_estimator_, x_test_scaled, y_test, "Random Forest Classifier")
    '''Save the best model.'''
    model_path = model_dir / "invoice_flagger.pkl"
    joblib.dump(grid_search.best_estimator_, model_path)

if __name__ == "__main__":
    main()

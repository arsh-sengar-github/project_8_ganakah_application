import joblib
from pathlib import Path
from data_preprocessing import load_data, prepare_features_and_target_variables, split_data
from model_evaluation import train_linear_regression, train_decision_tree, train_random_forest, evaluate_model

def main():
    db_path = Path(__file__).parent.parent / "data" / "inventory.db"
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    '''Load "vendor invoice" data from the SQLite database.'''
    df = load_data(db_path)
    '''Prepare features and target variables.'''
    x, y = prepare_features_and_target_variables(df)
    '''Split the data into training and testing sets.'''
    x_train, x_test, y_train, y_test = split_data(x, y)
    '''Train Linear Regression model.'''
    lr_model = train_linear_regression(x_train, y_train)
    '''Train Decision Tree Regressor model.'''
    dt_model = train_decision_tree(x_train, y_train)
    '''Train Random Forest Regressor model.'''
    rf_model = train_random_forest(x_train, y_train)
    '''Evaluate models.'''
    results = []
    results.append(evaluate_model(lr_model, x_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, x_test, y_test, "Decision Tree Regressor"))
    results.append(evaluate_model(rf_model, x_test, y_test, "Random Forest Regressor"))
    '''Select the best model, based on Mean Absolute Error.'''
    best_model_details = min(results, key=lambda x: x['mae'])
    best_model_name = best_model_details['model_name']
    best_model = {
        "Linear Regression": lr_model,
        "Decision Tree Regressor": dt_model,
        "Random Forest Regressor": rf_model
    }[best_model_name]
    '''Save the best model.'''
    model_path = model_dir / "freight_cost_predictor.pkl"
    joblib.dump(best_model, model_path)
    print(f"Best model: {best_model_name} saved to {model_path}")

if __name__ == "__main__":
    main()
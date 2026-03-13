from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

def train_linear_regression(x_train, y_train):
    '''Train a Linear Regression model.'''
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def train_decision_tree(x_train, y_train, max_depth=4):
    '''Train a Decision Tree Regressor model.'''
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)
    return model

def train_random_forest(x_train, y_train, max_depth=4):
    '''Train a Random Forest Regressor model.'''
    model = RandomForestRegressor(max_depth=max_depth, random_state=42)
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test, model_name:str) -> dict:
    '''Evaluate the model and print performance metrics.'''
    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds) * 100
    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}%")
    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }
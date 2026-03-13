from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score

def train_random_forest(x_train, y_train):
    '''Train a Random Forest Classifier model.'''
    rf = RandomForestClassifier(n_jobs=-1, random_state=42)
    param_grid = {
        "n_estimators": [100, 200, 300],
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [None, 4, 6],
        "min_samples_split": [2, 4, 6],
        "min_samples_leaf": [1, 3, 5]
    }
    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring=scorer, cv=5, verbose=2, n_jobs=-1)
    grid_search.fit(x_train, y_train)
    return grid_search

def evaluate_model(model, x_test, y_test, model_name):
    '''Evaluate the model and print performance metrics.'''
    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
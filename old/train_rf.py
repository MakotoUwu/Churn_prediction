import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
    - X_train (pd.DataFrame): Training feature matrix.
    - y_train (pd.Series): Training target variable.
    
    Returns:
    - RandomForestClassifier: Trained model.
    """
    # Initialize the Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)

    # Train the model
    rf_classifier.fit(X_train, y_train)
    
    return rf_classifier

def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate the model using accuracy, precision, recall, and ROC-AUC.
    
    Args:
    - model: Trained model.
    - X_test (pd.DataFrame): Test feature matrix.
    - y_test (pd.Series): Test target variable.
    
    Returns:
    - Dict[str, float]: Dictionary of evaluation metrics.
    """
    # Predictions on test data
    y_pred = model.predict(X_test)

    # Evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred)
    }
    
    return metrics

def plot_feature_importances(model, X_train: pd.DataFrame):
    """
    Plot feature importances of the model.
    
    Args:
    - model: Trained model.
    - X_train (pd.DataFrame): Training feature matrix.
    """
    # Extract feature importances
    importances = model.feature_importances_

    # Visualize feature importances
    plt.figure(figsize=(10, 8))
    indices = np.argsort(importances)[::-1]
    plt.title("Feature Importances")
    plt.bar(range(X_train.shape[1]), importances[indices], align="center")
    plt.xticks(range(X_train.shape[1]), X_train.columns[indices], rotation=90)
    plt.tight_layout()
    plt.show()

# Here's an example of how you can use these functions (you can add this part to the main execution of your script):
# model = train_random_forest(X_train, y_train)
# metrics = evaluate_model(model, X_test, y_test)
# print(metrics)
# plot_feature_importances(model, X_train)

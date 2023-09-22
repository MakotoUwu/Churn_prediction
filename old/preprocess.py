import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Dict

def encode_categorical_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features in the dataset.
    
    Args:
    - data (pd.DataFrame): The input dataframe.
    
    Returns:
    - pd.DataFrame: Dataframe with encoded features.
    """
    # Label Encoding for binary categories
    label_encoders = {}
    for column in ['Attrition_Flag', 'Gender']:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # One-Hot Encoding for multi-category columns
    data = pd.get_dummies(data, columns=['Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category'])
    
    return data

def scale_features(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Scale specified columns in the dataset.
    
    Args:
    - data (pd.DataFrame): The input dataframe.
    - columns (List[str]): List of columns to be scaled.
    
    Returns:
    - pd.DataFrame: Dataframe with scaled features.
    """
    scaler = StandardScaler()
    data[columns] = scaler.fit_transform(data[columns])
    
    return data

def select_features(X: pd.DataFrame, y: pd.Series) -> List[float]:
    """
    Determine feature importances using a RandomForest model.
    
    Args:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.
    
    Returns:
    - List[float]: Feature importances.
    """
    clf = RandomForestClassifier()
    clf.fit(X, y)
    
    return clf.feature_importances_

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = 0.3) -> Tuple[pd.DataFrame]:
    """
    Split data into training and testing sets.
    
    Args:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.
    - test_size (float): Proportion of data to be used for testing.
    
    Returns:
    - Tuple[pd.DataFrame]: X_train, X_test, y_train, y_test.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

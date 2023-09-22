import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(filename):
    """Load the dataset."""
    data = pd.read_csv(filename)
    return data

def encode_data(data):
    """Encode categorical columns."""
    data['Attrition_Flag'] = data['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})
    data_encoded = pd.get_dummies(data, drop_first=True)
    return data_encoded

def scale_features(X):
    """Scale the features."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def split_data(X, y, test_size=0.3, random_state=42):
    """Split the data into training and test sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

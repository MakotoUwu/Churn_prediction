import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(filepath):
    """Load the dataset from a given filepath."""
    return pd.read_csv("./data/BankChurnes_clean.csv")

def preprocess_data(df):
    """Preprocess the data: encode categorical columns, split into train/test, and scale features."""
    
    # Drop unnecessary columns
    df = df.drop(columns=['CLIENTNUM'])
    
    # Encode categorical columns
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    
    # Split the data into training and testing sets
    X = df.drop(columns=['Attrition_Flag'])
    y = df['Attrition_Flag']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, label_encoders, scaler
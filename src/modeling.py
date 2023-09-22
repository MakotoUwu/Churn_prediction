from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def train_random_forest(X_train, y_train, random_state=42):
    """Train Random Forest classifier."""
    rf_classifier = RandomForestClassifier(random_state=random_state)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def train_gradient_boosting(X_train, y_train, random_state=42):
    """Train Gradient Boosting classifier."""
    gb_classifier = GradientBoostingClassifier(random_state=random_state)
    gb_classifier.fit(X_train, y_train)
    return gb_classifier

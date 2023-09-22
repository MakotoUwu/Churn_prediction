from src.preprocessing import load_data, encode_data, scale_features, split_data
from src.visualization import plot_distribution
from src.modeling import train_random_forest, train_gradient_boosting
from src.clustering import determine_optimal_clusters, apply_kmeans
from src.evaluation import evaluate_model_performance, classification_metrics

# Load data
data = load_data('data/BankChurners_clean.csv')

# Preprocess data
data_encoded = encode_data(data)
X = data_encoded.drop('Attrition_Flag', axis=1)
y = data_encoded['Attrition_Flag']
X_scaled = scale_features(X)
X_train, X_test, y_train, y_test = split_data(X_scaled, y)

# Visualization
plot_distribution(data, 'Attrition_Flag')

# Train models
rf_model = train_random_forest(X_train, y_train)
gb_model = train_gradient_boosting(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)

# Evaluate models
rf_metrics = evaluate_model_performance(y_test, y_pred_rf)
gb_metrics = evaluate_model_performance(y_test, y_pred_gb)

print("Random Forest Metrics:", rf_metrics)
print("Gradient Boosting Metrics:", gb_metrics)

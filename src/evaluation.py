from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def evaluate_model_performance(y_true, y_pred):
    """Evaluate model performance."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1, 'ROC AUC': roc_auc}
    return metrics

def classification_metrics(y_true, y_pred):
    """Generate classification report and confusion matrix."""
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    return report, cm

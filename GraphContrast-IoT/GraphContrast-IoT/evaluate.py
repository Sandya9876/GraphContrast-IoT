from src.metrics import compute_metrics

def evaluate_predictions(y_true, scores, y_pred):
    return compute_metrics(y_true, scores, y_pred)

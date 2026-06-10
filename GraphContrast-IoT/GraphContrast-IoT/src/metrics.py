import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix

def compute_metrics(y_true, scores, y_pred):
    out = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try: out['roc_auc'] = float(roc_auc_score(y_true, scores))
    except Exception: out['roc_auc'] = None
    try: out['pr_auc'] = float(average_precision_score(y_true, scores))
    except Exception: out['pr_auc'] = None
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    out.update({'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),'false_alarm_rate':float(fp/(fp+tn+1e-9))})
    return out

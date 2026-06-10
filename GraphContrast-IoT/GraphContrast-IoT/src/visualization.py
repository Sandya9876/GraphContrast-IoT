from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay


def plot_loss(losses, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.plot(losses); plt.xlabel('Epoch'); plt.ylabel('InfoNCE Loss'); plt.title('Training Loss Convergence')
    plt.tight_layout(); plt.savefig(path, dpi=600); plt.close()

def plot_scores(scores, threshold, y, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.plot(scores, label='Anomaly score'); plt.axhline(threshold, linestyle='--', label='Threshold')
    if y is not None: plt.scatter(np.where(y==1)[0], scores[y==1], s=8, label='True anomaly')
    plt.xlabel('Window Index'); plt.ylabel('Score'); plt.title('Embedding-Distance Anomaly Scores'); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=600); plt.close()

def plot_roc_pr(y, scores, out_dir, prefix):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if len(set(y)) < 2: return
    fpr,tpr,_=roc_curve(y,scores); plt.figure(); plt.plot(fpr,tpr); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve'); plt.tight_layout(); plt.savefig(Path(out_dir)/f'{prefix}_roc.png', dpi=600); plt.close()
    p,r,_=precision_recall_curve(y,scores); plt.figure(); plt.plot(r,p); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision-Recall Curve'); plt.tight_layout(); plt.savefig(Path(out_dir)/f'{prefix}_pr.png', dpi=600); plt.close()

def plot_cm(y, pred, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix(y,pred,labels=[0,1]), display_labels=['Normal','Anomaly'])
    disp.plot(values_format='d'); plt.title('Confusion Matrix'); plt.tight_layout(); plt.savefig(path,dpi=600); plt.close()

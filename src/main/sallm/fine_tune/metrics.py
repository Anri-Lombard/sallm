import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(p):
    preds = np.argmax(p.predictions, axis=-1).flatten()
    labels = p.label_ids.flatten()
    mask = labels != -100
    preds, labels = preds[mask], labels[mask]
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

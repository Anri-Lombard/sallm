from seqeval.metrics import classification_report
from typing import List, Dict, Any


class MetricsComputer:
    def __init__(self, id_to_label: Dict[int, str], pad_token_id: int):
        self.id_to_label = id_to_label
        self.pad_token_id = pad_token_id

    def compute(
        self, pred_ids_list: List[List[int]], label_ids_list: List[List[int]]
    ) -> Dict[str, float]:
        true_labels, true_preds = [], []
        for pred_ids, label_ids in zip(pred_ids_list, label_ids_list):
            preds, labels = [], []
            for pred_id, label_id in zip(pred_ids, label_ids):
                if label_id != self.pad_token_id:
                    labels.append(self.id_to_label[label_id])
                    preds.append(self.id_to_label[pred_id])
            true_labels.append(labels)
            true_preds.append(preds)

        report = classification_report(
            true_labels, true_preds, output_dict=True, zero_division=0
        )

        metrics = {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1-score": float(report["weighted avg"]["f1-score"]),
        }
        return metrics

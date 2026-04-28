from __future__ import annotations

import collections
import re


def compute_ner_span_f1(
    references: list[str],
    predictions: list[str],
) -> float:
    items = [
        (_normalize_ner_prediction(reference), _normalize_ner_prediction(prediction))
        for reference, prediction in zip(references, predictions, strict=False)
    ]
    return _span_f1_agg(items)


def build_ner_debug_record(reference: str, prediction: str) -> dict[str, object]:
    normalized_reference = _normalize_ner_prediction(reference)
    normalized_prediction = _normalize_ner_prediction(prediction)
    gold_spans = _tags_to_spans(normalized_reference)
    predicted_spans = _tags_to_spans(normalized_prediction)

    unmatched_gold = list(gold_spans)
    true_positive_spans: list[tuple[str, str]] = []
    false_positive_spans: list[tuple[str, str]] = []
    for span in predicted_spans:
        if span in unmatched_gold:
            true_positive_spans.append(span)
            unmatched_gold.remove(span)
        else:
            false_positive_spans.append(span)

    false_negative_spans = unmatched_gold
    true_positive = len(true_positive_spans)
    false_positive = len(false_positive_spans)
    false_negative = len(false_negative_spans)
    precision = true_positive / (true_positive + false_positive + 1e-13)
    recall = true_positive / (true_positive + false_negative + 1e-13)
    f1 = 2.0 * ((precision * recall) / (precision + recall + 1e-13))

    return {
        "normalized_reference": normalized_reference,
        "normalized_prediction": normalized_prediction,
        "gold_spans": _format_spans(gold_spans),
        "predicted_spans": _format_spans(predicted_spans),
        "true_positive_spans": _format_spans(true_positive_spans),
        "false_positive_spans": _format_spans(false_positive_spans),
        "false_negative_spans": _format_spans(false_negative_spans),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "empty_prediction": not bool(prediction.strip()),
        "empty_normalized_prediction": not bool(normalized_prediction.strip()),
        "parse_failed": bool(prediction.strip()) and not bool(predicted_spans),
    }


def compute_pos_token_accuracy(
    references: list[str],
    predictions: list[str],
) -> float:
    if not references or not predictions:
        return 0.0

    scores: list[float] = []
    for reference, prediction in zip(references, predictions, strict=False):
        ref_tags = _extract_pos_tags(reference)
        pred_tags = _extract_pos_tags(prediction, fallback=["invalid"])
        if not ref_tags:
            scores.append(0.0)
            continue

        aligned = min(len(ref_tags), len(pred_tags))
        correct = sum(
            1
            for index in range(aligned)
            if str(ref_tags[index]) == str(pred_tags[index])
        )
        scores.append(correct / len(ref_tags))

    return sum(scores) / len(scores) if scores else 0.0


def build_pos_debug_record(reference: str, prediction: str) -> dict[str, object]:
    ref_tags = _extract_pos_tags(reference)
    pred_tags = _extract_pos_tags(prediction, fallback=["invalid"])
    aligned = min(len(ref_tags), len(pred_tags))
    mismatches = [
        {
            "index": index,
            "reference": str(ref_tags[index]),
            "prediction": str(pred_tags[index]),
        }
        for index in range(aligned)
        if str(ref_tags[index]) != str(pred_tags[index])
    ]
    correct = aligned - len(mismatches)
    accuracy = correct / len(ref_tags) if ref_tags else 0.0
    return {
        "reference_tags": ref_tags,
        "prediction_tags": pred_tags,
        "reference_tag_count": len(ref_tags),
        "prediction_tag_count": len(pred_tags),
        "aligned_tag_count": aligned,
        "length_delta": len(pred_tags) - len(ref_tags),
        "mismatches": mismatches,
        "token_accuracy": accuracy,
        "empty_prediction": not bool(prediction.strip()),
        "parse_failed": bool(prediction.strip()) and pred_tags == ["invalid"],
    }


def _normalize_ner_prediction(text: str) -> str:
    label_dict = {
        "person": "PER",
        "location": "LOC",
        "organization": "ORG",
        "counties": "LOC",
        "places": "LOC",
        "people": "PER",
        "persons": "PER",
        "company": "ORG",
        "country": "LOC",
        "continent": "LOC",
        "time": "DATE",
        "date": "DATE",
        "per": "PER",
        "loc": "LOC",
        "org": "ORG",
    }
    normalized = text.lower()
    for key, value in label_dict.items():
        normalized = normalized.replace(key, value)

    normalized = "$".join(item for item in normalized.split("$$"))
    normalized = normalized.rstrip("$")
    normalized = normalized.replace("\n", "$").strip()

    matches = re.findall(r"\b(PER|LOC|ORG|DATE):\s*([^$]+)", normalized)
    formatted_entities: list[str] = []
    for label, values in matches:
        for entity in (value.strip() for value in values.split(",")):
            if entity.lower() != "none":
                formatted_entities.append(f"{label.lower()}: {entity}")
    return " $ ".join(formatted_entities)


def _span_f1_agg(items: list[tuple[str, str]]) -> float:
    true_positives: dict[str, int] = collections.defaultdict(int)
    false_positives: dict[str, int] = collections.defaultdict(int)
    false_negatives: dict[str, int] = collections.defaultdict(int)

    for target, prediction in items:
        gold_spans = _tags_to_spans(target)
        predicted_spans = _tags_to_spans(prediction)

        for span in predicted_spans:
            if span in gold_spans:
                true_positives[span[0]] += 1
                gold_spans.remove(span)
            else:
                false_positives[span[0]] += 1
        for span in gold_spans:
            false_negatives[span[0]] += 1

    true_positive = sum(true_positives.values())
    false_positive = sum(false_positives.values())
    false_negative = sum(false_negatives.values())
    precision = true_positive / (true_positive + false_positive + 1e-13)
    recall = true_positive / (true_positive + false_negative + 1e-13)
    return 2.0 * ((precision * recall) / (precision + recall + 1e-13))


def _format_spans(spans: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"label": label, "text": text} for label, text in spans]


def _tags_to_spans(tag_sequence: str, delimiter: str = "$$") -> list[tuple[str, str]]:
    tag_sequence_split = [
        item.strip()
        for sub in tag_sequence.strip().split(delimiter)
        for item in sub.split("$")
        if item
    ]
    tag_sequence_split = [
        item.strip()
        for value in tag_sequence_split
        for sub in value.split(". ")
        for item in sub.split(", ")
    ]
    tags_entities: list[tuple[str, str]] = []
    for tag_entity in tag_sequence_split:
        tag_entity_split = tag_entity.split(": ")
        if len(tag_entity_split) != 2:
            continue
        tag = _normalize_span_text(tag_entity_split[0].strip())
        entity = _normalize_span_text(tag_entity_split[1].rstrip().lstrip())
        tags_entities.append((tag, entity))
    return tags_entities


def _normalize_span_text(text: str) -> str:
    normalized = re.sub(r"\s{3,}|\t", "", text)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r'[!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@.""-,`]+', " ", normalized)
    normalized = re.sub(r"\b(a|an|the)\b", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.lower().strip()


def _extract_pos_tags(text: str, fallback: list[str] | None = None) -> list[str]:
    tags = [pos for _, pos in re.findall(r"\('([^']*)', '([^']*)'\)", text)]
    if tags:
        return tags
    return list(fallback or [])

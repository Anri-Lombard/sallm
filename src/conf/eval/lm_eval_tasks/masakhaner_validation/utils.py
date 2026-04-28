import collections
import re

NER_TAGS = [
    "O",
    "B-PER",
    "I-PER",
    "B-ORG",
    "I-ORG",
    "B-LOC",
    "I-LOC",
    "B-DATE",
    "I-DATE",
]


def doc_to_target(doc):
    return transform_text(doc["ner_tags"], doc.get("tokens"))


def transform_text(text, tokens=None):
    if isinstance(text, list):
        if tokens is None:
            tokens = []
        return transform_iob_tags(tokens, text)

    entities = []
    current_entity_tokens = []
    current_tag = ""

    for pair in text.split("\n"):
        if pair:
            word, tag = pair.strip().split(": ")
            tag = tag.upper()
            word = word.strip(",.").strip()

            if tag.startswith("B-"):
                if current_entity_tokens:
                    entity = " ".join(current_entity_tokens)
                    entities.append(f"{current_tag}: {entity}")
                current_tag = tag.split("-")[1]
                current_entity_tokens = [word]
            elif tag.startswith("I-") and tag.split("-")[1] == current_tag:
                current_entity_tokens.append(word)
            else:
                if current_entity_tokens:
                    entity = " ".join(current_entity_tokens)
                    entities.append(f"{current_tag}: {entity}")
                    current_entity_tokens = []
                    current_tag = ""
    if current_entity_tokens:
        entity = " ".join(current_entity_tokens)
        entities.append(f"{current_tag}: {entity}")

    return " $$ ".join(entities)


def transform_iob_tags(tokens, tag_ids):
    entities = []
    current_entity_tokens = []
    current_tag = ""

    for token, tag_id in zip(tokens, tag_ids, strict=False):
        try:
            tag = NER_TAGS[int(tag_id)]
        except (ValueError, IndexError, TypeError):
            tag = str(tag_id).upper()

        if tag.startswith("B-"):
            if current_entity_tokens:
                entity = " ".join(current_entity_tokens)
                entities.append(f"{current_tag}: {entity}")
            current_tag = tag.split("-", 1)[1]
            current_entity_tokens = [str(token)]
        elif tag.startswith("I-") and tag.split("-", 1)[1] == current_tag:
            current_entity_tokens.append(str(token))
        else:
            if current_entity_tokens:
                entity = " ".join(current_entity_tokens)
                entities.append(f"{current_tag}: {entity}")
            current_entity_tokens = []
            current_tag = ""

    if current_entity_tokens:
        entity = " ".join(current_entity_tokens)
        entities.append(f"{current_tag}: {entity}")

    return " $$ ".join(entities)


def span_f1_agg(items):
    true_positives = collections.defaultdict(int)
    false_positives = collections.defaultdict(int)
    false_negatives = collections.defaultdict(int)

    def normalize_text(strings):
        def get_blank_spaces_pattern():
            return re.compile(r"\s{3,}|\t")

        def remove_blank_spaces(text):
            text = re.sub(pattern=get_blank_spaces_pattern(), repl="", string=text)
            text = re.sub(r"\s+", " ", text)
            return text

        def remove_punctuation(text):
            my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@.""-,`'
            text = re.sub("[" + my_punctuation + "]+", " ", str(text))
            return text

        def remove_articles(text):
            regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
            return re.sub(regex, " ", text)

        return remove_blank_spaces(remove_articles(remove_punctuation(strings))).lower()

    def tags_to_spans(tag_sequence, delimiter="$$"):
        if isinstance(tag_sequence, list):
            tag_sequence = " ".join(i.strip() for i in tag_sequence)
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
        tags_entities = []
        for tag_entity in tag_sequence_split:
            tag_entity_split = tag_entity.split(": ")
            if len(tag_entity_split) != 2:
                continue
            tag = normalize_text(tag_entity_split[0].strip())
            entity = normalize_text(tag_entity_split[1].rstrip().lstrip())
            tags_entities.append((tag, entity))
        return tags_entities

    def compute_f1_metrics(true_positive, false_positive, false_negative):
        precision = float(true_positive) / float(true_positive + false_positive + 1e-13)
        recall = float(true_positive) / float(true_positive + false_negative + 1e-13)
        f1_measures = 2.0 * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measures

    unzipped_list = list(zip(*items, strict=False))
    targets = unzipped_list[0]
    predictions = unzipped_list[1]

    for target, pred in zip(targets, predictions, strict=False):
        gold_spans = tags_to_spans(target)
        predicted_spans = tags_to_spans(pred)

        for span in predicted_spans:
            if span in gold_spans:
                true_positives[span[0]] += 1
                gold_spans.remove(span)
            else:
                false_positives[span[0]] += 1
        for span in gold_spans:
            false_negatives[span[0]] += 1

    _, _, f1_measure = compute_f1_metrics(
        sum(true_positives.values()),
        sum(false_positives.values()),
        sum(false_negatives.values()),
    )
    return f1_measure

import re

from sallm.evaluation.lm_eval_tasks.masakhaner import (  # noqa: F401
    doc_to_target,
    span_f1_agg,
    transform_iob_tags,
    transform_text,
)


def format_span_misc(resps, docs):
    """lm-eval's ``format_span`` filter with MISC entities preserved.

    The stock filter keeps only PER/LOC/ORG/DATE (MasakhaNER's tag set),
    dropping NCHLT MISC predictions as misses.
    """
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
        "misc": "MISC",
    }

    def format_ner_text(text):
        text = text.lower()
        for key, value in label_dict.items():
            text = text.replace(key, value)
        text = "$".join(i for i in text.split("$$"))
        return text.rstrip("$")

    def format_named_entities(text):
        text = text.replace("\n", "$").strip()
        out = []
        for label, values in re.findall(r"\b(PER|LOC|ORG|DATE|MISC):\s*([^$]+)", text):
            out += [
                f"{label.lower()}: {v.strip()}"
                for v in values.split(",")
                if v.strip().lower() != "none"
            ]
        return " $ ".join(out)

    return [
        [format_named_entities(format_ner_text(r.lower())) for r in inst]
        for inst in resps
    ]

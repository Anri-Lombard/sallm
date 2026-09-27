"""Compare EOS-fixed Mamba reruns against the official outputs."""
import collections
import json
import re
import sys
from pathlib import Path

R = Path("/scratch/lmbanr001/masters/sallm/results/mamba_eos_fix_20260924")
FM = Path("/scratch/lmbanr001/masters/sallm/results/full_matrix_execution_20260916_v1/official")
LANG = {"tn": "tsn", "xh": "xho", "zu": "zul"}
PUNCT = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@.""-,`'


def normalize_text(s):
    s = re.sub("[" + PUNCT + "]+", " ", str(s))
    s = re.sub(re.compile(r"\b(a|an|the)\b", re.UNICODE), " ", s)
    s = re.sub(r"\s{3,}|\t", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def tags_to_spans(seq, delimiter="$$"):
    parts = [i.strip() for sub in seq.strip().split(delimiter) for i in sub.split("$") if i]
    parts = [i.strip() for v in parts for sub in v.split(". ") for i in sub.split(", ")]
    out = []
    for te in parts:
        sp = te.split(": ")
        if len(sp) == 2:
            out.append((normalize_text(sp[0].strip()), normalize_text(sp[1].strip())))
    return out


def is_loop(raw):
    toks = raw.split()
    if len(toks) < 12:
        return False
    c = collections.Counter(tuple(toks[i:i + 3]) for i in range(len(toks) - 2))
    return c.most_common(1)[0][1] >= 5


def squash(s):
    return re.sub(r"\s+", "", s)


def ner_stats(rows):
    by = collections.defaultdict(collections.Counter)
    for r in rows:
        l = LANG[r["task"].split("_")[2]]
        s = by[l]
        gold, pred = tags_to_spans(r["target"]), tags_to_spans(r["prediction"])
        s["rows"] += 1
        s["gold_spans"] += len(gold)
        s["pred_spans"] += len(pred)
        s["gold_empty"] += not gold
        s["empty_pred_on_empty_gold"] += (not gold) and (not pred)
        s["loop"] += is_loop(r["raw_response"])
        s["chars"] += len(r["raw_response"])
        g = list(gold)
        for sp in pred:
            if sp in g:
                s["tp"] += 1
                g.remove(sp)
            else:
                s["fp"] += 1
        s["fn"] += len(g)
    out = {}
    for l, s in sorted(by.items()):
        p = s["tp"] / (s["tp"] + s["fp"] + 1e-13)
        rc = s["tp"] / (s["tp"] + s["fn"] + 1e-13)
        f = 2 * p * rc / (p + rc + 1e-13)
        out[l] = {"F1": round(100 * f, 2), "P": round(100 * p, 2), "R": round(100 * rc, 2),
                  "pred_spans": s["pred_spans"], "gold_spans": s["gold_spans"],
                  "loop_pct": round(100 * s["loop"] / s["rows"], 1),
                  "no_entity_rows": s["gold_empty"],
                  "empty_output_on_no_entity_pct": round(100 * s["empty_pred_on_empty_gold"] / max(1, s["gold_empty"]), 1),
                  "mean_raw_chars": round(s["chars"] / s["rows"], 1)}
    return out


def ner_compare(name, old_path, new_path):
    old = json.load(open(old_path))
    new = json.load(open(new_path))
    ok = {(r["task"], r["doc_id"]): r for r in old["rows"]}
    prefix = same = n = 0
    ex = []
    for r in new["rows"]:
        o = ok[(r["task"], r["doc_id"])]
        assert o["prompt_hash"] == r["prompt_hash"] and o["target"] == r["target"]
        n += 1
        a, b = squash(r["raw_response"]), squash(o["raw_response"])
        same += a == b
        prefix += b.startswith(a)
        if not b.startswith(a) and len(ex) < 3:
            ex.append((r["task"], r["doc_id"], r["raw_response"][:150], o["raw_response"][:150]))
    print(f"\n=== {name}: {n} rows; new raw is a prefix of official raw: {prefix}/{n}; identical: {same}/{n}")
    for e in ex:
        print("   non-prefix example:", e)
    print("   official metrics:", {k: round(100 * v, 2) for k, v in old["reported_metrics"].items()})
    print("   new metrics     :", {k: round(100 * v, 2) for k, v in new["reported_metrics"].items()})
    so, sn = ner_stats(old["rows"]), ner_stats(new["rows"])
    for l in so:
        print(f"   {l} OLD {so[l]}\n       NEW {sn[l]}")
    print("   mean F1 old %.2f new %.2f" % (sum(v["F1"] for v in so.values()) / 3, sum(v["F1"] for v in sn.values()) / 3))
    print("   elapsed_seconds new:", round(new["elapsed_seconds"]))


def mgsm_samples(payload, group):
    call = payload["calls"][group]
    out = {}
    for task, samples in call["samples"].items():
        if not task.startswith("afrimgsm_"):
            continue
        out[task] = {"results": call["results"][task], "samples": {}}
        for s in samples:
            out[task]["samples"].setdefault(s["doc_id"], {})[s["filter"]] = s
    return out


def mgsm_compare(name, old, new, old_group, new_group):
    o, nw = mgsm_samples(old, old_group), mgsm_samples(new, new_group)
    print(f"\n=== {name}")
    for task in sorted(nw):
        ro, rn = o[task]["results"], nw[task]["results"]
        prefix = changed_pred = loops_o = loops_n = 0
        chars_o = chars_n = 0
        docs = nw[task]["samples"]
        for d, fs in docs.items():
            so, sn = o[task]["samples"][d]["flexible-extract"], fs["flexible-extract"]
            assert so["prompt_hash"] == sn["prompt_hash"]
            ro_txt, rn_txt = str(so["resps"][0][0]), str(sn["resps"][0][0])
            prefix += squash(ro_txt).startswith(squash(rn_txt))
            changed_pred += so["filtered_resps"] != sn["filtered_resps"]
            loops_o += is_loop(ro_txt)
            loops_n += is_loop(rn_txt)
            chars_o += len(ro_txt)
            chars_n += len(rn_txt)
        k = len(docs)
        print(f"   {task}: flexible EM old {100 * ro['exact_match,flexible-extract']:.1f} -> new {100 * rn['exact_match,flexible-extract']:.1f} | "
              f"remove_whitespace EM old {100 * ro['exact_match,remove_whitespace']:.1f} -> new {100 * rn['exact_match,remove_whitespace']:.1f} | "
              f"new raw prefix of old {prefix}/{k} | extracted answer changed {changed_pred}/{k} | loops {100 * loops_o / k:.0f}% -> {100 * loops_n / k:.0f}% | "
              f"mean chars {chars_o / k:.0f} -> {chars_n / k:.0f}")


which = sys.argv[1:]
if "general_ner" in which:
    ner_compare("General NER", "/scratch/lmbanr001/masters/sallm/results/general_sequence_official_test_20260917_v2/raw/ner/mamba2.json",
                R / "general_ner/raw/ner/mamba2.json")
if "base_ner" in which:
    ner_compare("Base NER (unit 5)", FM / "sequence/5.json", R / "base_ner/sequence/5.json")
if "base_mgsm" in which:
    mgsm_compare("Base AfriMGSM (unit 7)", json.load(open(FM / "prompt/7.json")), json.load(open(R / "base_afrimgsm/prompt/7.json")), "raw", "raw")
if "general_mgsm" in which:
    mgsm_compare("General AfriMGSM", json.load(open(R / "reference/general_prompt_lm_eval_mamba2_kombuys_official.json")),
                 json.load(open(R / "general_afrimgsm/raw/test/lm_eval/mamba2.json")), "closed_and_generation", "closed_and_generation")

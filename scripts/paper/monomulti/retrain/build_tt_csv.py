import csv, json, sys
from pathlib import Path
O = Path("/scratch/lmbanr001/masters/sallm/results/mzansilm_trainingtemplate_20260925/out")
H = "/scratch/lmbanr001/masters/sallm/results/historical_selected_adapter_recovery_20260917_hex_v3/train"
SM = "/scratch/lmbanr001/masters/sallm_recovery/mzansilm_sib_mono_20260916_v4r2/runs"
NT, ST, RT = "masakhane_news_classification/v5@cf5e68ff", "sib_topic_classification/v1@d663772a", "masakhane_named_entity_recognition/v5@bcbda3ae"
w = csv.writer(sys.stdout, lineterminator="\n")
w.writerow(["model", "task", "language", "regime", "metric", "score_points", "prompt_template", "adapter_path"])
for name, regime in [("mzansilm_news_mono_eng", "Mono"), ("mzansilm_news_mono_xho", "Mono"), ("mzansilm_news_multi", "Multi")]:
    d = json.loads((O / f"news/{name}.train.json").read_text())
    for lang, v in d["languages"].items():
        w.writerow(["MzansiLM", "News", lang, regime, "support_weighted_f1", f"{v['weighted_f1'] * 100:.4f}", NT, f"hex:{H}/{name}/final_adapter"])
for lang in ["afr", "eng", "nso", "sot", "xho", "zul"]:
    d = json.loads((O / f"sib/mono_{lang}/final_adapter.test.v1.json").read_text())
    w.writerow(["MzansiLM", "SIB", lang, "Mono", "support_weighted_f1", f"{d['languages'][lang]['f1'] * 100:.4f}", ST, f"hex:{SM}/{lang}/output/final_adapter"])
d = json.loads((O / "sib/multi/final_adapter.test.v1.json").read_text())
for lang, v in d["languages"].items():
    w.writerow(["MzansiLM", "SIB", lang, "Multi", "support_weighted_f1", f"{v['f1'] * 100:.4f}", ST, f"hex:{H}/mzansilm_sib_multi/final_adapter"])
for lang in ["tsn", "xho", "zul"]:
    d = json.loads((O / f"ner/mzansilm_ner_mono_{lang}.v5.json").read_text())
    (v,) = d["reported_metrics"].values()
    w.writerow(["MzansiLM", "NER", lang, "Mono", "span_f1", f"{v * 100:.4f}", RT, f"hex:{H}/mzansilm_ner_mono_{lang}/final_adapter"])

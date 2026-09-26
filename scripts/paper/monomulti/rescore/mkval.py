D="/scratch/alombard/sallm/results/monomulti_rescore_20260924"
PY="/scratch/alombard/sallm/.venv/bin/python"
V15="/scratch/alombard/sallm_snapshots/news-general-corrected-official-20260914-v15/src/main"
SIBRT="/scratch/alombard/sallm/snapshots/retained-sib-evaluator-20260913-v6/src/main"
A="/scratch/alombard/sallm/assets"
news_assets = {"mzansilm": f"{A}/mzansilm-news-general-validation-20260914-v11", "mamba2": "/scratch/alombard/sallm/results/mamba_sixfamily_priority_replacement_20260914_v4/assets/news",
               "xlstm": f"{A}/xlstm-news-general-validation-20260914-v11", "gdn": f"{A}/gdn-news-general-validation-20260914-v11"}
lines = []
for a, root in news_assets.items():
    out = f"{D}/validation/news_general_{a}_runtime.json"
    lines.append((f"val_news_{a}_runtime", out, f"PYTHONPATH={V15} {PY} {D}/scripts/score_news_trainpos.py --arch {a} --base {root}/base --adapter {root}/runtime_adapters/general --out {out}"))
sib = {"mzansilm": ("/scratch/alombard/sallm/results/retained_standardized_20260913_v1/bases/mzansilm", "/scratch/alombard/sallm/results/retained_standardized_20260913_v1/adapters/mzansilm/general"),
       "mamba2": ("/scratch/alombard/sallm/results/retained_standardized_20260913_v1/bases/mamba2", "/scratch/alombard/sallm/checkpoints/ft_mamba_125m_sa_general_sixfamily_tokenbalanced_r1/final_adapter"),
       "xlstm": ("/scratch/alombard/sallm/results/retained_standardized_20260913_v1/bases/xlstm", "/scratch/alombard/sallm/results/retained_standardized_20260913_v1/adapters/xlstm/general/final_adapter"),
       "gdn": ("/scratch/alombard/sallm/checkpoints/sallm-pure-gdn-125m/a10080-3epoch-20260802/final_model", "/scratch/alombard/sallm/results/retained_standardized_20260913_v1/adapters/gdn/general/final_adapter")}
for a, (b, ad) in sib.items():
    out = f"{D}/validation/sib_general_{a}.json"
    lines.append((f"val_sib_{a}", out, f"PYTHONPATH={SIBRT} {PY} {D}/scripts/score_sib_trainpos.py --arch {a} --base {b} --adapter {ad} --out {out}"))
for a, root in news_assets.items():
    out = f"{D}/raw/news/general_{a}.json"
    lines.append((f"news_general_{a}", out, f"PYTHONPATH={V15} {PY} {D}/scripts/score_news_trainpos.py --arch {a} --base {root}/base --adapter {root}/source_adapters/general --out {out}"))
print("\n".join("\t".join(l) for l in lines))

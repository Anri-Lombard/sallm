#!/usr/bin/env python3
"""Offline check of the early-stopping rule and the sanity thresholds (no cluster): python3 selfcheck.py"""
import rollout as R

src = open(R.HERE / "train_fft.py").read()
ns: dict = {}
exec(src[src.index("def early_stop_decision"):src.index("if CTRL:")], ns)
d = ns["early_stop_decision"]
ns = {"ARCH": "xlstm", "sys": type("S", (), {"argv": []})}
exec(src[src.index("KERNEL = "):src.index("_ctrl_run =")], ns)
ps, S = ns["pick_settings"], ns["SETTINGS"]
assert ps(None, None, False) == S["new"] and ps(None, None, True) == S["legacy"]  # a pre-27-Sep run resumed stays legacy
assert ps(None, S["new"], True) == S["new"] and ps("legacy", S["new"], False) == S["legacy"]
assert S["new"]["optimizer_impl"] == "adamw_torch_fused" and S["new"]["train_kernel"] == S["legacy"]["train_kernel"]
assert d([1, 2, 3], 3) == (3, False)
assert d([5, 4, 4, 4], 3) == (1, True)  # 3 epochs without a strictly greater score
assert d([5, 5, 5, 5], 3) == (1, True)  # ties are not improvement; the earlier epoch stays best
assert d([5, 4, 6, 6, 6, 6], 3) == (3, True)
assert R.expected_epochs("gdn", "sib", 10) == 7 and R.expected_epochs("xlstm", "t2x", 10) == 5 and R.expected_epochs("gdn", "news", 4) == 4
f = R.sanity_flags("sib", {"afr": 10.0}, {"afr": {"gold": ["a"] * 6 + ["b"] * 4, "pred": ["a"] * 9 + ["b"]}})["afr"]["flags"]
assert len(f) == 2
assert len(R.sanity_flags("t2x", {"xho": 30}, {"xho": {"pred": ["a b c d " * 4] * 4 + ["", "ok"]}})["xho"]["flags"]) == 1
assert len(R.sanity_flags("ner", {"xho": 0.0}, {"xho": {"pred": ["", "", "x", "y"]}})["xho"]["flags"]) == 2
print("selfcheck ok")

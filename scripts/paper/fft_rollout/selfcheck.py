#!/usr/bin/env python3
"""Offline check of the early-stopping rule and the sanity thresholds (no cluster): python3 selfcheck.py"""
import rollout as R

src = open(R.HERE / "train_fft.py").read()
ns: dict = {}
exec(src[src.index("def early_stop_decision"):src.index("if CTRL:")], ns)
d = ns["early_stop_decision"]
n = R.next_edge_lr
G = ["3e-5", "1e-4", "3e-4"]
assert n(G, "1e-4", 0) is None and n(G, "3e-4", 0) == "1e-3" and n(G, "3e-5", 0) == "1e-5"
assert n(G + ["1e-3"], "1e-3", 1) == "3e-3" and n(G + ["1e-3", "3e-3"], "3e-3", 2) is None  # at most 2 extra points
assert n(["1e-5"] + G, "1e-5", 1) == "3e-6" and n(["1e-5"] + G, "3e-5", 1) is None
ns = {"ARCH": "xlstm", "sys": type("S", (), {"argv": []})}
exec(src[src.index("KERNEL = "):src.index("_ctrl_run =")], ns)
ps, S = ns["pick_settings"], ns["SETTINGS"]
assert ps(None, None, False) == S["new"] and ps(None, None, True) == S["legacy"]  # a pre-27-Sep run resumed stays legacy
assert ps(None, S["new"], True) == S["new"] and ps("legacy", S["new"], False) == S["legacy"]
assert S["new"]["optimizer_impl"] == "adamw_torch_fused" and S["new"]["train_kernel"] == "tfla_padded128"
assert S["legacy"]["train_kernel"].startswith("chunkwise--native_autograd")
assert d([1, 2, 3], 3) == (3, False)
assert d([5, 4, 4, 4], 3) == (1, True)  # 3 epochs without a strictly greater score
assert d([5, 5, 5, 5], 3) == (1, True)  # ties are not improvement; the earlier epoch stays best
assert d([5, 4, 6, 6, 6, 6], 3) == (3, True)
assert R.expected_epochs("gdn", "sib", 10) == 7 and R.expected_epochs("xlstm", "t2x", 10) == 5 and R.expected_epochs("gdn", "news", 4) == 4
f = R.sanity_flags("sib", {"afr": 10.0}, {"afr": {"gold": ["a"] * 6 + ["b"] * 4, "pred": ["a"] * 9 + ["b"]}})["afr"]["flags"]
assert len(f) == 2
assert len(R.sanity_flags("t2x", {"xho": 30}, {"xho": {"pred": ["a b c d " * 4] * 4 + ["", "ok"]}})["xho"]["flags"]) == 1
assert len(R.sanity_flags("ner", {"xho": 0.0}, {"xho": {"pred": ["", "", "x", "y"], "gold": ["PER: a", "LOC: b", "x", ""]}})["xho"]["flags"]) == 2
print("selfcheck ok")
assert R.epochs_for(43637, "general") == 10 and R.epochs_for(43637) == 4 and R.epochs_for(4999) == 10  # Multitask amendment 28 Sep
assert R.epochs_for(7046, "intent") == 10 and R.epochs_for(24649, "afrihg") == 4
import val_subsample  # noqa: E402 - NCHLT languages agree between the rollout and the validation subsample (29 Sep)
assert all(set(R.FAMILIES[t]["langs"]) == set(val_subsample.SIZES[t]) for t in ("nchlt_ner", "nchlt_pos"))
_g = next(u for u in R.plan("mzansilm", False) if u["id"] == "train-general-general-s42")
assert _g["lr_from"] == ["news", "sib", "intent", "ner", "pos", "afrihg"], _g["lr_from"]  # NCHLT never gates/votes
# opt-in extras (30 Sep 2026): the default plan is unchanged; extras only append the units listed here
import hashlib  # noqa: E402

for _a in R.ARCHS:
    _ids = [u["id"] for u in R.plan(_a, False)]
    assert len(_ids) == 96 and hashlib.sha256("\n".join(_ids).encode()).hexdigest().startswith("b6607d576414f317"), _a  # pre-extras plan
    _all = R.plan(_a, False, extras=("posthoc", "seeds", "lrcheck"))
    assert [u["id"] for u in _all if not u.get("extra")] == _ids, _a  # extras append only, existing units untouched
    _new = [u["id"] for u in _all if u.get("extra")]
    _want = ([f"train-{f}-multi-s{s}" for f in ("news", "sib", "intent", "ner", "pos", "afrihg", "nchlt_ner") for s in (43, 44)]
             + ["posthoc-zs-sib-ext", "posthoc-zs-nchlt-ner", "posthoc-mt-nchlt-ner"]
             + (["train-general-general-lr1e-4-s42-lrcheck"] if _a in ("mamba2", "gdn") else []))  # lrcheck only mamba2/gdn
    assert _new == _want, (_a, _new)
    assert not any("nchlt_pos" in i for i in _new)
    _byid = {u["id"]: u for u in _all}
    for _c in ("collect", "collect-beam"):  # never a dependency of the main collectors
        assert not set(_byid[_c]["deps"]) & set(_new), _c
    assert all(d in _byid for u in _all for d in u["deps"])
    for _e in ("posthoc", "seeds", "lrcheck"):
        assert {u["id"] for u in R.plan(_a, False, extras=(_e,)) if u.get("extra")} <= set(_new)
_lc = next(u for u in R.plan("gdn", False, extras=("lrcheck",)) if u.get("variant") == "lrcheck")
assert R.run_id(_lc, "1e-4") == "general-general-lr1e-4-s42-lrcheck" != R.run_id({**_lc, "variant": None}, "1e-4")  # own run dir
assert _lc["lr"] == "1e-4" and "lr_from" not in _lc  # fixed LR: transferred_lr() (LR_TRANSFER.json) is never called
assert next(u for u in R.plan("gdn", False, extras=("seeds",)) if u["id"] == "train-sib-multi-s43")["lr"] is None  # selected_lr
for _a in R.ARCHS:  # seeds_mt / seeds_mono (30 Sep): Multitask s43/s44 + every Mono cell (nchlt_pos excluded) x s43/s44
    _base = [u["id"] for u in R.plan(_a, False, extras=("posthoc", "seeds", "lrcheck"))]
    _full = R.plan(_a, False, extras=("posthoc", "seeds", "lrcheck", "seeds_mt", "seeds_mono"))
    assert set(_base) <= {u["id"] for u in _full} and len({u["id"] for u in _full}) == len(_full)  # ids unique
    _add = [u for u in _full if u["id"] not in set(_base)]
    _mt = [u for u in _add if u["family"] == "general"]
    assert sorted(u["id"] for u in _mt) == ["train-general-general-s43", "train-general-general-s44"]
    _mono = [u for u in _add if u["family"] != "general"]
    assert all(u["regime"] == "mono" and u["seed"] in (43, 44) and u["family"] != "nchlt_pos" for u in _mono)
    assert len(_mono) == 2 * 29, (_a, len(_mono))  # 29 Mono cells outside nchlt_pos (t2x has its own seeds)
for _a in R.ARCHS:  # beam_seeds (30 Sep): beam units for every AfriHG and Multitask seed run (+ the LR-check Multitask run)
    _all = R.plan(_a, False, extras=R.EXTRAS)
    _ids = {u["id"] for u in _all}
    _b = [u for u in _all if u.get("extra") == "beam_seeds"]
    assert all(u["src"] in _ids and u["deps"] == [u["src"]] for u in _b), _a
    assert len(_b) == 6 + 2 + (_a in R.LRCHECK_ARCHS), (_a, len(_b))
    assert len(_ids) == len(_all)
print("selfcheck extras ok")

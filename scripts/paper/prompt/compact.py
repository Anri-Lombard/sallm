import json, sys, re
out = {}
for path in sys.argv[2:]:
    d = json.load(open(path))
    for task, S in d["samples"].items():
        base = re.sub(r"_trainpos$", "", task)
        rows = []
        for s in S:
            if s.get("filter", "none") != "none":
                continue
            lls = [r[0][0] if isinstance(r[0], list) else r[0] for r in s["resps"]]
            rows.append([s["doc_id"], s["target"] if not isinstance(s["target"], str) else s["target"], lls, s["arguments"][0][0][-60:]])
        out[base] = {"rows": rows, "results": d["results"].get(task), "choices": d["configs"][task]["doc_to_choice"]}
json.dump(out, open(sys.argv[1], "w"))

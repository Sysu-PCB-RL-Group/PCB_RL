import json, glob, os
for fn in sorted(glob.glob(r"data\expert_transitions\*_expert.json")):
    rs = [t["reward"] for t in json.load(open(fn, encoding="utf-8"))]
    top = sorted(round(r) for r in rs)[-5:]
    print(os.path.basename(fn))
    print("  n=%d min=%.2f max=%.2f mean=%.2f neg=%d top5=%s" % (
        len(rs), min(rs), max(rs), sum(rs) / len(rs),
        sum(1 for r in rs if r < 0), top))

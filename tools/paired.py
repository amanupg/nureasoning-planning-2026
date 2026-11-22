import json, sys, collections, numpy as np
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); k = sys.argv[3] if len(sys.argv) > 3 else "edge"
names = sorted(set(a) & set(b)); log = lambda c: "_".join(c.split("_")[:2])
by = collections.defaultdict(list)
for c in names: by[log(c)].append(a[c][k] - b[c][k])
L = sorted(by); s = np.array([sum(by[l]) for l in L]); n = np.array([len(by[l]) for l in L])
rs = np.random.RandomState(0); bs = []
for _ in range(10000):
    i = rs.randint(0, len(L), len(L)); bs.append(s[i].sum() / n[i].sum())
print(f"{k}: n={len(names)} A {np.mean([a[c][k] for c in names]):.4f} B {np.mean([b[c][k] for c in names]):.4f} "
      f"diff {s.sum()/n.sum():+.4f} [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]")

"""
verify_dataset.py — independent audit of dataset_clean/.

Re-reads every file from disk and re-derives hashes itself; it does not trust
manifest.csv or prepare_report.json. Exits non-zero if any check fails.
"""
from __future__ import annotations
import os, sys, hashlib, json
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

DST = "dataset_clean"
SPLITS = ("train", "valid", "test", "holdout_unseen_gen")
HAMMING_CAND, CORR_CONFIRM = 10, 0.95
failures = []


def load():
    items = []
    for split in SPLITS:
        for cls in ("real", "fake"):
            d = os.path.join(DST, split, cls)
            for fn in sorted(os.listdir(d)):
                p = os.path.join(d, fn)
                raw = open(p, "rb").read()
                im = Image.open(p)
                g = im.convert("L")
                s = np.asarray(g.resize((9, 8), Image.LANCZOS), dtype=np.int16)
                bits = (s[:, 1:] > s[:, :-1]).flatten()
                h = 0
                for b in bits:
                    h = (h << 1) | int(b)
                t = np.asarray(g.resize((32, 32), Image.LANCZOS), dtype=np.float32)
                items.append({"path": p, "split": split, "cls": cls,
                              "group": fn.split("__")[0], "size": im.size,
                              "md5": hashlib.md5(raw).hexdigest(), "dhash": h,
                              "thumb": (t - t.mean()) / (t.std() + 1e-6)})
    return items


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        failures.append(label)


items = load()
print(f"\nloaded {len(items)} images from {DST}/\n")

print("--- integrity ---")
sizes = Counter(i["size"] for i in items)
check("all images 299x299", list(sizes) == [(299, 299)], str(dict(sizes)))
md5s = Counter(i["md5"] for i in items)
check("no byte-identical duplicates anywhere",
      max(md5s.values()) == 1, f"max copies of one file = {max(md5s.values())}")
lab = defaultdict(set)
for i in items:
    lab[i["md5"]].add(i["cls"])
check("no image labeled both real and fake",
      all(len(v) == 1 for v in lab.values()))

print("\n--- leakage: is any eval image also in train? ---")
for cls in ("real", "fake"):
    tr = [i for i in items if i["split"] == "train" and i["cls"] == cls]
    T = np.unpackbits(np.array([i["dhash"] for i in tr], dtype=np.uint64)
                      .view(np.uint8).reshape(-1, 8)[:, ::-1], axis=1).astype(np.float32)
    ts = T.sum(1, keepdims=True).T
    for split in ("valid", "test", "holdout_unseen_gen"):
        q = [i for i in items if i["split"] == split and i["cls"] == cls]
        if not q:
            continue
        Q = np.unpackbits(np.array([i["dhash"] for i in q], dtype=np.uint64)
                          .view(np.uint8).reshape(-1, 8)[:, ::-1], axis=1).astype(np.float32)
        D = Q.sum(1, keepdims=True) + ts - 2.0 * (Q @ T.T)
        hits = []
        for k in range(len(q)):
            cand = np.where(D[k] <= HAMMING_CAND)[0]
            for j in cand[:80]:
                if float((q[k]["thumb"] * tr[j]["thumb"]).mean()) > CORR_CONFIRM:
                    hits.append((q[k]["path"], tr[j]["path"])); break
        check(f"{split}/{cls} clean of train content",
              len(hits) == 0, f"{len(hits)}/{len(q)} contaminated")
        for a, b in hits[:3]:
            print(f"        {a}\n          ~= {b}")

print("\n--- generator mix: does valid match test? ---")
for cls in ("real", "fake"):
    mixes = {}
    for split in ("train", "valid", "test"):
        sub = [i for i in items if i["split"] == split and i["cls"] == cls]
        c = Counter(i["group"] for i in sub)
        mixes[split] = {k: v / len(sub) for k, v in c.items()}
    keys = sorted(set().union(*[m.keys() for m in mixes.values()]))
    print(f"  {cls}:")
    print(f"    {'group':22s} {'train':>8s} {'valid':>8s} {'test':>8s}")
    worst = 0.0
    for k in keys:
        tv, vv, sv = (mixes[s].get(k, 0) for s in ("train", "valid", "test"))
        worst = max(worst, abs(vv - sv))
        print(f"    {k:22s} {tv:7.1%} {vv:7.1%} {sv:7.1%}")
    check(f"{cls}: valid/test mix aligned (max gap < 3pp)",
          worst < 0.03, f"max gap {worst:.1%}")

print("\n--- balance ---")
for split in SPLITS:
    r = sum(1 for i in items if i["split"] == split and i["cls"] == "real")
    f = sum(1 for i in items if i["split"] == split and i["cls"] == "fake")
    print(f"  {split:20s} real {r:5d}  fake {f:5d}   fake share {f/(r+f):5.1%}")

print("\n--- holdout is genuinely unseen ---")
hg = {i["group"] for i in items if i["split"] == "holdout_unseen_gen" and i["cls"] == "fake"}
tg = {i["group"] for i in items if i["split"] == "train" and i["cls"] == "fake"}
check("holdout generator absent from train", not (hg & tg), f"holdout={hg}")

print("\n--- residual sharpness gap (the shortcut) ---")
for split in ("train", "valid", "test", "holdout_unseen_gen"):
    v = {}
    for cls in ("real", "fake"):
        ps = [i["path"] for i in items if i["split"] == split and i["cls"] == cls]
        st = max(1, len(ps) // 300)
        vals = []
        for p in ps[::st]:
            g = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
            lap = (-4*g[1:-1,1:-1]+g[:-2,1:-1]+g[2:,1:-1]+g[1:-1,:-2]+g[1:-1,2:])
            vals.append(float(lap.var()))
        v[cls] = float(np.mean(vals))
    print(f"  {split:20s} real {v['real']:7.0f}   fake {v['fake']:7.0f}   "
          f"ratio {v['fake']/v['real']:.2f}x")

print("\n" + "=" * 62)
if failures:
    print(f"{len(failures)} CHECK(S) FAILED:")
    for f in failures:
        print("   -", f)
    sys.exit(1)
print("ALL CHECKS PASSED")
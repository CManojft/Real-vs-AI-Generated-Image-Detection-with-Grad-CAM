"""
prepare_dataset.py — rebuild dataset_xception into a leak-free, source-stratified dataset.

Fixes the four flaws found in the audit of the original split:
  1. train/eval contamination  -> global near-duplicate clustering; a cluster never spans splits
  2. valid/test generator mismatch -> splits stratified per (class, source-group)
  3. sharpness shortcut        -> reported here, mitigated by augmentation at train time
  4. no generalization test    -> one generator family held out entirely

Reads  : dataset_xception/{train,valid,test}/{real,fake}/*.jpg   (never modified)
Writes : dataset_clean/{train,valid,test}/{real,fake}/
         dataset_clean/holdout_unseen_gen/{real,fake}/
         reports/manifest.csv, reports/prepare_report.json
"""
from __future__ import annotations
import os, re, json, csv, hashlib, shutil, random, sys
from collections import Counter, defaultdict
import numpy as np
from PIL import Image

SRC       = "dataset_xception"
DST       = "dataset_clean"
REPORTS   = "reports"
SEED      = 1337
RATIOS    = {"train": 0.70, "valid": 0.15, "test": 0.15}
HOLDOUT_GROUP = "gen_studio_batch"   # one generator family, excluded from train/valid/test
HOLDOUT_REALS = 500                  # matched real images so the holdout is balanced
HAMMING_CAND  = 10                   # dHash candidate radius
CORR_CONFIRM  = 0.95                 # pixel-correlation threshold that confirms a duplicate

random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------- source groups
def source_group(fn: str, cls: str) -> str:
    """Map a filename to the generator / capture source it came from."""
    b = os.path.splitext(fn)[0]
    if cls == "real":
        if b.startswith("lfw_"):                    return "real_lfw"
        if b.startswith("tran_real"):               return "real_pool"
        if re.fullmatch(r"\d{5,6}", b):             return "real_celeba"
        return "real_other"
    if b.startswith("Face"):                        return "gen_studio_batch"
    if re.fullmatch(r"(man|woman|men|women)[ _]?\(?\d+\)?", b):  return "gen_stock_portrait"
    if re.fullmatch(r"[0-9A-Z]{10}", b):            return "gan_stylegan_web"
    if re.fullmatch(r"\d+_x_[\dx_]+", b):           return "gan_stylegan_grid"
    if re.fullmatch(r"\d{3}_\d{1,2}", b) or re.fullmatch(r"0_01 \(\d+\)", b) \
       or re.fullmatch(r"\d+ \(\d+\)", b) or re.fullmatch(r"\d{1,4}", b):
        return "swap_video_frames"
    if re.fullmatch(r"\d{5,6}", b):                 return "gan_manipulated"
    return "fake_other"


# ---------------------------------------------------------------- inventory
def dhash(img_gray_9x8: np.ndarray) -> int:
    bits = (img_gray_9x8[:, 1:] > img_gray_9x8[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def build_inventory():
    items = []
    for split in ("train", "valid", "test"):
        for cls in ("real", "fake"):
            d = os.path.join(SRC, split, cls)
            for fn in sorted(os.listdir(d)):
                p = os.path.join(d, fn)
                raw = open(p, "rb").read()
                im = Image.open(p).convert("L")
                small = np.asarray(im.resize((9, 8), Image.LANCZOS), dtype=np.int16)
                thumb = np.asarray(im.resize((32, 32), Image.LANCZOS), dtype=np.float32)
                items.append({
                    "path": p, "fn": fn, "cls": cls, "orig_split": split,
                    "group": source_group(fn, cls),
                    "md5": hashlib.md5(raw).hexdigest(),
                    "dhash": dhash(small),
                    "thumb": (thumb - thumb.mean()) / (thumb.std() + 1e-6),
                    "bytes": len(raw),
                })
    return items


# ---------------------------------------------------------------- dedup + cluster
def drop_exact_duplicates(items):
    """Keep one file per md5. Prefer the copy whose name carries real provenance."""
    def rank(it):
        return (0 if it["fn"].startswith("lfw_") else
                1 if it["group"] in ("real_celeba",) else 2, it["fn"])
    by_md5 = defaultdict(list)
    for it in items:
        by_md5[it["md5"]].append(it)
    kept, dropped = [], []
    for _, group in by_md5.items():
        group.sort(key=rank)
        kept.append(group[0])
        dropped.extend(group[1:])
    return kept, dropped


def cluster_near_duplicates(items):
    """Union-find over pixel-confirmed near-duplicates. Clusters never cross class."""
    parent = list(range(len(items)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cls in ("real", "fake"):
        idx = [i for i, it in enumerate(items) if it["cls"] == cls]
        H = np.array([items[i]["dhash"] for i in idx], dtype=np.uint64)
        bits = np.unpackbits(H.view(np.uint8).reshape(-1, 8)[:, ::-1], axis=1).astype(np.float32)
        s = bits.sum(1, keepdims=True)
        D = s + s.T - 2.0 * (bits @ bits.T)
        np.fill_diagonal(D, 999)
        cand = np.argwhere(D <= HAMMING_CAND)
        confirmed = 0
        for a, b in cand:
            if a >= b:
                continue
            ia, ib = idx[a], idx[b]
            if find(ia) == find(ib):
                continue
            corr = float((items[ia]["thumb"] * items[ib]["thumb"]).mean())
            if corr > CORR_CONFIRM:
                union(ia, ib)
                confirmed += 1
        print(f"    {cls}: {len(cand)//2} candidate pairs -> {confirmed} confirmed merges")

    for i, it in enumerate(items):
        it["cluster"] = find(i)
    return items


# ---------------------------------------------------------------- split assignment
def assign_splits(items):
    """
    Assign whole clusters to splits, stratified within each (class, source-group).
    A cluster is atomic: every member lands in the same split, so no duplicate can
    straddle a split boundary.
    """
    clusters = defaultdict(list)
    for it in items:
        clusters[it["cluster"]].append(it)

    strata = defaultdict(list)
    for cid, members in clusters.items():
        cls = members[0]["cls"]
        group = Counter(m["group"] for m in members).most_common(1)[0][0]
        strata[(cls, group)].append((cid, len(members), group))

    assignment = {}
    for key in sorted(strata):
        bucket = sorted(strata[key], key=lambda t: (-t[1], t[0]))
        rng = random.Random(SEED + hash(key) % 10_000)
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(round(n * RATIOS["train"]))
        n_valid = int(round(n * RATIOS["valid"]))
        for i, (cid, _, _) in enumerate(bucket):
            assignment[cid] = ("train" if i < n_train else
                               "valid" if i < n_train + n_valid else "test")
    return assignment, clusters


def carve_holdout(items, clusters):
    """
    Pull one generator family out entirely -> the unseen-generator test.
    Pair it with real images (whole clusters) so the holdout is class-balanced.
    """
    holdout = [it for it in items if it["group"] == HOLDOUT_GROUP]
    held_clusters = {it["cluster"] for it in holdout}

    real_clusters = sorted({it["cluster"] for it in items if it["cls"] == "real"})
    rng = random.Random(SEED + 99)
    rng.shuffle(real_clusters)
    picked, n = [], 0
    for cid in real_clusters:
        members = [m for m in clusters[cid] if m["cls"] == "real"]
        if n + len(members) > HOLDOUT_REALS:
            continue
        picked.append(cid); n += len(members)
        if n >= HOLDOUT_REALS:
            break
    for cid in picked:
        holdout.extend(m for m in clusters[cid] if m["cls"] == "real")
    held_clusters |= set(picked)

    remainder = [it for it in items if it["cluster"] not in held_clusters]
    return holdout, remainder, held_clusters


def collapse_eval(items_for_split):
    """
    In valid/test keep ONE image per cluster, so repeated video frames of the same
    face cannot dominate the metrics. train keeps every member (free augmentation).
    """
    out, seen = [], set()
    for it in sorted(items_for_split, key=lambda x: x["fn"]):
        if it["split"] in ("valid", "test"):
            if it["cluster"] in seen:
                it["dropped_reason"] = "eval_cluster_collapse"
                continue
            seen.add(it["cluster"])
        out.append(it)
    return out


# ---------------------------------------------------------------- materialize
def materialize(final_items):
    if os.path.exists(DST):
        shutil.rmtree(DST)
    for split in ("train", "valid", "test", "holdout_unseen_gen"):
        for cls in ("real", "fake"):
            os.makedirs(os.path.join(DST, split, cls), exist_ok=True)

    linked = copied = 0
    for it in final_items:
        # prefix with the source group so provenance survives in the filename
        safe = re.sub(r"[^A-Za-z0-9._()-]", "_", it["fn"])
        out = os.path.join(DST, it["split"], it["cls"], f"{it['group']}__{safe}")
        k = 1
        while os.path.exists(out):
            out = os.path.join(DST, it["split"], it["cls"],
                               f"{it['group']}__{k}_{safe}")
            k += 1
        try:
            os.link(it["path"], out); linked += 1
        except OSError:
            shutil.copy2(it["path"], out); copied += 1
        it["dst"] = out
    return linked, copied


# ---------------------------------------------------------------- reporting
def sharpness(paths, sample=350):
    step = max(1, len(paths) // sample)
    vals = []
    for p in paths[::step]:
        g = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
        lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1]
               + g[1:-1, :-2] + g[1:-1, 2:])
        vals.append(float(lap.var()))
    return float(np.mean(vals)) if vals else 0.0


def write_reports(final_items, dropped_exact, held_clusters, stats):
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["new_split", "cls", "source_group", "cluster_id",
                    "orig_split", "orig_path", "new_path", "md5"])
        for it in sorted(final_items, key=lambda x: (x["split"], x["cls"], x["fn"])):
            w.writerow([it["split"], it["cls"], it["group"], it["cluster"],
                        it["orig_split"], it["path"], it.get("dst", ""), it["md5"]])
    with open(os.path.join(REPORTS, "prepare_report.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)


def main():
    print("=" * 74)
    print("STEP 1  inventory")
    items = build_inventory()
    print(f"    {len(items)} images read from {SRC}/")

    print("STEP 2  drop byte-identical duplicates")
    items, dropped_exact = drop_exact_duplicates(items)
    print(f"    removed {len(dropped_exact)} exact copies -> {len(items)} remain")

    print("STEP 3  cluster near-duplicates (pixel-confirmed)")
    items = cluster_near_duplicates(items)
    clusters = defaultdict(list)
    for it in items:
        clusters[it["cluster"]].append(it)
    print(f"    {len(items)} images -> {len(clusters)} independent clusters")

    print(f"STEP 4  carve unseen-generator holdout ({HOLDOUT_GROUP})")
    holdout, remainder, held = carve_holdout(items, clusters)
    for it in holdout:
        it["split"] = "holdout_unseen_gen"
    hc = Counter(it["cls"] for it in holdout)
    print(f"    holdout: {hc['fake']} fake ({HOLDOUT_GROUP}) + {hc['real']} real")

    print("STEP 5  stratified, cluster-atomic split of the remainder")
    assignment, _ = assign_splits(remainder)
    for it in remainder:
        it["split"] = assignment[it["cluster"]]

    print("STEP 6  collapse duplicate clusters inside valid/test")
    before = len(remainder)
    remainder = collapse_eval(remainder)
    print(f"    {before - len(remainder)} redundant eval images removed")

    final_items = remainder + holdout
    print("STEP 7  materialize dataset_clean/")
    linked, copied = materialize(final_items)
    print(f"    {linked} hardlinked, {copied} copied")

    print("STEP 8  reports")
    stats = {"seed": SEED, "ratios": RATIOS, "holdout_group": HOLDOUT_GROUP,
             "source_images": len(items) + len(dropped_exact),
             "exact_duplicates_removed": len(dropped_exact),
             "clusters": len(clusters), "final_images": len(final_items),
             "counts": {}, "group_mix": {}, "sharpness": {}}
    for split in ("train", "valid", "test", "holdout_unseen_gen"):
        stats["counts"][split] = {}
        for cls in ("real", "fake"):
            sub = [i for i in final_items if i["split"] == split and i["cls"] == cls]
            stats["counts"][split][cls] = len(sub)
            stats["group_mix"].setdefault(split, {})[cls] = dict(
                Counter(i["group"] for i in sub).most_common())
            stats["sharpness"].setdefault(split, {})[cls] = round(
                sharpness([i["path"] for i in sub]), 1)
    n_r = stats["counts"]["train"]["real"]; n_f = stats["counts"]["train"]["fake"]
    tot = n_r + n_f
    stats["class_weight"] = {"real": round(tot / (2 * n_r), 4),
                             "fake": round(tot / (2 * n_f), 4)}
    write_reports(final_items, dropped_exact, held, stats)
    print(json.dumps(stats["counts"], indent=2))
    print(f"    class_weight for training: {stats['class_weight']}")
    print("=" * 74)
    print("done -> dataset_clean/   reports/manifest.csv   reports/prepare_report.json")


if __name__ == "__main__":
    main()
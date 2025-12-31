import os, json
import numpy as np
import pandas as pd

IN_PATH = r"data\processed\processed_table.parquet"
OUT_DIR = r"data\processed\clients"
LABEL_COL = "label"
CLIENT_COL = "sample_id"
SPLIT = (0.8, 0.1, 0.1)

os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_parquet(IN_PATH)

def stratified_split(df0, label_col, seed=42, split=SPLIT):
    rng = np.random.default_rng(seed)
    tr_parts, va_parts, te_parts = [], [], []
    for _, sub in df0.groupby(label_col):
        idx = sub.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * split[0])
        n_va = int(n * split[1])
        tr_parts.append(df0.loc[idx[:n_tr]])
        va_parts.append(df0.loc[idx[n_tr:n_tr+n_va]])
        te_parts.append(df0.loc[idx[n_tr+n_va:]])
    return pd.concat(tr_parts), pd.concat(va_parts), pd.concat(te_parts)

global_meta = {}

for i, (client_id, cdf) in enumerate(df.groupby(CLIENT_COL), start=1):
    client_name = f"client_{i:02d}"
    cdir = os.path.join(OUT_DIR, client_name)
    os.makedirs(cdir, exist_ok=True)

    tr, va, te = stratified_split(cdf, LABEL_COL, seed=42)

    tr.to_parquet(os.path.join(cdir, "train.parquet"), index=False)
    va.to_parquet(os.path.join(cdir, "val.parquet"), index=False)
    te.to_parquet(os.path.join(cdir, "test.parquet"), index=False)

    meta = {
        "client_name": client_name,
        "group_value": str(client_id),
        "split_axis": CLIENT_COL,
        "n_total": int(len(cdf)),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "label_counts_total": cdf[LABEL_COL].value_counts().to_dict(),
    }
    with open(os.path.join(cdir, "client_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    global_meta[client_name] = meta

with open(r"data\processed\global_metadata.json", "w") as f:
    json.dump(global_meta, f, indent=2)

print(f"Saved {len(global_meta)} clients to {OUT_DIR}")

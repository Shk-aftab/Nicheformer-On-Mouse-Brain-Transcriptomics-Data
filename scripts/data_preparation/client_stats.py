import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLIENTS_DIR = os.path.join("data", "processed", "clients")
OUT_DIR = os.path.join("md", "figures")
SUMMARY_MD = os.path.join("md", "client_stats_summary.md")

os.makedirs(OUT_DIR, exist_ok=True)

def safe_read_meta(meta_path):
    with open(meta_path, "r") as f:
        return json.load(f)

def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def js_divergence(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)
    def kl(a, b):
        a = a[a > 0]
        b = b[:len(a)]
        return float((a * np.log(a / (b + 1e-12))).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

# Discover clients
client_dirs = sorted([d for d in glob.glob(os.path.join(CLIENTS_DIR, "client_*")) if os.path.isdir(d)])
if not client_dirs:
    raise RuntimeError(f"No clients found in {CLIENTS_DIR}")

# Load per-client metadata and compute summary table
rows = []
all_label_ids = set()

client_label_counts = {}  # client -> dict(label->count)

for cdir in client_dirs:
    cname = os.path.basename(cdir)
    meta_path = os.path.join(cdir, "client_meta.json")
    meta = safe_read_meta(meta_path)

    counts = meta.get("label_counts_total", {})
    # keys might be str(label_id) from value_counts().to_dict()
    counts = {int(k): int(v) for k, v in counts.items()}
    client_label_counts[cname] = counts
    all_label_ids.update(counts.keys())

    n_total = meta["n_total"]
    n_train = meta["n_train"]
    n_val = meta["n_val"]
    n_test = meta["n_test"]

    # Simple imbalance metrics
    max_frac = max(counts.values()) / max(1, sum(counts.values()))
    min_count = min(counts.values()) if counts else 0
    n_classes = len(counts)

    rows.append({
        "client": cname,
        "group_value": meta.get("group_value", ""),
        "n_total": n_total,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
        "n_classes_present": n_classes,
        "max_label_fraction": max_frac,
        "min_label_count": min_count,
    })

summary_df = pd.DataFrame(rows).sort_values("client")
summary_csv = os.path.join(OUT_DIR, "client_summary.csv")
summary_df.to_csv(summary_csv, index=False)

# Build label distribution matrix (clients x labels)
label_ids = sorted(all_label_ids)
dist_mat = []
for cname in summary_df["client"]:
    counts = client_label_counts[cname]
    vec = np.array([counts.get(l, 0) for l in label_ids], dtype=float)
    dist_mat.append(vec)
dist_mat = np.vstack(dist_mat)

# Normalize to probabilities
prob_mat = dist_mat / (dist_mat.sum(axis=1, keepdims=True) + 1e-12)

# Global distribution
global_counts = dist_mat.sum(axis=0)
global_prob = global_counts / (global_counts.sum() + 1e-12)

# Compute non-IID metrics per client vs global
non_iid_rows = []
for i, cname in enumerate(summary_df["client"]):
    p = prob_mat[i]
    ent = entropy(p)
    jsd = js_divergence(p, global_prob)
    non_iid_rows.append({
        "client": cname,
        "entropy": ent,
        "js_divergence_to_global": jsd,
    })

non_iid_df = pd.DataFrame(non_iid_rows).merge(summary_df, on="client")
non_iid_csv = os.path.join(OUT_DIR, "client_noniid_metrics.csv")
non_iid_df.to_csv(non_iid_csv, index=False)

# ---------- Plots ----------
# 1) Client sizes
plt.figure()
plt.bar(summary_df["client"], summary_df["n_total"])
plt.title("Client sizes (n_total)")
plt.ylabel("Number of samples")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "client_sizes.png"), dpi=200)
plt.close()

# 2) Max label fraction per client (imbalance)
plt.figure()
plt.bar(summary_df["client"], summary_df["max_label_fraction"])
plt.title("Imbalance per client (max label fraction)")
plt.ylabel("Max label fraction")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "client_imbalance_max_fraction.png"), dpi=200)
plt.close()

# 3) JSD to global (non-IID severity)
plt.figure()
plt.bar(non_iid_df["client"], non_iid_df["js_divergence_to_global"])
plt.title("Non-IID severity (Jensen-Shannon divergence to global)")
plt.ylabel("JSD (nats)")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "client_jsd_to_global.png"), dpi=200)
plt.close()

# 4) Top-k label distribution per client (stacked-ish table as CSV)
# Save label probability table (wide) for reference
prob_df = pd.DataFrame(prob_mat, columns=[f"label_{l}" for l in label_ids])
prob_df.insert(0, "client", list(summary_df["client"]))
prob_df.to_csv(os.path.join(OUT_DIR, "client_label_probabilities.csv"), index=False)

# ---------- Write summary markdown ----------
md_lines = []
md_lines.append("# Client Statistics & Diagnostics (Milestone 2)\n")
md_lines.append("## What this analysis covers\n")
md_lines.append("- Per-client dataset sizes and split sizes\n")
md_lines.append("- Label distribution imbalance within each client\n")
md_lines.append("- Non-IID severity vs global distribution using Jensen–Shannon divergence\n")
md_lines.append("\n")

md_lines.append("## Key outputs\n")
md_lines.append(f"- `{summary_csv}`\n")
md_lines.append(f"- `{non_iid_csv}`\n")
md_lines.append(f"- `{os.path.join(OUT_DIR, 'client_label_probabilities.csv')}`\n")
md_lines.append("\n")

md_lines.append("## Figures\n")
md_lines.append(f"- `{os.path.join(OUT_DIR, 'client_sizes.png')}`\n")
md_lines.append(f"- `{os.path.join(OUT_DIR, 'client_imbalance_max_fraction.png')}`\n")
md_lines.append(f"- `{os.path.join(OUT_DIR, 'client_jsd_to_global.png')}`\n")
md_lines.append("\n")

md_lines.append("## Summary table (per client)\n")
md_lines.append(summary_df.to_markdown(index=False))
md_lines.append("\n\n")

md_lines.append("## Non-IID metrics (per client)\n")
md_lines.append(non_iid_df[["client","group_value","n_total","n_classes_present","max_label_fraction","entropy","js_divergence_to_global"]].to_markdown(index=False))
md_lines.append("\n")

with open(SUMMARY_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print("Wrote:")
print(" -", summary_csv)
print(" -", non_iid_csv)
print(" -", SUMMARY_MD)
print(" - Figures in:", OUT_DIR)

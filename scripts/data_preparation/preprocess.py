import os, json
import scanpy as sc
import pandas as pd

RAW_PATH = "data/raw/10xgenomics_xenium_mouse_brain_replicates.h5ad"
OUT_DIR = "data/processed"
CLIENT_COL = "library_key"   # for federated split later
X_COL, Y_COL = "x", "y"

os.makedirs(OUT_DIR, exist_ok=True)

print("Loading data...")
adata = sc.read_h5ad(RAW_PATH)

# -----------------------------
# 1) Basic filtering (safe)
# -----------------------------
sc.pp.filter_cells(adata, min_counts=10)

# -----------------------------
# 2) Normalize + log1p
# -----------------------------
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# -----------------------------
# 3) PCA + neighborhood graph
# -----------------------------
sc.pp.pca(adata, n_comps=30)
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

# -----------------------------
# 4) Leiden clustering (LABELS)
# -----------------------------
sc.tl.leiden(adata, resolution=0.5, key_added="cluster")

labels = adata.obs["cluster"].astype(str)
uniq = sorted(labels.unique())

label_map = {lab: i for i, lab in enumerate(uniq)}
with open(os.path.join(OUT_DIR, "label_map.json"), "w") as f:
    json.dump(label_map, f, indent=2)

encoded_labels = labels.map(label_map).astype("int32")

print(f"Created {len(uniq)} clusters")

# -----------------------------
# 5) Save gene schema
# -----------------------------
gene_names = list(adata.var_names)
with open(os.path.join(OUT_DIR, "genes.txt"), "w") as f:
    f.write("\n".join(gene_names))

# -----------------------------
# 6) Build processed table
# -----------------------------
X = adata.X
try:
    import scipy.sparse as sp
    if sp.issparse(X):
        X = X.toarray()
except Exception:
    pass

expr_df = pd.DataFrame(X, columns=gene_names)

meta_df = pd.DataFrame({
    "id": adata.obs_names.astype(str),
    "sample_id": adata.obs[CLIENT_COL].astype(str).values,
    "x": adata.obs[X_COL].astype("float32").values,
    "y": adata.obs[Y_COL].astype("float32").values,
    "label": encoded_labels.values,
})

out = pd.concat([meta_df, expr_df], axis=1)
out_path = os.path.join(OUT_DIR, "processed_table.parquet")
out.to_parquet(out_path, index=False)

print("Saved:", out_path)
print("Rows:", len(out), "Genes:", len(gene_names))

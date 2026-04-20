# Databricks notebook source
# MAGIC %md
# MAGIC # TravelFraudGraph — Notebook 4: HuggingFace Upload & Croissant Metadata
# MAGIC
# MAGIC **Purpose:** Publish the generated dataset and Croissant metadata to HuggingFace
# MAGIC Datasets Hub. This is a **hard requirement** for NeurIPS D&B track submission —
# MAGIC the dataset must be publicly accessible at submission time.
# MAGIC
# MAGIC **What gets uploaded:**
# MAGIC - `small`, `medium`, `large` scale CSV files
# MAGIC - `metadata.json` for each scale
# MAGIC - `croissant.json` — machine-readable dataset description (NeurIPS D&B requirement)
# MAGIC - `README.md` — HuggingFace dataset card
# MAGIC
# MAGIC **Prereqs:**
# MAGIC - `HF_TOKEN` secret in Databricks Secret Scope (`dbutils.secrets.get("tfg", "hf_token")`)
# MAGIC - HuggingFace repo created: `YOUR_HF_ORG/travel-fraud-graphs`

# COMMAND ----------
# MAGIC %md ## 0. Setup

# COMMAND ----------

# %pip install huggingface_hub datasets travel-fraud-graphs

# COMMAND ----------
# MAGIC %md ## 1. Configuration

# COMMAND ----------

import os

# ---- Set your HuggingFace org/user and repo name ----
HF_REPO_ID = "YOUR_HF_ORG/travel-fraud-graphs"  # e.g. "bsajja/travel-fraud-graphs"
HF_TOKEN   = dbutils.secrets.get(scope="tfg", key="hf_token")  # type: ignore
LOCAL_TMP  = "/tmp/tfg_hf_upload"

os.makedirs(LOCAL_TMP, exist_ok=True)
print(f"Upload target: {HF_REPO_ID}")

# COMMAND ----------
# MAGIC %md ## 2. Generate All Scales

# COMMAND ----------

from travel_fraud_graphs import generate
from travel_fraud_graphs.exporters import export_csv
from travel_fraud_graphs.stats import compute_stats
import json, time

SCALES = ["small", "medium", "large"]
SEED   = 42

scale_metadata = {}
for scale in SCALES:
    print(f"\nGenerating: {scale} ...")
    t0   = time.time()
    data = generate(scale=scale, seed=SEED)
    elapsed = time.time() - t0
    out_dir = f"{LOCAL_TMP}/{scale}"
    export_csv(data, out_dir)
    scale_metadata[scale] = data.metadata
    scale_metadata[scale]["generation_time_sec"] = round(elapsed, 1)
    print(f"  Done in {elapsed:.1f}s  |  "
          f"{data.metadata['n_users_total']:,} users  "
          f"{data.metadata['fraud_user_ratio']:.1%} fraud")

# Save combined metadata
with open(f"{LOCAL_TMP}/all_scales_metadata.json", "w") as f:
    json.dump(scale_metadata, f, indent=2)

# COMMAND ----------
# MAGIC %md ## 3. Write Croissant Metadata (NeurIPS Hard Requirement)

# COMMAND ----------

# Croissant is a machine-readable dataset format required by NeurIPS D&B track.
# See: https://github.com/mlcommons/croissant

croissant = {
    "@context": {
        "@language": "en",
        "@vocab": "https://schema.org/",
        "sc": "https://schema.org/",
        "ml": "http://mlcommons.org/schema/",
        "cr": "http://mlcommons.org/croissant/"
    },
    "@type": "sc:Dataset",
    "name": "TravelFraudGraph",
    "description": (
        "A labeled heterogeneous property graph dataset for GNN-based fraud detection "
        "in travel networks. Contains three fraud ring types: ticketing fraud rings, "
        "ghost hotel schemes, and account takeover (ATO) rings. 9 node types, 11 edge "
        "relation types, per-node fraud labels, and ring membership annotations. "
        "Available at five scales (toy to xlarge). Exportable to NetworkX, PyG, and DGL."
    ),
    "url":     f"https://huggingface.co/datasets/{HF_REPO_ID}",
    "version": "1.0.0",
    "license": "https://opensource.org/licenses/MIT",
    "keywords": [
        "fraud detection", "graph neural networks", "benchmark",
        "travel", "heterogeneous graph", "synthetic dataset",
        "anomaly detection", "GNN"
    ],
    "creator": {
        "@type": "sc:Person",
        "name":  "TFG Authors",
        "email": "YOUR_EMAIL@example.com"
    },
    "distribution": [
        {
            "@type": "cr:FileObject",
            "@id":   f"nodes-user-{scale}",
            "name":  f"User nodes ({scale} scale)",
            "contentUrl": f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/{scale}/nodes/user.csv",
            "encodingFormat": "text/csv",
            "sha256": "TO_BE_FILLED"
        }
        for scale in SCALES
    ],
    "recordSet": [
        {
            "@type": "cr:RecordSet",
            "@id":   "user-nodes",
            "name":  "User Nodes",
            "description": "One row per user account with 11 features, is_fraud label, and ring_id/ring_type.",
            "field": [
                {"@type": "cr:Field", "name": "node_id",            "dataType": "sc:Integer"},
                {"@type": "cr:Field", "name": "is_fraud",           "dataType": "sc:Integer",
                 "description": "0=legitimate, 1=fraud"},
                {"@type": "cr:Field", "name": "ring_id",            "dataType": "sc:Integer",
                 "description": "-1 for legitimate nodes"},
                {"@type": "cr:Field", "name": "ring_type",          "dataType": "sc:Integer",
                 "description": "0=legit, 1=ticketing, 2=ghost_hotel, 3=ato"},
                {"@type": "cr:Field", "name": "account_age_days",   "dataType": "sc:Float"},
                {"@type": "cr:Field", "name": "booking_count_30d",  "dataType": "sc:Integer"},
                {"@type": "cr:Field", "name": "cancellation_rate",  "dataType": "sc:Float"},
                {"@type": "cr:Field", "name": "chargeback_count",   "dataType": "sc:Integer"},
                {"@type": "cr:Field", "name": "distinct_device_count", "dataType": "sc:Integer"},
                {"@type": "cr:Field", "name": "distinct_ip_count",  "dataType": "sc:Integer"},
                {"@type": "cr:Field", "name": "velocity_score",     "dataType": "sc:Float"},
            ]
        }
    ]
}

with open(f"{LOCAL_TMP}/croissant.json", "w") as f:
    json.dump(croissant, f, indent=2)

print("Croissant metadata written.")
print(json.dumps(croissant, indent=2)[:1000], "...")

# COMMAND ----------
# MAGIC %md ## 4. Write HuggingFace Dataset Card (README.md)

# COMMAND ----------

# Compute stats for the dataset card
med_data = generate(scale="medium", seed=42)
med_stats = compute_stats(med_data)
n_nodes  = sum(v["total"] for v in med_stats["node_counts"].values())
n_edges  = med_stats["total_edges"]

readme = f"""---
license: mit
task_categories:
  - graph-ml
  - node-classification
tags:
  - fraud-detection
  - graph-neural-networks
  - benchmark
  - travel
  - synthetic
  - heterogeneous-graph
pretty_name: TravelFraudGraph (TFG)
size_categories:
  - 10K<n<100K
---

# TravelFraudGraph (TFG)

**The first publicly available labeled graph-structured fraud dataset for travel networks.**

TFG fills a critical gap in the GNN fraud detection benchmark landscape: while PaySim,
AMLSim, YelpChi, and Elliptic cover payments, AML, reviews, and crypto respectively, no
graph-structured travel fraud dataset existed — despite travel being a \\$1.5T industry
with distinct fraud ring topologies.

## Dataset Summary

| Property | Value |
|----------|-------|
| Node types | 9 (user, device, IP, booking, flight, hotel, review, payment_card, loyalty_account) |
| Edge types | 11 relations |
| Fraud ring types | 3 (ticketing, ghost hotel, account takeover) |
| Labels | Per-node binary + ring_id + ring_type |
| Scales | toy / small / medium / large / xlarge |
| Generator | Open-source Python package (Apache 2.0) |
| Export formats | CSV, NetworkX, PyG HeteroData, DGL |

## Why Travel Fraud is Graph-Structured

Three structurally distinct fraud ring types, each producing unique motifs:

1. **Ticketing fraud rings** — Star topology: orchestrator + satellite accounts sharing
   1-4 devices/IPs, all targeting the same flight(s), with chargeback burst.
2. **Ghost hotel schemes** — Dense bipartite clique: reviewer cluster × ghost hotel cluster,
   plus fake booking cluster with high cancellation rates.
3. **Account takeover rings** — Attacker device cluster → compromised accounts → loyalty
   point drain chains (transfer path subgraphs).

## Scales

| Scale | Users | Hotels | Flights | Bookings | Edges | Rings |
|-------|-------|--------|---------|----------|-------|-------|
| toy   | ~500  | 50     | 80      | ~1,100   | ~7K   | 7     |
| small | ~2K   | 200    | 300     | ~5K      | ~30K  | 20    |
| medium| ~10K  | 1K     | 1.5K    | ~26K     | ~150K | 80    |
| large | ~50K  | 5K     | 8K      | ~130K    | ~750K | 260   |
| xlarge| ~200K | 20K    | 30K     | ~520K    | ~3M   | 800   |

## Usage

```python
# Install
pip install travel-fraud-graphs

# Generate
from travel_fraud_graphs import generate
data = generate(scale="medium", seed=42)

# PyG HeteroData
from travel_fraud_graphs.exporters import export_pyg
hetero = export_pyg(data)

# Train a GNN (node classification on user.y)
# hetero["user"].x  — features
# hetero["user"].y  — 0=legitimate, 1=fraud
```

## Baselines

| Model | AUC-ROC | Avg Prec | Macro-F1 |
|-------|---------|----------|----------|
| MLP (tabular only) | — | — | — |
| GraphSAGE | — | — | — |
| HAN | — | — | — |
| RGCN | — | — | — |

*(Fill after running Notebook 2)*

## Citation

```bibtex
@dataset{{tfg2026,
  title     = {{TravelFraudGraph: A Graph-Based Synthetic Fraud Ring Benchmark for Travel Networks}},
  author    = {{YOUR AUTHORS}},
  year      = {{2026}},
  url       = {{https://huggingface.co/datasets/{HF_REPO_ID}}},
  note      = {{arXiv:XXXX.XXXXX}}
}}
```

## License

MIT License
"""

with open(f"{LOCAL_TMP}/README.md", "w") as f:
    f.write(readme)
print("README.md written.")

# COMMAND ----------
# MAGIC %md ## 5. Upload to HuggingFace

# COMMAND ----------

from huggingface_hub import HfApi
import os

api = HfApi(token=HF_TOKEN)

# Create repo if it doesn't exist
try:
    api.create_repo(repo_id=HF_REPO_ID, repo_type="dataset", exist_ok=True)
    print(f"Repo ready: https://huggingface.co/datasets/{HF_REPO_ID}")
except Exception as e:
    print(f"Repo creation note: {e}")

# Upload README and croissant
for fname in ["README.md", "croissant.json", "all_scales_metadata.json"]:
    fpath = f"{LOCAL_TMP}/{fname}"
    if os.path.exists(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=HF_REPO_ID,
            repo_type="dataset",
        )
        print(f"  Uploaded: {fname}")

# Upload CSV files for each scale
for scale in SCALES:
    scale_dir = f"{LOCAL_TMP}/{scale}"
    for root, dirs, files in os.walk(scale_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            repo_path  = local_path.replace(LOCAL_TMP + "/", "")
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
            )
    print(f"  Uploaded: {scale} scale CSVs")

print(f"\nDataset live at: https://huggingface.co/datasets/{HF_REPO_ID}")
print("Now submit the Croissant file with your NeurIPS/KDD paper.")

# COMMAND ----------
# MAGIC %md ## 6. Post-Upload Checklist

# COMMAND ----------

checklist = """
POST-UPLOAD CHECKLIST (Complete before paper submission)
=========================================================

 [ ] HuggingFace dataset card is complete and renders correctly
 [ ] Croissant JSON validates at https://huggingface.co/spaces/MLCommons/croissant-editor
 [ ] All scale CSV files accessible without login (public repo)
 [ ] DOI registered via Zenodo (for permanent archival citation):
       https://zenodo.org  -> Link HuggingFace repo -> Get DOI
 [ ] README baseline table filled in (from Notebook 2 results)
 [ ] requirements.txt / pyproject.toml pinned versions match runtime
 [ ] arXiv preprint submitted with dataset URL in abstract
 [ ] GitHub repo README links to HuggingFace dataset
"""
print(checklist)

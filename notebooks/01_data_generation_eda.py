# Databricks notebook source
# MAGIC %md
# MAGIC # TravelFraudGraph — Notebook 1: Data Generation & Exploratory Data Analysis
# MAGIC
# MAGIC **Purpose:** Generate the TravelFraudGraph dataset at multiple scales, export to Delta tables
# MAGIC on DBFS, and produce exploratory visualisations for the paper (Section 3: Dataset Statistics).
# MAGIC
# MAGIC **Runtime:** Single-node cluster, Python 3.10+, no GPU required.
# MAGIC
# MAGIC **Estimated wall time:** ~3 min (medium scale), ~18 min (large scale)

# COMMAND ----------
# MAGIC %md ## 0. Setup & Installation

# COMMAND ----------

# %pip install travel-fraud-graphs networkx matplotlib seaborn pandas pyarrow
# Uncomment above for first run; comment out after cluster restart to save time.
# If installing from local wheel:
# %pip install /dbfs/FileStore/travel_fraud_graphs-0.1.0-py3-none-any.whl

# For development, install from the project source on DBFS:
import subprocess, sys
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "git+https://github.com/YOUR-ORG/travel-fraud-graphs.git"],
    capture_output=True, text=True
)
print(result.stdout[-500:] if result.stdout else "")
print(result.stderr[-300:] if result.stderr else "")

# COMMAND ----------
# MAGIC %md ## 1. Generate Dataset

# COMMAND ----------

import time
import pandas as pd
import numpy as np
from travel_fraud_graphs import generate, SCALE_PRESETS
from travel_fraud_graphs.exporters import export_csv
from travel_fraud_graphs.stats import compute_stats, format_report

# ----- Configuration -----
SCALE  = "medium"   # toy | small | medium | large | xlarge
SEED   = 42
DBFS_OUTPUT_PATH = f"/dbfs/FileStore/tfg/{SCALE}_seed{SEED}"

print(f"Scale preset: {SCALE}")
print(f"Config: {SCALE_PRESETS[SCALE]}")

t0 = time.time()
data = generate(scale=SCALE, seed=SEED)
elapsed = time.time() - t0

print(f"\nGenerated in {elapsed:.1f}s")
print(f"Metadata: {data.metadata}")

# COMMAND ----------
# MAGIC %md ## 2. Export to CSV / Delta

# COMMAND ----------

import os
os.makedirs(DBFS_OUTPUT_PATH, exist_ok=True)
export_csv(data, DBFS_OUTPUT_PATH)
print(f"CSV written to: {DBFS_OUTPUT_PATH}")

# Load nodes as Spark DataFrames and register as temp views
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

NODE_TYPES = [
    "user", "device", "ip_address", "booking", "flight",
    "hotel", "review", "payment_card", "loyalty_account"
]

node_dfs = {}
for ntype in NODE_TYPES:
    csv_path = f"{DBFS_OUTPUT_PATH}/nodes/{ntype}.csv"
    if os.path.exists(csv_path):
        sdf = spark.read.option("header", True).option("inferSchema", True).csv(
            csv_path.replace("/dbfs", "dbfs:")
        )
        node_dfs[ntype] = sdf
        sdf.createOrReplaceTempView(f"tfg_{ntype}")
        print(f"  {ntype}: {sdf.count():,} rows")

# COMMAND ----------
# MAGIC %md ## 3. Dataset Statistics Report

# COMMAND ----------

stats = compute_stats(data)
print(format_report(stats))

# COMMAND ----------
# MAGIC %md ## 4. Node Count & Class Balance Visualisation

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(f"TravelFraudGraph — {SCALE.capitalize()} Scale (seed={SEED})", fontsize=14)

# Left: Node counts per type
node_counts = {k: v["total"] for k, v in stats["node_counts"].items() if v["total"] > 0}
axes[0].barh(list(node_counts.keys()), list(node_counts.values()), color="#4C72B0")
axes[0].set_xlabel("Node Count")
axes[0].set_title("Nodes per Type")
for i, v in enumerate(node_counts.values()):
    axes[0].text(v * 1.01, i, f"{v:,}", va='center', fontsize=8)

# Right: Fraud % per node type
fraud_pcts = {k: v["fraud_pct"] for k, v in stats["node_counts"].items() if v["total"] > 0}
colors = ["#C44E52" if p > 10 else "#55A868" for p in fraud_pcts.values()]
axes[1].barh(list(fraud_pcts.keys()), list(fraud_pcts.values()), color=colors)
axes[1].set_xlabel("Fraud %")
axes[1].set_title("Fraud Node Ratio per Type\n(red = >10%)")
axes[1].axvline(x=5, linestyle='--', color='gray', linewidth=0.8, alpha=0.5)

plt.tight_layout()
plt.savefig(f"{DBFS_OUTPUT_PATH}/fig1_node_class_balance.png", dpi=150, bbox_inches='tight')
display(fig)
plt.close()

# COMMAND ----------
# MAGIC %md ## 5. User Feature Distributions: Fraud vs. Legitimate

# COMMAND ----------

import seaborn as sns

user_df = node_dfs.get("user")
if user_df is not None:
    pdf = user_df.toPandas()
    pdf["label"] = pdf["is_fraud"].map({0: "Legitimate", 1: "Fraud"})

    features_to_plot = [
        "account_age_days", "booking_count_30d", "cancellation_rate",
        "chargeback_count", "distinct_device_count", "velocity_score"
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("User Feature Distributions: Fraud vs. Legitimate", fontsize=13)

    for ax, feat in zip(axes.flatten(), features_to_plot):
        if feat in pdf.columns:
            for lbl, grp in pdf.groupby("label"):
                vals = grp[feat].dropna()
                ax.hist(vals, bins=30, alpha=0.6,
                        label=lbl, density=True,
                        color="#C44E52" if lbl == "Fraud" else "#4C72B0")
            ax.set_title(feat.replace("_", " ").title())
            ax.legend(fontsize=7)
            ax.set_xlabel("")

    plt.tight_layout()
    plt.savefig(f"{DBFS_OUTPUT_PATH}/fig2_user_feature_distributions.png", dpi=150, bbox_inches='tight')
    display(fig)
    plt.close()

# COMMAND ----------
# MAGIC %md ## 6. Booking Value Distribution

# COMMAND ----------

booking_df = node_dfs.get("booking")
if booking_df is not None:
    pdf_b = booking_df.toPandas()
    pdf_b["label"] = pdf_b["is_fraud"].map({0: "Legitimate", 1: "Fraud"})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for lbl, grp in pdf_b.groupby("label"):
        axes[0].hist(
            grp["booking_value_usd"].clip(0, 3000), bins=50,
            alpha=0.6, label=lbl, density=True,
            color="#C44E52" if lbl == "Fraud" else "#4C72B0"
        )
    axes[0].set_title("Booking Value Distribution (USD)")
    axes[0].set_xlabel("USD")
    axes[0].legend()

    for lbl, grp in pdf_b.groupby("label"):
        axes[1].hist(
            grp["lead_time_days"].clip(0, 200), bins=40,
            alpha=0.6, label=lbl, density=True,
            color="#C44E52" if lbl == "Fraud" else "#4C72B0"
        )
    axes[1].set_title("Lead Time Distribution (Days Before Travel)")
    axes[1].set_xlabel("Days")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{DBFS_OUTPUT_PATH}/fig3_booking_distributions.png", dpi=150, bbox_inches='tight')
    display(fig)
    plt.close()

# COMMAND ----------
# MAGIC %md ## 7. Degree Distribution (User Out-Degree)

# COMMAND ----------

import networkx as nx
from travel_fraud_graphs.exporters import export_networkx

G = export_networkx(data)
print(f"NetworkX: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

user_nodes   = [(n, d) for n, d in G.nodes(data=True) if d.get("type") == "user"]
fraud_users  = [n for n, d in user_nodes if d.get("is_fraud") == 1]
legit_users  = [n for n, d in user_nodes if d.get("is_fraud") == 0]

fraud_deg  = [G.out_degree(n) for n in fraud_users]
legit_deg  = [G.out_degree(n) for n in legit_users]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(legit_deg, bins=40, alpha=0.6, label="Legitimate", density=True, color="#4C72B0")
ax.hist(fraud_deg, bins=40, alpha=0.6, label="Fraud",      density=True, color="#C44E52")
ax.set_xlabel("User Out-Degree")
ax.set_ylabel("Density")
ax.set_title("User Out-Degree Distribution")
ax.legend()
plt.tight_layout()
plt.savefig(f"{DBFS_OUTPUT_PATH}/fig4_degree_distribution.png", dpi=150, bbox_inches='tight')
display(fig)
plt.close()

print(f"Fraud users  — mean degree: {np.mean(fraud_deg):.2f}  max: {max(fraud_deg) if fraud_deg else 0}")
print(f"Legit users  — mean degree: {np.mean(legit_deg):.2f}  max: {max(legit_deg) if legit_deg else 0}")

# COMMAND ----------
# MAGIC %md ## 8. Ring Size Distribution

# COMMAND ----------

from collections import Counter

ring_ids = data.node_ring_ids.get("user", [])
labels   = data.node_labels.get("user", [])

ring_sizes = Counter()
for rid, lbl in zip(ring_ids, labels):
    if lbl == 1 and rid >= 0:
        ring_sizes[rid] += 1

sizes = list(ring_sizes.values())
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(sizes, bins=20, color="#8172B2", edgecolor="white")
ax.set_xlabel("Ring Size (# Users)")
ax.set_ylabel("# Rings")
ax.set_title(f"Fraud Ring Size Distribution  (n={len(sizes)} rings)")
plt.tight_layout()
plt.savefig(f"{DBFS_OUTPUT_PATH}/fig5_ring_size_distribution.png", dpi=150, bbox_inches='tight')
display(fig)
plt.close()

print(f"Rings: {len(sizes)}  |  min={min(sizes)}  mean={np.mean(sizes):.1f}  max={max(sizes)}")

# COMMAND ----------
# MAGIC %md ## 9. Save Data to Delta Lake

# COMMAND ----------

# Save each node type to a managed Delta table for downstream notebooks
DELTA_DB = "tfg"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {DELTA_DB}")

for ntype, sdf in node_dfs.items():
    table_name = f"{DELTA_DB}.nodes_{ntype.replace('_', '')}"
    sdf.write.format("delta").mode("overwrite").saveAsTable(table_name)
    print(f"  Saved: {table_name}  ({sdf.count():,} rows)")

# Save metadata
import json
meta_json = json.dumps(data.metadata, indent=2)
dbutils.fs.put(
    f"dbfs:/FileStore/tfg/{SCALE}_seed{SEED}/metadata.json",
    meta_json, overwrite=True
)

print("\nAll tables saved. Proceed to Notebook 2 (GNN Baselines).")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC | Metric | Value |
# MAGIC |--------|-------|
# MAGIC | Scale | medium |
# MAGIC | Total users | ~10,000 |
# MAGIC | Total edges | ~150,000 |
# MAGIC | Fraud user ratio | ~16% |
# MAGIC | Node types | 9 |
# MAGIC | Edge types | 11 |
# MAGIC | Fraud ring types | 3 |
# MAGIC
# MAGIC **Next:** Run `02_gnn_baselines.py` to train GraphSAGE / HAN / RGCN on this data.

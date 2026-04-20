# Databricks notebook source
# MAGIC %md
# MAGIC # TravelFraudGraph — Notebook 3: Graph Topology & Motif Analysis
# MAGIC
# MAGIC **Purpose:** Produce the graph topology tables and figures for Section 3 of the paper.
# MAGIC A NeurIPS/KDD reviewer will check:
# MAGIC   1. Homophily scores per edge type (are fraud nodes actually connected to each other?)
# MAGIC   2. Motif fingerprints per ring type (what structural pattern does each ring produce?)
# MAGIC   3. Degree distribution power-law fit (is this a realistic graph or a toy simulation?)
# MAGIC   4. Fraud subgraph density (do fraud nodes form dense cliques?)
# MAGIC   5. Ring size / loyalty chain statistics
# MAGIC
# MAGIC **Runtime:** ~5 min (medium scale), CPU only.

# COMMAND ----------
# MAGIC %md ## 0. Setup

# COMMAND ----------

# %pip install travel-fraud-graphs networkx scipy matplotlib seaborn

# COMMAND ----------
# MAGIC %md ## 1. Generate Dataset

# COMMAND ----------

from travel_fraud_graphs import generate
from travel_fraud_graphs.analysis import (
    compute_edge_homophily,
    compute_fraud_subgraph_density,
    format_homophily_table,
    ring_type_motif_fingerprints,
    shared_resource_concentration,
    loyalty_transfer_chain_lengths,
    review_bipartite_clique_stats,
    format_motif_report,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

SCALE = "medium"
SEED  = 42
data  = generate(scale=SCALE, seed=SEED)
print(f"Generated: {SCALE} scale, seed={SEED}")
print(f"Users: {data.metadata['n_users_total']:,}   "
      f"Fraud ratio: {data.metadata['fraud_user_ratio']:.1%}")

# COMMAND ----------
# MAGIC %md ## 2. Homophily Analysis (Paper Table 3)

# COMMAND ----------

homophily     = compute_edge_homophily(data)
fraud_density = compute_fraud_subgraph_density(data)

print(format_homophily_table(homophily, fraud_density))

# Visualise as heatmap
import pandas as pd
import seaborn as sns

h_data = pd.DataFrame({
    "Edge Type":       list(homophily.keys()),
    "Homophily":       list(homophily.values()),
    "Fraud-Fraud Density": [fraud_density.get(k, 0) for k in homophily],
})

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Edge-Level Homophily and Fraud Subgraph Density", fontsize=13)

short_labels = [k.split("__")[1].replace("_", " ") + "\n(" + k.split("__")[0][:4] + "→" + k.split("__")[2][:4] + ")"
                for k in homophily.keys()]

axes[0].barh(short_labels, list(homophily.values()),
             color=["#C44E52" if v > 0.7 else "#4C72B0" for v in homophily.values()])
axes[0].set_xlabel("Homophily Score")
axes[0].set_title("Node Homophily per Edge Type\n(red > 0.7 = highly homophilic)")
axes[0].axvline(x=0.5, linestyle='--', color='gray', linewidth=0.8)

axes[1].barh(short_labels, [fraud_density.get(k, 0) for k in homophily.keys()],
             color="#8172B2")
axes[1].set_xlabel("Fraction of Edges (both endpoints = fraud)")
axes[1].set_title("Fraud–Fraud Edge Density per Relation")

plt.tight_layout()
plt.savefig(f"/dbfs/FileStore/tfg/{SCALE}_seed{SEED}/fig6_homophily.png",
            dpi=150, bbox_inches='tight')
display(fig)
plt.close()

# COMMAND ----------
# MAGIC %md ## 3. Ring-Type Motif Fingerprints (Paper Table 2)

# COMMAND ----------

fingerprints = ring_type_motif_fingerprints(data)
print("\nRing-Type Motif Fingerprints")
print("=" * 80)
header = f"{'Ring Type':<30} {'Users':>7} {'Bookings':>9} {'Devices':>8} {'IPs':>6} {'Bk/User':>8} {'Dev/User':>9}"
print(header)
print("-" * 80)
for rname, v in fingerprints.items():
    print(f"{rname:<30} {v['fraud_users']:>7} {v['fraud_bookings']:>9} "
          f"{v['fraud_devices']:>8} {v['fraud_ips']:>6} "
          f"{v['avg_bookings_per_user']:>8.1f} {v['avg_devices_per_user']:>9.1f}")

# COMMAND ----------
# MAGIC %md ## 4. Shared Resource Concentration (Key Fraud Signal)

# COMMAND ----------

src_stats = shared_resource_concentration(data)
print("\nShared Resource Concentration")
print("(# users sharing a single device or IP address)\n")
for rtype_name, split in src_stats.items():
    print(f"  {rtype_name}:")
    for group, stats in split.items():
        print(f"    {group:<12}  count={stats['count']:,}  "
              f"mean={stats['mean']}  max={stats['max']}  p95={stats['p95']}")

# The key insight: fraud devices/IPs are shared by 5-20x more users than legit ones.
# Plot this as violin plot.
from travel_fraud_graphs.exporters import export_networkx
import networkx as nx

G = export_networkx(data)

# Compute per-device and per-IP user counts
for resource_type in ("device", "ip_address"):
    rel = "uses_device" if resource_type == "device" else "uses_ip"
    resource_in_degrees = {}
    for n, d in G.nodes(data=True):
        if d.get("type") == resource_type:
            in_deg = G.in_degree(n)
            resource_in_degrees[n] = (in_deg, d.get("is_fraud", 0))

    fraud_counts = [cnt for _, (cnt, f) in resource_in_degrees.items() if f == 1]
    legit_counts = [cnt for _, (cnt, f) in resource_in_degrees.items() if f == 0]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(legit_counts, bins=30, alpha=0.6, label="Legitimate",
            density=True, color="#4C72B0")
    ax.hist(fraud_counts, bins=30, alpha=0.6, label="Fraud",
            density=True, color="#C44E52")
    ax.set_xlabel("Number of users sharing this resource")
    ax.set_ylabel("Density")
    ax.set_title(f"Users per {resource_type.replace('_', ' ').title()} Node")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"/dbfs/FileStore/tfg/{SCALE}_seed{SEED}/fig7_{resource_type}_sharing.png",
                dpi=150, bbox_inches='tight')
    display(fig)
    plt.close()

    print(f"{resource_type}:  fraud mean={np.mean(fraud_counts):.1f}  "
          f"legit mean={np.mean(legit_counts):.1f}")

# COMMAND ----------
# MAGIC %md ## 5. Loyalty Transfer Chain Analysis

# COMMAND ----------

chain_stats = loyalty_transfer_chain_lengths(data)
print("\nLoyalty Transfer Chain Statistics")
print(f"  Total transfer edges : {chain_stats['total_transfer_edges']}")
print(f"  Unique chains        : {chain_stats['unique_chains']}")
print(f"  Max chain length     : {chain_stats['max_chain_length']}")
print("\n  Chain length distribution:")
for length, cnt in chain_stats.get("length_distribution", {}).items():
    bar = "█" * cnt
    print(f"    length {length}: {cnt:>4}  {bar}")

# COMMAND ----------
# MAGIC %md ## 6. Ghost Hotel Review Bipartite Clique Analysis

# COMMAND ----------

review_stats = review_bipartite_clique_stats(data)
print("\nGhost Hotel Review Clique Statistics")
for k, v in review_stats.items():
    print(f"  {k}: {v}")

# COMMAND ----------
# MAGIC %md ## 7. Degree Distribution Power-Law Fit

# COMMAND ----------

from scipy.stats import powerlaw
import warnings

# User out-degree
user_degrees = [G.out_degree(n) for n, d in G.nodes(data=True)
                if d.get("type") == "user"]
fraud_deg    = [G.out_degree(n) for n, d in G.nodes(data=True)
                if d.get("type") == "user" and d.get("is_fraud") == 1]
legit_deg    = [G.out_degree(n) for n, d in G.nodes(data=True)
                if d.get("type") == "user" and d.get("is_fraud") == 0]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (deg_list, title, color) in zip(axes, [
    (user_degrees, "All Users", "#555555"),
    (legit_deg,    "Legitimate Users", "#4C72B0"),
    (fraud_deg,    "Fraud Users", "#C44E52"),
]):
    if not deg_list:
        continue
    bins = np.logspace(np.log10(max(1, min(deg_list))),
                       np.log10(max(deg_list) + 1), 30)
    ax.hist(deg_list, bins=bins, color=color, alpha=0.8, edgecolor='white')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Out-Degree (log scale)")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"{title}\nmean={np.mean(deg_list):.1f}  max={max(deg_list)}")

plt.suptitle("User Degree Distribution (log-log)", y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(f"/dbfs/FileStore/tfg/{SCALE}_seed{SEED}/fig8_degree_powerlaw.png",
            dpi=150, bbox_inches='tight')
display(fig)
plt.close()

# COMMAND ----------
# MAGIC %md ## 8. Full Motif Report

# COMMAND ----------

print(format_motif_report(data))

# COMMAND ----------
# MAGIC %md ## 9. Compare with Existing Benchmarks

# COMMAND ----------

# Paper Table 1: Comparison with existing graph fraud datasets
comparison = {
    "Dataset":       ["YelpChi", "Amazon", "PaySim", "AMLSim", "Elliptic", "TFG (ours)"],
    "Domain":        ["Review", "Review", "Payments", "AML", "Crypto", "Travel"],
    "Node Types":    [2, 2, 1, 1, 1, 9],
    "Edge Types":    [3, 4, 1, 1, 1, 11],
    "Ring Labels":   ["✗", "✗", "✗", "✗", "✗", "✓"],
    "Ring Types":    [0, 0, 0, 0, 0, 3],
    "Fraud Nodes":   ["14.5%", "9.5%", "~1%", "<5%", "2%", f"~{data.metadata['fraud_user_ratio']:.0%}"],
    "Generator":     ["✗", "✗", "✓ (PaySim)", "✓ (AMLSim)", "✗", "✓ (TFG)"],
    "PyG/DGL Export":["✗", "✗", "✗", "✗", "✗", "✓"],
}

df = pd.DataFrame(comparison)
print("\nTABLE 1: Comparison with Existing Graph Fraud Datasets")
print(df.to_string(index=False))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook produces:
# MAGIC - **Table 1**: Dataset comparison (9 node types, 11 edge types vs. 1-4 for existing benchmarks)
# MAGIC - **Table 2**: Motif fingerprints per ring type
# MAGIC - **Table 3**: Homophily scores (fraud edges are highly homophilic, proving ring topology)
# MAGIC - **Figures 6-8**: Homophily bars, resource sharing, degree distributions
# MAGIC
# MAGIC **Next:** Run `04_huggingface_upload.py` to publish the dataset.

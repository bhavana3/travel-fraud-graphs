# Databricks notebook source
# MAGIC %md
# MAGIC # TravelFraudBench — Notebook 5: Controlled Difficulty Study
# MAGIC
# MAGIC **Purpose:** Produce **Figure 3** — "AUC vs. Ring Size by Ring Type" — validating that
# MAGIC TravelFraudBench provides a controllable difficulty axis (Evaluative Claims E2 & E3).
# MAGIC
# MAGIC **Evaluative Claims Tested:**
# MAGIC > **E2:** A model's AUC declines predictably as ring size decreases, establishing a
# MAGIC > difficulty axis absent from all flat fraud benchmarks.
# MAGIC > **E3:** Detection performance differs across ring topologies — a model may excel
# MAGIC > on star-topology ticketing rings and fail on bipartite ghost hotel cliques.
# MAGIC
# MAGIC **Model:** GraphSAGE (best model from Notebook 2; consistent methodology).
# MAGIC **Split:** Ring-based 60/20/20, consistent with main results (Notebook 2).
# MAGIC **Scale:** Small (2,000 users) — faster iteration across 6 ring-size points.
# MAGIC **Ring sizes tested:** {3, 5, 8, 12, 20, 30} users per ring.
# MAGIC
# MAGIC **Runtime:** ~25 min on GPU, ~60 min CPU (6 points × ~5–10 min each).

# COMMAND ----------
# MAGIC %md ## 0. Setup

# COMMAND ----------

# %pip install travel-fraud-graphs torch torch_geometric scikit-learn matplotlib

# COMMAND ----------
# MAGIC %md ## 1. Imports and Config

# COMMAND ----------

import os
import torch
import numpy as np
import pandas as pd
import tempfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from travel_fraud_graphs import generate
from travel_fraud_graphs.exporters import export_pyg
from travel_fraud_graphs.exporters.csv_exp import export_csv

try:
    from torch_geometric.nn import SAGEConv
    PYG_OK = True
except ImportError:
    PYG_OK = False
    print("WARNING: torch_geometric not found.")

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED    = 42
SCALE   = "small"    # 2,000 users; fast enough for 6-point sweep
OUTDIR  = "/dbfs/FileStore/tfg/difficulty_study"
os.makedirs(OUTDIR, exist_ok=True)

# Ring size axis: hardest (3) → easiest (30); matches paper Figure 3 description
RING_SIZE_AXIS = [3, 5, 8, 12, 20, 30]

print(f"Device: {DEVICE}  |  Scale: {SCALE}  |  Output: {OUTDIR}")
print(f"Ring size axis: {RING_SIZE_AXIS}")

# COMMAND ----------
# MAGIC %md ## 2. Model Definition (GraphSAGE — identical to Notebook 2)

# COMMAND ----------

def _build_user_user_edges(edge_index_dict, device):
    """Project user→device and user→IP edges into user-user co-occurrence edges.
    Two users are connected if they share a device or IP address.
    This is the same helper used in Notebook 2."""
    src_all, dst_all = [], []
    for via_rel in ["uses_device", "uses_ip"]:
        ei = None
        for (s, r, d), e in edge_index_dict.items():
            if s == "user" and via_rel in r:
                ei = e; break
        if ei is None:
            continue
        via2users = defaultdict(list)
        for u, v in zip(ei[0].tolist(), ei[1].tolist()):
            via2users[v].append(u)
        for users in via2users.values():
            for u in users:
                for v in users:
                    if u != v:
                        src_all.append(u)
                        dst_all.append(v)
    if not src_all:
        return None
    return torch.tensor([src_all, dst_all], dtype=torch.long, device=device)


class GraphSAGEFraud(nn.Module):
    """GraphSAGE on projected user-user co-occurrence graph (shared devices/IPs).
    Identical architecture to Notebook 2."""
    def __init__(self, in_dim, hidden=64, out_dim=2, dropout=0.3):
        super().__init__()
        self.emb   = nn.Linear(in_dim, hidden)
        self.conv1 = SAGEConv(hidden, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.head  = nn.Linear(hidden, out_dim)
        self.drop  = dropout

    def forward(self, x_dict, edge_index_dict):
        x  = F.relu(self.emb(x_dict["user"].float()))
        ei = _build_user_user_edges(edge_index_dict, x.device)
        if ei is None:
            n  = x.size(0)
            ei = torch.stack([torch.arange(n, device=x.device)] * 2)
        x = F.relu(self.conv1(x, ei))
        x = F.dropout(x, p=self.drop, training=self.training)
        x = F.relu(self.conv2(x, ei))
        return self.head(F.dropout(x, p=self.drop, training=self.training))


class MLPBaseline(nn.Module):
    """Tabular MLP (no graph); serves as reference line in Figure 3."""
    def __init__(self, in_dim, hidden=64, out_dim=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x_dict, edge_index_dict):
        return self.net(x_dict["user"].float())

# COMMAND ----------
# MAGIC %md ## 3. Ring-Based Split Helper (consistent with Notebook 2)

# COMMAND ----------

def ring_based_split(users_df, hetero, rng_seed=42):
    """Apply ring-based 60/20/20 split: each fraud ring appears entirely in one partition.
    Identical logic to Notebook 2 Section 3, ensuring methodological consistency."""
    ring_id_all  = users_df["ring_id"].values
    is_fraud_all = users_df["is_fraud"].values
    ring_type_all = users_df["ring_type"].values
    n_users      = len(is_fraud_all)

    fraud_ring_ids = np.unique(ring_id_all[is_fraud_all == 1])
    rng = np.random.default_rng(rng_seed)
    shuffled = rng.permutation(fraud_ring_ids)
    n = len(shuffled)
    train_rings = set(shuffled[:int(0.60 * n)].tolist())
    val_rings   = set(shuffled[int(0.60 * n):int(0.80 * n)].tolist())
    test_rings  = set(shuffled[int(0.80 * n):].tolist())

    legit_idx = np.where(is_fraud_all == 0)[0]
    legit_idx = rng.permutation(legit_idx)
    n_l = len(legit_idx)
    legit_train = set(legit_idx[:int(0.60 * n_l)].tolist())
    legit_val   = set(legit_idx[int(0.60 * n_l):int(0.80 * n_l)].tolist())

    train_idx, val_idx, test_idx = [], [], []
    for i in range(n_users):
        rid = ring_id_all[i]
        if is_fraud_all[i] == 1:
            if rid in train_rings:  train_idx.append(i)
            elif rid in val_rings:  val_idx.append(i)
            else:                   test_idx.append(i)
        else:
            if i in legit_train:    train_idx.append(i)
            elif i in legit_val:    val_idx.append(i)
            else:                   test_idx.append(i)

    train_idx = np.array(train_idx)
    val_idx   = np.array(val_idx)
    test_idx  = np.array(test_idx)

    hetero["user"].train_mask = torch.zeros(n_users, dtype=torch.bool)
    hetero["user"].val_mask   = torch.zeros(n_users, dtype=torch.bool)
    hetero["user"].test_mask  = torch.zeros(n_users, dtype=torch.bool)
    hetero["user"].train_mask[train_idx] = True
    hetero["user"].val_mask[val_idx]     = True
    hetero["user"].test_mask[test_idx]   = True

    return train_idx, val_idx, test_idx, ring_id_all, is_fraud_all, ring_type_all

# COMMAND ----------
# MAGIC %md ## 4. Train and Evaluate Helper

# COMMAND ----------

def train_and_evaluate(data, model_cls=GraphSAGEFraud, epochs=100, hidden=64):
    """
    Train model on data (with ring-based split) and return per-ring-type AUC.
    Returns dict with keys matching DifficultyResult fields:
      auc_overall, ap_overall, f1_overall, auc_ticketing, auc_ghost, auc_ato
    """
    hetero = export_pyg(data)

    # Add reverse edges so GNNs can receive messages at user nodes
    for (src, rel, dst), ei in list(hetero.edge_index_dict.items()):
        hetero[dst, f"rev_{rel}", src].edge_index = ei.flip(0)

    # Ring-based split (consistent with Notebook 2)
    with tempfile.TemporaryDirectory() as td:
        export_csv(data, td)
        users_df = pd.read_csv(os.path.join(td, "nodes", "user.csv"))

    train_idx, val_idx, test_idx, ring_id_all, is_fraud_all, ring_type_all = \
        ring_based_split(users_df, hetero, rng_seed=SEED)

    # Normalise features using training set statistics only
    x_all   = hetero["user"].x.float()
    x_train = x_all[train_idx]
    mu  = x_train.mean(dim=0)
    std = x_train.std(dim=0).clamp(min=1e-6)
    hetero["user"].x = (x_all - mu) / std

    hetero = hetero.to(DEVICE)
    user_labels = hetero["user"].y.cpu().numpy()

    # Bail early if test set has no fraud (can happen at ring_size=3 with very few rings)
    test_labels_np = user_labels[test_idx]
    if test_labels_np.sum() == 0:
        print("  WARNING: No fraud users in test set — returning 0.5 AUC")
        return dict(auc_overall=0.5, ap_overall=0.0, f1_overall=0.0,
                    auc_ticketing=0.5, auc_ghost=0.5, auc_ato=0.5)

    in_dim = hetero["user"].x.size(1)
    model  = model_cls(in_dim, hidden=hidden).to(DEVICE)

    n_fraud = int(hetero["user"].y[hetero["user"].train_mask].sum())
    n_total = int(hetero["user"].train_mask.sum())
    pos_w   = (n_total - n_fraud) / max(n_fraud, 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_w], dtype=torch.float).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0.0
    best_state   = None
    patience     = 20
    patience_ctr = 0

    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(hetero.x_dict, hetero.edge_index_dict)
        loss = criterion(out[hetero["user"].train_mask],
                         hetero["user"].y[hetero["user"].train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if ep % 10 == 0:
            model.eval()
            with torch.no_grad():
                v_out   = model(hetero.x_dict, hetero.edge_index_dict)
                v_probs = torch.softmax(v_out[hetero["user"].val_mask], 1)[:, 1].cpu().numpy()
                v_true  = hetero["user"].y[hetero["user"].val_mask].cpu().numpy()
                if v_true.sum() > 0:
                    v_auc = roc_auc_score(v_true, v_probs)
                    if v_auc > best_val_auc:
                        best_val_auc = v_auc
                        best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                        patience_ctr = 0
                    else:
                        patience_ctr += 1
                        if patience_ctr >= patience:
                            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        out    = model(hetero.x_dict, hetero.edge_index_dict)
        t_mask = hetero["user"].test_mask
        probs  = torch.softmax(out[t_mask], 1)[:, 1].cpu().numpy()
        preds  = out[t_mask].argmax(1).cpu().numpy()
        truth  = hetero["user"].y[t_mask].cpu().numpy()

    def safe_auc(y, p):
        return float(roc_auc_score(y, p)) if (y.sum() > 0 and (1 - y).sum() > 0) else 0.5

    # Per-ring-type AUC: fraud users of that type vs ALL legit test users
    ring_types_test = ring_type_all[test_idx]
    is_fraud_test   = is_fraud_all[test_idx]

    type_aucs = {}
    for rtype_id, rtype_name in [(1, "auc_ticketing"), (2, "auc_ghost"), (3, "auc_ato")]:
        fraud_mask  = ring_types_test == rtype_id
        legit_mask  = is_fraud_test == 0
        combined    = fraud_mask | legit_mask
        if combined.sum() > 0 and truth[combined].sum() > 0:
            type_aucs[rtype_name] = safe_auc(truth[combined], probs[combined])
        else:
            type_aucs[rtype_name] = 0.5

    overall_ap = float(average_precision_score(truth, probs)) if truth.sum() > 0 else 0.0
    overall_f1 = float(f1_score(truth, preds, average="macro", zero_division=0))

    return dict(
        auc_overall=safe_auc(truth, probs),
        ap_overall=overall_ap,
        f1_overall=overall_f1,
        **type_aucs,
    )

# COMMAND ----------
# MAGIC %md ## 5. Run Difficulty Study — Ring Size Axis
# MAGIC
# MAGIC For each ring size target, generates a fresh dataset with rings narrowly centered
# MAGIC around that size (±2 users), trains GraphSAGE with ring-based split, and records
# MAGIC AUC decomposed by ring type.

# COMMAND ----------

print("=" * 65)
print("  TravelFraudBench — Controlled Difficulty Study (Figure 3)")
print(f"  Model: GraphSAGE  |  Scale: {SCALE}  |  Seed: {SEED}")
print(f"  Ring sizes: {RING_SIZE_AXIS}")
print("=" * 65)

# Scale-aware ring budget: keep total fraud users ≈ 15% of n_users
_scale_user_budget = {"toy": 75, "small": 300, "medium": 1500}
fraud_budget = _scale_user_budget.get(SCALE, 300)

results_rows = []

for rs in RING_SIZE_AXIS:
    # Keep total fraud users roughly constant by adjusting ring count
    n_rings = max(3, fraud_budget // (rs * 3))

    print(f"\n{'─'*55}")
    print(f"  ring_size_target={rs}  |  rings_per_type={n_rings}")

    data = generate(
        scale=SCALE,
        seed=SEED,
        n_ticketing_rings=n_rings,
        n_ghost_hotel_rings=n_rings,
        n_ato_rings=n_rings,
        ring_size_target=rs,
    )

    n_fraud = sum(data.node_labels.get("user", []))
    n_total = len(data.node_labels.get("user", []))
    fraud_ratio = n_fraud / max(n_total, 1)
    print(f"  Generated: {n_total:,} users  fraud={n_fraud}  ({fraud_ratio:.1%})")

    metrics = train_and_evaluate(data, model_cls=GraphSAGEFraud if PYG_OK else MLPBaseline)

    row = {
        "ring_size_target": rs,
        "n_rings_per_type": n_rings,
        "n_fraud_users": n_fraud,
        "fraud_ratio": round(fraud_ratio, 4),
        **metrics,
    }
    results_rows.append(row)

    print(f"  AUC={metrics['auc_overall']:.4f}  "
          f"Tick={metrics['auc_ticketing']:.4f}  "
          f"Ghost={metrics['auc_ghost']:.4f}  "
          f"ATO={metrics['auc_ato']:.4f}")

# COMMAND ----------
# MAGIC %md ## 6. Results Table (Figure 3 source data)

# COMMAND ----------

print("\n" + "=" * 80)
print("  Figure 3 Source Data: AUC vs. Ring Size by Ring Type (GraphSAGE, ring-based split)")
print("=" * 80)
print(f"{'Ring Size':>10} {'Fraud%':>7} {'n_rings':>8} "
      f"{'AUC-All':>9} {'AUC-Tick':>10} {'AUC-Ghost':>11} {'AUC-ATO':>9}")
print("-" * 80)
for r in results_rows:
    print(f"{r['ring_size_target']:>10} "
          f"{r['fraud_ratio']:>7.1%} "
          f"{r['n_rings_per_type']:>8} "
          f"{r['auc_overall']:>9.4f} "
          f"{r['auc_ticketing']:>10.4f} "
          f"{r['auc_ghost']:>11.4f} "
          f"{r['auc_ato']:>9.4f}")
print("=" * 80)
print("Expected: AUC should decrease monotonically as ring_size_target decreases.")
print("Differential curves across ring types validate Evaluative Claim E3.")

# COMMAND ----------
# MAGIC %md ## 7. Figure 3: AUC vs. Ring Size by Ring Type

# COMMAND ----------

ring_sizes   = [r["ring_size_target"]  for r in results_rows]
auc_overall  = [r["auc_overall"]       for r in results_rows]
auc_tick     = [r["auc_ticketing"]     for r in results_rows]
auc_ghost    = [r["auc_ghost"]         for r in results_rows]
auc_ato      = [r["auc_ato"]           for r in results_rows]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(ring_sizes, auc_overall, "k-o",  label="Overall",           linewidth=2,   markersize=7)
ax.plot(ring_sizes, auc_tick,    "b--s", label="Ticketing (star)",  linewidth=1.5, markersize=6)
ax.plot(ring_sizes, auc_ghost,   "r--^", label="Ghost Hotel (bipartite)", linewidth=1.5, markersize=6)
ax.plot(ring_sizes, auc_ato,     "g--D", label="ATO (chain)",       linewidth=1.5, markersize=6)
ax.axhline(y=0.5, linestyle=":", color="gray", linewidth=0.8, alpha=0.6, label="Random baseline (AUC=0.5)")

ax.set_xlabel("Ring Size (users per ring)", fontsize=12)
ax.set_ylabel("AUC-ROC", fontsize=12)
ax.set_title(
    "TravelFraudBench — Detection Difficulty vs. Ring Size\n"
    f"(GraphSAGE, {SCALE} scale, ring-based split, seed={SEED})",
    fontsize=11,
)
ax.legend(fontsize=10, loc="lower right")
ax.set_ylim(0.45, 1.02)
ax.set_xlim(2, 32)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out_path = f"{OUTDIR}/figure3_auc_vs_ring_size.png"
fig.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Figure 3 saved: {out_path}")

# COMMAND ----------
# MAGIC %md ## 8. Verification: Monotonicity Check

# COMMAND ----------

print("\nMonotonicity check (AUC should decrease as ring_size decreases):")
all_mono = True
for i in range(len(auc_overall) - 1):
    direction = "↓ OK" if auc_overall[i] <= auc_overall[i + 1] else "↑ VIOLATION"
    if "VIOLATION" in direction:
        all_mono = False
    print(f"  ring_size {ring_sizes[i]:>3} → {ring_sizes[i+1]:>3}: "
          f"{auc_overall[i]:.4f} → {auc_overall[i+1]:.4f}  {direction}")

if all_mono:
    print("\n  ✓ E2 VALIDATED: AUC declines monotonically with ring size.")
else:
    print("\n  ⚠ Non-monotonic result — check if caused by stochastic variation "
          "at border points. Consider running with 3 seeds and averaging.")

print("\nRing-type differential check (E3):")
for rs, t, g, a in zip(ring_sizes, auc_tick, auc_ghost, auc_ato):
    print(f"  ring_size={rs:>2}:  Tick={t:.4f}  Ghost={g:.4f}  ATO={a:.4f}  "
          f"spread={max(t,g,a)-min(t,g,a):.4f}")

# COMMAND ----------
# MAGIC %md ## 9. Summary for Paper
# MAGIC
# MAGIC Copy the printed table above into the paper's Figure 3 caption or appendix data table.
# MAGIC Update the Section 6.3 fill-me:
# MAGIC   - Which ring type is hardest at ring_size=3?
# MAGIC   - At which ring size does AUC first drop below 0.80?
# MAGIC   - Does monotonicity hold for all three ring types individually?

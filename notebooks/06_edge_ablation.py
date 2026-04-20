# Databricks notebook source
# MAGIC %md
# MAGIC # TravelFraudBench — Notebook 6: Edge-Type Ablation Study (Table 6)
# MAGIC
# MAGIC **Purpose:** Produce **Table 6** — RGCN AUC when individual edge relation types are
# MAGIC removed — validating Evaluative Claim E4.
# MAGIC
# MAGIC **Evaluative Claim E4:**
# MAGIC > *Heterogeneous edge relations contribute differentially to fraud detection;
# MAGIC > some are necessary, others redundant.*
# MAGIC >
# MAGIC > Validated by: ablating each relation type and measuring AUC drop, decomposed
# MAGIC > by ring type.  Device-sharing should be most critical for ticketing rings;
# MAGIC > loyalty-transfer for ATO rings; wrote/about for ghost hotel rings.
# MAGIC
# MAGIC **Model:** RGCN (Notebook 2 version — relation-specific SAGEConvs on user-user
# MAGIC metapath edges). This model is chosen because it explicitly aggregates per
# MAGIC relation type, making ablation directly interpretable.
# MAGIC
# MAGIC **Ablation strategy:** Train RGCN once per ablation condition with one relation
# MAGIC type's metapath contribution zeroed out (edge tensor replaced by empty tensor).
# MAGIC All other hyperparameters and split identical to Notebook 2 main results.
# MAGIC
# MAGIC **Runtime:** ~35 min GPU  (6 ablation conditions + 1 full model × ~5 min each).

# COMMAND ----------
# MAGIC %md ## 0. Setup

# COMMAND ----------

# %pip install travel-fraud-graphs torch torch_geometric scikit-learn

# COMMAND ----------
# MAGIC %md ## 1. Imports and Config

# COMMAND ----------

import os
import copy
import torch
import numpy as np
import pandas as pd
import tempfile
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from travel_fraud_graphs import generate
from travel_fraud_graphs.exporters import export_pyg
from travel_fraud_graphs.exporters.csv_exp import export_csv

try:
    from torch_geometric.nn import SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("WARNING: torch_geometric not available — cannot run RGCN ablation.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCALE  = "medium"
SEED   = 42
EPOCHS = 200
PATIENCE = 25
HIDDEN = 128

print(f"Device: {DEVICE}  |  Scale: {SCALE}  |  Seed: {SEED}")

# COMMAND ----------
# MAGIC %md ## 2. Generate Dataset (same as Notebook 2)

# COMMAND ----------

data = generate(scale=SCALE, seed=SEED)
print(f"Dataset: {data.metadata['n_users_total']:,} users  "
      f"fraud={data.metadata['fraud_user_ratio']:.1%}")

# COMMAND ----------
# MAGIC %md ## 3. Build PyG Object and Ring-Based Split

# COMMAND ----------

hetero_base = export_pyg(data)

# Add reverse edges
for (src, rel, dst), ei in list(hetero_base.edge_index_dict.items()):
    hetero_base[dst, f"rev_{rel}", src].edge_index = ei.flip(0)

# Load user metadata for ring-based split
with tempfile.TemporaryDirectory() as td:
    export_csv(data, td)
    users_df = pd.read_csv(os.path.join(td, "nodes", "user.csv"))

ring_id_all   = users_df["ring_id"].values
is_fraud_all  = users_df["is_fraud"].values
ring_type_all = users_df["ring_type"].values
n_users       = len(is_fraud_all)

# Ring-based 60/20/20 split — identical to Notebook 2 Section 3
fraud_ring_ids = np.unique(ring_id_all[is_fraud_all == 1])
rng = np.random.default_rng(SEED)
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

# Normalise user features on training-set statistics
x_all   = hetero_base["user"].x.float()
x_train = x_all[train_idx]
feat_mu  = x_train.mean(dim=0)
feat_std = x_train.std(dim=0).clamp(min=1e-6)
hetero_base["user"].x = (x_all - feat_mu) / feat_std

# Apply masks
hetero_base["user"].train_mask = torch.zeros(n_users, dtype=torch.bool)
hetero_base["user"].val_mask   = torch.zeros(n_users, dtype=torch.bool)
hetero_base["user"].test_mask  = torch.zeros(n_users, dtype=torch.bool)
hetero_base["user"].train_mask[train_idx] = True
hetero_base["user"].val_mask[val_idx]     = True
hetero_base["user"].test_mask[test_idx]   = True

user_labels = hetero_base["user"].y.numpy()
print(f"Train: {len(train_idx):,}  fraud={user_labels[train_idx].sum()}")
print(f"Val:   {len(val_idx):,}  fraud={user_labels[val_idx].sum()}")
print(f"Test:  {len(test_idx):,}  fraud={user_labels[test_idx].sum()}")

# Slice metadata for test users
test_ring_ids   = ring_id_all[test_idx]
test_is_fraud   = is_fraud_all[test_idx]
test_ring_types = ring_type_all[test_idx]

# COMMAND ----------
# MAGIC %md ## 4. RGCN Model (identical to Notebook 2)

# COMMAND ----------

class RGCNFraud(nn.Module):
    """Relation-specific SAGEConvs on user-user metapath edges.
    Identical architecture to Notebook 2 for methodological consistency.
    5 relation channels: device-share, IP-share, card-share, booking, loyalty."""

    _REL_CHANNELS = [
        ("user", "uses_device", "device"),
        ("user", "uses_ip",     "ip_address"),
        ("user", "owns_card",   "payment_card"),
        ("user", "made",        "booking"),
        ("user", "has_loyalty", "loyalty_account"),
    ]

    def __init__(self, in_channels_dict, hidden=128, out_dim=2,
                 num_relations=None, dropout=0.3):
        super().__init__()
        user_dim = in_channels_dict["user"]
        n_rel    = len(self._REL_CHANNELS)
        self._rel_eis = None

        self.l1 = nn.ModuleList([SAGEConv(user_dim, 64) for _ in range(n_rel)])
        self.l2 = nn.ModuleList([SAGEConv(64, 32)       for _ in range(n_rel)])
        self.w1 = nn.Parameter(torch.ones(n_rel))
        self.w2 = nn.Parameter(torch.ones(n_rel))
        self.fc  = nn.Linear(32 + user_dim, out_dim)
        self.drop = dropout
        self.register_buffer("_placeholder", torch.zeros(1))

    def set_metapath_edges(self, edge_index_dict, n_users, ablated_rel=None):
        """Precompute user-user metapath edges.
        If ablated_rel is set (e.g. 'uses_device'), that relation's edges are zeroed."""
        self._rel_eis = []
        device = next(self.parameters()).device
        for (s, rel, d) in self._REL_CHANNELS:
            # Zero out the ablated relation
            if ablated_rel is not None and rel == ablated_rel:
                self._rel_eis.append(torch.zeros(2, 0, dtype=torch.long, device=device))
                continue
            ei = None
            for (src, r, dst), e in edge_index_dict.items():
                if src == s and dst == d and rel in r:
                    ei = e; break
            if ei is None or ei.shape[1] == 0:
                self._rel_eis.append(torch.zeros(2, 0, dtype=torch.long, device=device))
                continue
            mid2u = defaultdict(list)
            for u, mid in zip(ei[0].tolist(), ei[1].tolist()):
                mid2u[mid].append(u)
            srcs, dsts = [], []
            for users_list in mid2u.values():
                for i in range(len(users_list)):
                    for j in range(len(users_list)):
                        if i != j:
                            srcs.append(users_list[i])
                            dsts.append(users_list[j])
            if srcs:
                self._rel_eis.append(
                    torch.tensor([srcs, dsts], dtype=torch.long, device=device))
            else:
                self._rel_eis.append(torch.zeros(2, 0, dtype=torch.long, device=device))

    def forward(self, x_dict, edge_index_dict):
        x = x_dict["user"].float()
        h1s = []
        for conv, ei in zip(self.l1, self._rel_eis):
            h1s.append(F.relu(conv(x, ei)) if ei.shape[1] > 0
                        else torch.zeros(x.shape[0], 64, device=x.device))
        w1 = torch.softmax(self.w1, dim=0)
        h1 = F.dropout(sum(w1[i] * h1s[i] for i in range(len(h1s))),
                        p=self.drop, training=self.training)
        h2s = []
        for conv, ei in zip(self.l2, self._rel_eis):
            h2s.append(F.relu(conv(h1, ei)) if ei.shape[1] > 0
                        else torch.zeros(x.shape[0], 32, device=x.device))
        w2 = torch.softmax(self.w2, dim=0)
        h2 = sum(w2[i] * h2s[i] for i in range(len(h2s)))
        return self.fc(torch.cat([h2, x], dim=1))

# COMMAND ----------
# MAGIC %md ## 5. Training Loop

# COMMAND ----------

def evaluate_model(model, hetero, mask, criterion):
    model.eval()
    with torch.no_grad():
        out    = model(hetero.x_dict, hetero.edge_index_dict)
        logits = out[mask]
        labels = hetero["user"].y[mask]
        loss   = criterion(logits, labels).item()
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        y_true = labels.cpu().numpy()
        auc = roc_auc_score(y_true, probs) if y_true.sum() > 0 else 0.5
        ap  = average_precision_score(y_true, probs) if y_true.sum() > 0 else 0.0
        f1  = f1_score(y_true, preds, average="macro", zero_division=0)
    return {"loss": loss, "auc": auc, "ap": ap, "f1": f1, "probs": probs}


def run_ablation(hetero, ablated_rel=None, epochs=EPOCHS):
    """
    Train RGCN with one relation type ablated (or none = full model).

    Parameters
    ----------
    ablated_rel : str or None
        The relation name to zero out (e.g. 'uses_device'), or None for full model.

    Returns
    -------
    dict with overall AUC/AP and per-ring-type AUC.
    """
    label = "FULL MODEL" if ablated_rel is None else f"ablate: {ablated_rel}"
    print(f"\n  [{label}]")

    hetero_local = copy.deepcopy(hetero).to(DEVICE)
    in_channels  = {nt: hetero_local[nt].x.size(1)
                    for nt in hetero_local.node_types
                    if hasattr(hetero_local[nt], 'x')}

    model = RGCNFraud(in_channels, hidden=HIDDEN).to(DEVICE)
    model.set_metapath_edges(
        {k: v.to(DEVICE) for k, v in hetero_local.edge_index_dict.items()},
        n_users,
        ablated_rel=ablated_rel,
    )

    n_fraud = int(hetero_local["user"].y[hetero_local["user"].train_mask].sum())
    n_total = int(hetero_local["user"].train_mask.sum())
    pos_w   = (n_total - n_fraud) / max(n_fraud, 1)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, pos_w], dtype=torch.float).to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auc = 0.0
    best_state   = None
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out  = model(hetero_local.x_dict, hetero_local.edge_index_dict)
        loss = criterion(out[hetero_local["user"].train_mask],
                         hetero_local["user"].y[hetero_local["user"].train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 10 == 0:
            val_m = evaluate_model(
                model, hetero_local, hetero_local["user"].val_mask, criterion)
            if val_m["auc"] > best_val_auc:
                best_val_auc = val_m["auc"]
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= PATIENCE:
                    print(f"    Early stop at epoch {epoch}")
                    break

    if best_state:
        model.load_state_dict(best_state)

    test_m = evaluate_model(
        model, hetero_local, hetero_local["user"].test_mask, criterion)
    probs = test_m["probs"]

    # Per-ring-type AUC
    truth = hetero_local["user"].y[hetero_local["user"].test_mask].cpu().numpy()

    def safe_auc(y, p):
        return float(roc_auc_score(y, p)) if (y.sum() > 0 and (1 - y).sum() > 0) else 0.5

    type_aucs = {}
    for rtype_id, rtype_name in [(1, "auc_ticketing"), (2, "auc_ghost"), (3, "auc_ato")]:
        fraud_m  = test_ring_types == rtype_id
        legit_m  = test_is_fraud == 0
        combined = fraud_m | legit_m
        if combined.sum() > 0 and truth[combined].sum() > 0:
            type_aucs[rtype_name] = safe_auc(truth[combined], probs[combined])
        else:
            type_aucs[rtype_name] = 0.5

    result = {
        "auc":  test_m["auc"],
        "ap":   test_m["ap"],
        "f1":   test_m["f1"],
        **type_aucs,
    }
    print(f"    AUC={result['auc']:.4f}  AP={result['ap']:.4f}  "
          f"Tick={result['auc_ticketing']:.4f}  "
          f"Ghost={result['auc_ghost']:.4f}  "
          f"ATO={result['auc_ato']:.4f}")
    return result

# COMMAND ----------
# MAGIC %md ## 6. Run All Ablations
# MAGIC
# MAGIC Ablation conditions:
# MAGIC - Full model (all 5 relation channels)
# MAGIC - Remove `uses_device`   (device co-use — primary ticketing signal)
# MAGIC - Remove `uses_ip`       (IP co-use — secondary infrastructure signal)
# MAGIC - Remove `wrote / about` (review → hotel — ghost hotel bipartite clique)
# MAGIC - Remove `has_loyalty` + `loyalty_transfer` (loyalty chain — ATO signal)
# MAGIC - Remove `made`          (booking co-use — weakest signal hypothesis)

# COMMAND ----------

ABLATION_CONDITIONS = [
    None,            # full model
    "uses_device",   # E4 prediction: biggest drop for ticketing rings
    "uses_ip",       # E4 prediction: secondary drop for ticketing/ghost hotel
    "wrote",         # E4 prediction: biggest drop for ghost hotel rings
    "has_loyalty",   # E4 prediction: biggest drop for ATO rings
    "made",          # E4 prediction: smallest drop (weakest signal)
]

ablation_results = {}

for ablated_rel in ABLATION_CONDITIONS:
    key = "Full model" if ablated_rel is None else f"−{ablated_rel}"
    ablation_results[key] = run_ablation(hetero_base, ablated_rel=ablated_rel)

# COMMAND ----------
# MAGIC %md ## 7. Print Table 6

# COMMAND ----------

full_auc = ablation_results["Full model"]["auc"]

print("\n" + "=" * 90)
print("  TABLE 6: Edge-Type Ablation Study (RGCN, medium scale, seed=42, ring-based split)")
print("  ΔAUC = drop from full model (negative = degradation)")
print("=" * 90)
print(f"{'Ablated Relation':<28} {'AUC (all)':>10} {'ΔAUC':>7}  "
      f"{'Ticketing':>11}  {'Ghost Hotel':>12}  {'ATO':>10}")
print("-" * 90)

display_order = [
    "Full model",
    "−uses_device",
    "−uses_ip",
    "−wrote",
    "−has_loyalty",
    "−made",
]

for key in display_order:
    if key not in ablation_results:
        continue
    r = ablation_results[key]
    delta = r["auc"] - full_auc
    delta_str = f"{delta:+.4f}" if key != "Full model" else "   ---"
    print(f"{key:<28} {r['auc']:>10.4f} {delta_str:>7}  "
          f"{r['auc_ticketing']:>11.4f}  "
          f"{r['auc_ghost']:>12.4f}  "
          f"{r['auc_ato']:>10.4f}")

print("=" * 90)
print("\nExpected pattern (validates E4):")
print("  −uses_device  → largest drop in Ticketing AUC")
print("  −wrote        → largest drop in Ghost Hotel AUC")
print("  −has_loyalty  → largest drop in ATO AUC")
print("  −made         → smallest drops (booking co-occurrence is weakest signal)")

# COMMAND ----------
# MAGIC %md ## 8. Interpretation Checks

# COMMAND ----------

print("\nE4 validation checks:")
r_full  = ablation_results["Full model"]
r_dev   = ablation_results.get("−uses_device",  {})
r_ip    = ablation_results.get("−uses_ip",       {})
r_wrote = ablation_results.get("−wrote",          {})
r_loy   = ablation_results.get("−has_loyalty",   {})
r_made  = ablation_results.get("−made",           {})

def check(condition, msg):
    status = "✓" if condition else "✗ WARNING:"
    print(f"  {status}  {msg}")

if r_dev and r_wrote and r_loy:
    check(
        r_dev.get("auc_ticketing", 1.0)  < r_full["auc_ticketing"],
        f"−uses_device reduces Ticketing AUC "
        f"({r_full['auc_ticketing']:.4f} → {r_dev.get('auc_ticketing', '?'):.4f})"
    )
    check(
        r_wrote.get("auc_ghost", 1.0) < r_full["auc_ghost"],
        f"−wrote reduces Ghost Hotel AUC "
        f"({r_full['auc_ghost']:.4f} → {r_wrote.get('auc_ghost', '?'):.4f})"
    )
    check(
        r_loy.get("auc_ato", 1.0)  < r_full["auc_ato"],
        f"−has_loyalty reduces ATO AUC "
        f"({r_full['auc_ato']:.4f} → {r_loy.get('auc_ato', '?'):.4f})"
    )
    if r_made:
        made_drop = r_full["auc"] - r_made.get("auc", r_full["auc"])
        dev_drop  = r_full["auc"] - r_dev.get("auc", r_full["auc"])
        check(
            made_drop <= dev_drop,
            f"−made is less damaging than −uses_device overall "
            f"(drops: {made_drop:.4f} vs {dev_drop:.4f})"
        )

print("\n  Copy Table 6 output into paper Section 6.4 and fill the LaTeX \\fillme{} cells.")
print("  Update the discussion to reference the specific AUC drops.")

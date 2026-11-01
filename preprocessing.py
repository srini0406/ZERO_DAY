"""
preprocessing.py — Stage 1: Data Loading, Feature Selection, Encoding, Compression

UNSW-NB15 Hybrid DL+QML Zero-Day Intrusion Detection Framework
----------------------------------------------------------------
This file handles the complete data pipeline from raw CSVs to model-ready arrays.
It produces two parallel feature paths:
  1. FULL path  → X_train_full / X_test_full  (all 49 features, for CNN)
  2. QUANTUM path → quantum_input_train / quantum_input_test  (6 PCA components,
     one per qubit — the qubit budget is the primary constraint here)
  3. INTERPRETABLE path → X_train_15 / X_test_15  (selected+encoded, pre-PCA —
     used for SHAP explanations; PCA components have no analyst meaning)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.decomposition import PCA

# ── reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_data(train_path: str, test_path: str):
    """Load both CSVs, return raw DataFrames keeping label + attack_cat."""
    print("\n[1/8] Loading data...")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    # Normalise attack_cat: strip whitespace, title-case for consistency
    train["attack_cat"] = train["attack_cat"].str.strip().str.title()
    test["attack_cat"]  = test["attack_cat"].str.strip().str.title()

    print(f"  Train : {train.shape[0]:,} rows × {train.shape[1]} cols")
    print(f"  Test  : {test.shape[0]:,} rows × {test.shape[1]} cols")
    print(f"  Train label distribution: {train['label'].value_counts().to_dict()}")
    print(f"  Attack categories: {sorted(train['attack_cat'].unique())}")
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# 2.  INTERPRETABLE FEATURE POOL
#     These 20 features are analyst-understandable raw connection attributes.
#     Aggregated connection-count features (ct_srv_dst, ct_dst_sport_ltm, etc.)
#     are deliberately excluded: they rank well statistically but a security
#     analyst reading a SHAP explanation for an alert cannot act on
#     "ct_dst_sport_ltm = 4" — it has no direct packet-level meaning.
# ─────────────────────────────────────────────────────────────────────────────

INTERPRETABLE_POOL = [
    "dur",       # Connection duration in seconds
    "proto",     # Network protocol (tcp / udp / icmp / …) — CATEGORICAL
    "service",   # Application-layer service (http / dns / ftp / …) — CATEGORICAL
    "state",     # Connection state at close (FIN / INT / CON / …) — CATEGORICAL
    "spkts",     # Source-to-dest packet count
    "dpkts",     # Dest-to-source packet count
    "sbytes",    # Source-to-dest byte total
    "dbytes",    # Dest-to-source byte total
    "rate",      # Packet rate (packets / second)
    "sttl",      # Source TTL value
    "dttl",      # Destination TTL value
    "sload",     # Source bit-load per second
    "dload",     # Destination bit-load per second
    "swin",      # Source TCP window advertisement size
    "dwin",      # Destination TCP window advertisement size
    "tcprtt",    # TCP handshake round-trip time (SYN→SYN-ACK→ACK)
    "synack",    # Time between SYN and SYN-ACK (server response latency)
    "ackdat",    # Time between SYN-ACK and ACK (client acknowledgement latency)
    "smean",     # Mean packet size from source
    "dmean",     # Mean packet size from destination
]

# Which of the pool are categorical (need special MI handling + encoding)
CATEGORICAL_IN_POOL = {"proto", "service", "state"}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE SELECTION — Mutual Information on the interpretable pool
#     WHY Mutual Information over ANOVA F-test:
#       ANOVA assumes approximately Gaussian-distributed features and roughly
#       linear separation between class means.  Network traffic features
#       (packet counts, byte loads, TTLs) are heavily skewed / zero-inflated,
#       and attacks often separate via non-linear combinations rather than
#       simple mean shifts.  MI captures ANY dependency structure, linear or
#       not, making it far more appropriate here.
#
#     WHY select BEFORE one-hot encoding:
#       Encoding first would fragment "proto" (133 unique values) into 133+
#       binary columns.  MI would then score each dummy column individually,
#       making it impossible to select the whole "proto" feature as a unit,
#       and would balloon the search space from ~20 columns to 100+.
#       Selecting on the raw label-encoded categorical preserves feature
#       identity and keeps the space tractable.
#
#     WHY top 10 (not more):
#       Whichever categorical columns survive will each expand during encoding.
#       Selecting fewer raw features up front keeps post-encoding dimensionality
#       manageable (target ~20-50 columns) rather than exploding unnecessarily.
# ─────────────────────────────────────────────────────────────────────────────

def select_features_mi(train_df: pd.DataFrame, top_k: int = 10):
    """
    Rank the interpretable pool by Mutual Information (MI) against 'label',
    return the names of the top_k features.

    Categorical columns (proto / service / state) are label-encoded to integers
    for MI scoring, with discrete_features=True so sklearn treats them as
    discrete distributions rather than continuous — this scores them correctly.
    """
    print(f"\n[2/8] Feature selection via Mutual Information (top {top_k})...")

    pool_df = train_df[INTERPRETABLE_POOL].copy()
    y = train_df["label"].values

    # Temporary integer encoding for categorical columns (MI only — not saved)
    discrete_mask = []
    le_temp = {}
    for col in INTERPRETABLE_POOL:
        if col in CATEGORICAL_IN_POOL:
            le = LabelEncoder()
            pool_df[col] = le.fit_transform(pool_df[col].astype(str))
            le_temp[col] = le
            discrete_mask.append(True)   # mark as discrete for MI
        else:
            discrete_mask.append(False)  # treat as continuous for MI

    # Mutual information — discrete_features mask ensures correct scoring
    mi_scores = mutual_info_classif(
        pool_df.values, y,
        discrete_features=discrete_mask,
        random_state=RANDOM_SEED
    )

    # Build ranking DataFrame
    mi_df = pd.DataFrame({
        "feature": INTERPRETABLE_POOL,
        "type": ["categorical" if f in CATEGORICAL_IN_POOL else "numeric"
                 for f in INTERPRETABLE_POOL],
        "mi_score": mi_scores
    }).sort_values("mi_score", ascending=False).reset_index(drop=True)

    print("\n  MI Ranking (full interpretable pool):")
    print(f"  {'Rank':<5} {'Feature':<15} {'Type':<12} {'MI Score':<10}")
    print("  " + "-" * 44)
    for i, row in mi_df.iterrows():
        marker = " ◄ selected" if i < top_k else ""
        print(f"  {i+1:<5} {row['feature']:<15} {row['type']:<12} {row['mi_score']:.4f}{marker}")

    selected = mi_df.head(top_k)["feature"].tolist()
    print(f"\n  Selected top {top_k}: {selected}")

    mi_df.to_csv(os.path.join(ARTIFACTS_DIR, "mi_feature_ranking.csv"), index=False)
    print(f"  Saved → artifacts/mi_feature_ranking.csv")

    return selected, mi_df


# ─────────────────────────────────────────────────────────────────────────────
# 4.  ENCODING — one-hot encode ONLY selected categoricals; Min-Max scale numerics
#     WHY only selected categoricals, not all three automatically:
#       A categorical column that didn't make the MI top-10 would add noise
#       columns that the model didn't need.  Encode only what was selected.
#     WHY the 1% "other" bucket for high-cardinality categoricals:
#       "proto" has 133 unique values; rare protocols (< 1% of rows) would each
#       contribute a near-zero column, wasting dimensions and potentially
#       causing OOV (out-of-vocabulary) errors at inference time.
# ─────────────────────────────────────────────────────────────────────────────

def encode_selected_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_features: list
):
    """
    Encode selected features for the interpretable/quantum path.
    Returns: X_train_enc, X_test_enc, encoded_feature_names, artifacts_dict
    """
    print("\n[3/8] Encoding selected features...")

    selected_cats = [f for f in selected_features if f in CATEGORICAL_IN_POOL]
    selected_nums = [f for f in selected_features if f not in CATEGORICAL_IN_POOL]

    print(f"  Selected numeric   : {selected_nums}")
    print(f"  Selected categorical: {selected_cats}")

    # ── numeric: Min-Max normalization ───────────────────────────────────────
    scaler = MinMaxScaler()
    X_train_num = scaler.fit_transform(train_df[selected_nums].values.astype(float))
    X_test_num  = scaler.transform(test_df[selected_nums].values.astype(float))
    num_feature_names = selected_nums.copy()

    # ── categorical: rare-bucket + one-hot ───────────────────────────────────
    ohe_artifacts = {}  # {col: {"keep_cats": [...], "other_threshold": N}}
    train_cat_parts, test_cat_parts = [], []
    cat_feature_names = []

    for col in selected_cats:
        train_col = train_df[col].astype(str)
        test_col  = test_df[col].astype(str)

        n_unique = train_col.nunique()
        print(f"  '{col}': {n_unique} unique values in training")

        if n_unique > 15:
            # Group categories appearing in < 1% of training rows into "other"
            threshold = int(0.01 * len(train_df))
            counts = train_col.value_counts()
            keep_cats = counts[counts >= threshold].index.tolist()
            print(f"    → {n_unique} > 15 unique; grouping {n_unique - len(keep_cats)} "
                  f"rare categories into 'other' (threshold = {threshold} rows)")

            train_col = train_col.where(train_col.isin(keep_cats), other="other")
            test_col  = test_col.where(test_col.isin(keep_cats), other="other")
        else:
            keep_cats = train_col.unique().tolist()

        # Ensure test unseen values also map to "other"
        test_col = test_col.where(test_col.isin(keep_cats + ["other"]), other="other")
        all_cats = sorted(set(keep_cats + ["other"]))

        # One-hot encode manually (avoids sklearn OHE train/test mismatch issues)
        for cat in all_cats:
            col_name = f"{col}_{cat}"
            train_cat_parts.append((train_col == cat).astype(float).values)
            test_cat_parts.append((test_col == cat).astype(float).values)
            cat_feature_names.append(col_name)

        ohe_artifacts[col] = {"keep_cats": keep_cats, "all_cats": all_cats}

    # ── assemble ─────────────────────────────────────────────────────────────
    parts_train = [X_train_num]
    parts_test  = [X_test_num]

    if train_cat_parts:
        parts_train.append(np.column_stack(train_cat_parts))
        parts_test.append(np.column_stack(test_cat_parts))

    X_train_enc = np.hstack(parts_train).astype(np.float32)
    X_test_enc  = np.hstack(parts_test).astype(np.float32)

    encoded_feature_names = num_feature_names + cat_feature_names

    print(f"\n  Post-encoding feature count : {len(encoded_feature_names)}")
    print(f"  Feature list: {encoded_feature_names}")
    print(f"  X_train_enc shape: {X_train_enc.shape}")
    print(f"  X_test_enc  shape: {X_test_enc.shape}")

    pd.DataFrame({"feature": encoded_feature_names}).to_csv(
        os.path.join(ARTIFACTS_DIR, "encoded_feature_names.csv"), index=False
    )

    artifacts = {
        "scaler_15": scaler,
        "selected_nums": selected_nums,
        "selected_cats": selected_cats,
        "ohe_artifacts": ohe_artifacts,
        "encoded_feature_names": encoded_feature_names,
    }
    return X_train_enc, X_test_enc, encoded_feature_names, artifacts


# ─────────────────────────────────────────────────────────────────────────────
# 5.  COMPRESSION — PCA to exactly 6 components (one per qubit)
#     WHY 6 components:
#       The quantum circuit uses N_QUBITS = 6 (angle embedding: one feature per
#       qubit).  The post-encoding feature count (~20-50 columns) must be
#       compressed to exactly 6 to match the qubit budget.
#     WHY fit PCA on training data only:
#       Fitting on the test set (or combined) would leak information about the
#       test distribution into the transformation, inflating test performance.
#       PCA is fitted on train, then applied identically to both sets.
# ─────────────────────────────────────────────────────────────────────────────

N_QUANTUM_COMPONENTS = 6  # must match N_QUBITS in quantum_models.py

def compress_to_quantum_input(
    X_train_enc: np.ndarray,
    X_test_enc: np.ndarray,
    n_components: int = N_QUANTUM_COMPONENTS
):
    """PCA compression to n_components for quantum circuit input."""
    print(f"\n[4/8] PCA compression → {n_components} quantum input components...")

    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    X_train_q = pca.fit_transform(X_train_enc)   # fit on train ONLY
    X_test_q  = pca.transform(X_test_enc)         # apply same PCA to test

    evr = pca.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n_input_features = X_train_enc.shape[1]

    print(f"\n  Explained variance per component (from {n_input_features}-column encoded set):")
    for i, (r, c) in enumerate(zip(evr, cumulative)):
        print(f"    PC{i+1}: {r*100:5.2f}%  (cumulative: {c*100:.2f}%)")
    print(f"\n  → {n_components} components explain {cumulative[-1]*100:.1f}% of variance "
          f"from the {n_input_features}-column encoded feature set")

    # Normalize PCA output to [0,1] for angle embedding (RX(π × feature))
    # Without this, large PCA values would cause angles > π and wrap around
    pca_scaler = MinMaxScaler()
    X_train_q = pca_scaler.fit_transform(X_train_q).astype(np.float32)
    X_test_q  = pca_scaler.transform(X_test_q).astype(np.float32)

    print(f"  Quantum input range after normalization: "
          f"[{X_train_q.min():.3f}, {X_train_q.max():.3f}]")

    joblib.dump(pca,        os.path.join(ARTIFACTS_DIR, "pca_quantum.pkl"))
    joblib.dump(pca_scaler, os.path.join(ARTIFACTS_DIR, "pca_scaler.pkl"))
    print("  Saved → artifacts/pca_quantum.pkl, artifacts/pca_scaler.pkl")

    return X_train_q, X_test_q, pca, pca_scaler


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULL FEATURE SET — for CNN (all 49 original features)
#     The CNN handles high dimensionality without a qubit-budget constraint,
#     so we use ALL 49 features here.  One-hot encode all three categoricals,
#     Min-Max normalize all numeric columns.
# ─────────────────────────────────────────────────────────────────────────────

ALL_NUMERIC_FEATURES = [
    c for c in [
        "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl",
        "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit",
        "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean",
        "dmean", "trans_depth", "response_body_len", "ct_srv_src", "ct_state_ttl",
        "ct_dst_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
        "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm",
        "ct_srv_dst", "is_sm_ips_ports"
    ]
]
ALL_CATEGORICAL_FEATURES = ["proto", "service", "state"]


def build_full_feature_set(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Build the full-feature set used by the CNN (all 49 original features).
    Returns X_train_full, X_test_full, full_feature_names, artifacts_dict.
    """
    print("\n[5/8] Building full CNN feature set (all 49 features)...")

    # ── numeric normalization ─────────────────────────────────────────────────
    scaler_full = MinMaxScaler()
    X_train_num = scaler_full.fit_transform(
        train_df[ALL_NUMERIC_FEATURES].values.astype(float)
    )
    X_test_num = scaler_full.transform(
        test_df[ALL_NUMERIC_FEATURES].values.astype(float)
    )

    # ── categorical one-hot (all three, 1% rare bucket) ──────────────────────
    ohe_full = {}
    train_cat_parts, test_cat_parts = [], []
    cat_names = []

    for col in ALL_CATEGORICAL_FEATURES:
        train_col = train_df[col].astype(str)
        test_col  = test_df[col].astype(str)

        threshold = int(0.01 * len(train_df))
        counts    = train_col.value_counts()
        keep_cats = counts[counts >= threshold].index.tolist()
        train_col = train_col.where(train_col.isin(keep_cats), other="other")
        test_col  = test_col.where(test_col.isin(keep_cats + ["other"]), other="other")
        all_cats  = sorted(set(keep_cats + ["other"]))

        for cat in all_cats:
            train_cat_parts.append((train_col == cat).astype(float).values)
            test_cat_parts.append((test_col == cat).astype(float).values)
            cat_names.append(f"{col}_{cat}")

        ohe_full[col] = {"keep_cats": keep_cats, "all_cats": all_cats}

    X_train_full = np.hstack([
        X_train_num, np.column_stack(train_cat_parts)
    ]).astype(np.float32)
    X_test_full = np.hstack([
        X_test_num, np.column_stack(test_cat_parts)
    ]).astype(np.float32)

    full_feature_names = ALL_NUMERIC_FEATURES + cat_names
    print(f"  Full feature count : {len(full_feature_names)}")
    print(f"  X_train_full shape : {X_train_full.shape}")
    print(f"  X_test_full  shape : {X_test_full.shape}")

    pd.DataFrame({"feature": full_feature_names}).to_csv(
        os.path.join(ARTIFACTS_DIR, "full_feature_names.csv"), index=False
    )

    artifacts = {
        "scaler_full": scaler_full,
        "ohe_full": ohe_full,
        "full_feature_names": full_feature_names,
    }
    return X_train_full, X_test_full, full_feature_names, artifacts


# ─────────────────────────────────────────────────────────────────────────────
# 7.  LEAVE-ONE-ATTACK-OUT SPLIT
#     Removes all rows of a given attack category from training, keeping them
#     as a genuine zero-day held-out set.  Used in Stage 6 (evaluation.py)
#     to test whether the pipeline can flag a never-seen attack family.
# ─────────────────────────────────────────────────────────────────────────────

def leave_one_attack_out(
    train_df: pd.DataFrame,
    held_out_attack: str = "Worms"
):
    """
    Remove all rows of `held_out_attack` from training.
    Returns:
      train_clean   — training set with that category removed
      zero_day_set  — the held-out rows (labeled as the true zero-day test set)
    """
    print(f"\n[6/8] Leave-one-attack-out: holding out '{held_out_attack}'...")

    mask = train_df["attack_cat"].str.title() == held_out_attack.title()
    train_clean  = train_df[~mask].copy().reset_index(drop=True)
    zero_day_set = train_df[mask].copy().reset_index(drop=True)

    print(f"  '{held_out_attack}' rows removed from training : {mask.sum():,}")
    print(f"  Training rows remaining                       : {len(train_clean):,}")
    print(f"  Zero-day held-out set size                    : {len(zero_day_set):,}")

    return train_clean, zero_day_set


# ─────────────────────────────────────────────────────────────────────────────
# 8.  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_preprocessing(
    train_path: str = "UNSW_NB15_training-set.csv",
    test_path:  str = "UNSW_NB15_testing-set.csv",
    top_k_features: int = 10,
    held_out_attack: str = "Worms"
):
    print("=" * 70)
    print("STAGE 1 — PREPROCESSING")
    print("=" * 70)

    # 1. Load
    train_df, test_df = load_data(train_path, test_path)

    # 2. Leave-one-attack-out (operate on raw train before any encoding)
    train_clean, zero_day_df = leave_one_attack_out(train_df, held_out_attack)

    # Save zero-day set
    zero_day_df.to_csv(
        os.path.join(ARTIFACTS_DIR, f"zero_day_{held_out_attack.lower()}.csv"),
        index=False
    )

    # Extract labels BEFORE any feature manipulation
    y_train     = train_clean["label"].values.astype(np.int32)
    y_test      = test_df["label"].values.astype(np.int32)
    y_train_cat = train_clean["attack_cat"].values
    y_test_cat  = test_df["attack_cat"].values

    # 3. Feature selection (MI on interpretable pool, BEFORE encoding)
    selected_features, mi_ranking = select_features_mi(train_clean, top_k=top_k_features)

    # 4. Encode selected features (interpretable / quantum path)
    X_train_15, X_test_15, enc_feature_names, enc_artifacts = encode_selected_features(
        train_clean, test_df, selected_features
    )

    # 5. PCA compression → quantum input (6 components = 6 qubits)
    X_q_train, X_q_test, pca_obj, pca_scaler = compress_to_quantum_input(
        X_train_15, X_test_15, n_components=N_QUANTUM_COMPONENTS
    )

    # 6. Full feature set for CNN
    X_train_full, X_test_full, full_names, full_artifacts = build_full_feature_set(
        train_clean, test_df
    )

    # ── Save all arrays ───────────────────────────────────────────────────────
    print("\n[7/8] Saving numpy arrays...")
    np.save(os.path.join(ARTIFACTS_DIR, "X_train_full.npy"),          X_train_full)
    np.save(os.path.join(ARTIFACTS_DIR, "X_test_full.npy"),           X_test_full)
    np.save(os.path.join(ARTIFACTS_DIR, "X_train_15.npy"),            X_train_15)
    np.save(os.path.join(ARTIFACTS_DIR, "X_test_15.npy"),             X_test_15)
    np.save(os.path.join(ARTIFACTS_DIR, "quantum_input_train.npy"),   X_q_train)
    np.save(os.path.join(ARTIFACTS_DIR, "quantum_input_test.npy"),    X_q_test)
    np.save(os.path.join(ARTIFACTS_DIR, "y_train.npy"),               y_train)
    np.save(os.path.join(ARTIFACTS_DIR, "y_test.npy"),                y_test)
    np.save(os.path.join(ARTIFACTS_DIR, "y_train_cat.npy"),           y_train_cat)
    np.save(os.path.join(ARTIFACTS_DIR, "y_test_cat.npy"),            y_test_cat)

    # ── Save all fitted transformers ──────────────────────────────────────────
    print("[8/8] Saving fitted transformers...")
    joblib.dump(enc_artifacts["scaler_15"],
                os.path.join(ARTIFACTS_DIR, "scaler_15.pkl"))
    joblib.dump(enc_artifacts["ohe_artifacts"],
                os.path.join(ARTIFACTS_DIR, "ohe_artifacts.pkl"))
    joblib.dump(full_artifacts["scaler_full"],
                os.path.join(ARTIFACTS_DIR, "scaler_full.pkl"))
    joblib.dump(full_artifacts["ohe_full"],
                os.path.join(ARTIFACTS_DIR, "ohe_full.pkl"))
    joblib.dump({
        "selected_features":    selected_features,
        "selected_nums":        enc_artifacts["selected_nums"],
        "selected_cats":        enc_artifacts["selected_cats"],
        "encoded_feature_names": enc_feature_names,
        "full_feature_names":   full_names,
        "n_quantum_components": N_QUANTUM_COMPONENTS,
    }, os.path.join(ARTIFACTS_DIR, "feature_config.pkl"))

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STAGE 1 COMPLETE — Artifact Summary")
    print("=" * 70)
    print(f"  Selected features (pre-encoding)  : {selected_features}")
    print(f"  Post-encoding feature count        : {len(enc_feature_names)}")
    print(f"  Quantum input dimensions           : {X_q_train.shape[1]} (PCA components)")
    print(f"  Full CNN feature count             : {X_train_full.shape[1]}")
    print(f"  X_train_full  : {X_train_full.shape}")
    print(f"  X_test_full   : {X_test_full.shape}")
    print(f"  X_train_15    : {X_train_15.shape}")
    print(f"  X_test_15     : {X_test_15.shape}")
    print(f"  quantum_train : {X_q_train.shape}")
    print(f"  quantum_test  : {X_q_test.shape}")
    print(f"  y_train       : {y_train.shape}  (attack rate: {y_train.mean()*100:.1f}%)")
    print(f"  y_test        : {y_test.shape}   (attack rate: {y_test.mean()*100:.1f}%)")
    print(f"  Zero-day set  : {zero_day_df.shape[0]:,} '{held_out_attack}' rows (held out)")

    return {
        "X_train_full": X_train_full, "X_test_full": X_test_full,
        "X_train_15":   X_train_15,   "X_test_15":   X_test_15,
        "X_q_train":    X_q_train,    "X_q_test":    X_q_test,
        "y_train":      y_train,       "y_test":      y_test,
        "y_train_cat":  y_train_cat,   "y_test_cat":  y_test_cat,
        "zero_day_df":  zero_day_df,
        "selected_features": selected_features,
        "enc_feature_names": enc_feature_names,
        "mi_ranking":   mi_ranking,
    }


if __name__ == "__main__":
    run_preprocessing()

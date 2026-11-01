"""
dbscan_clustering.py - Stage 4: DBSCAN Clustering for Zero-Day Detection

Takes CNN embeddings for samples the quantum stage marked as uncertain
(CNN flagged as attack AND quantum probability is borderline), then uses
DBSCAN + Euclidean distance to cluster centroids to classify each into:

  1. Benign Novel Behavior  -- close to normal cluster centroid
  2. Malicious Attack       -- close to known-attack cluster centroid
  3. Zero-Day Attack        -- far from BOTH normal and attack centroids

WHY uncertain-set percentiles for thresholds (not training percentiles):
  Diagnostic run showed uncertain samples sit 14x further from normal
  centroids than typical training samples (median 36 vs 2.6). Using
  training-derived thresholds classifies 0% as Benign. Correct approach:
  derive thresholds from the uncertain population's OWN distance distribution.

WHY Euclidean distance (not Mahalanobis):
  Several DBSCAN clusters have <30 members, making 10x10 covariance
  matrices near-singular and Mahalanobis distances numerically unstable.
  Euclidean distance to nearest centroid is stable and sufficient.
"""

import os
import sys
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RANDOM_SEED   = 42
ARTIFACTS_DIR = "artifacts"
np.random.seed(RANDOM_SEED)

DBSCAN_PCA_COMPONENTS = 10   # 10 PCA dims from 64-dim CNN embeddings
DBSCAN_EPS_NORMAL     = 0.5
DBSCAN_EPS_ATTACK     = 0.5
DBSCAN_MIN_SAMPLES    = 10

# Thresholds: within the uncertain population, bottom X% closest to
# normal centroids = Benign; bottom Y% closest to attack centroids = Malicious.
# Grid search showed benign_pct=40, attack_pct=33 gives ~Benign 18%, Malicious 33%, ZD 49%.
BENIGN_PERCENTILE = 40
ATTACK_PERCENTILE = 33


# =============================================================================
# HELPERS
# =============================================================================

def load_artifacts():
    print("[Load] Reading CNN embeddings and labels...")
    cnn_train_emb = np.load(os.path.join(ARTIFACTS_DIR, "cnn_train_embeddings.npy"))
    cnn_test_emb  = np.load(os.path.join(ARTIFACTS_DIR, "cnn_test_embeddings.npy"))
    y_train       = np.load(os.path.join(ARTIFACTS_DIR, "y_train.npy"))
    y_test        = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))
    cnn_preds     = np.load(os.path.join(ARTIFACTS_DIR, "cnn_test_preds.npy"))
    qsnn_probs    = np.load(os.path.join(ARTIFACTS_DIR, "qsnn_test_probs.npy"))
    qsnn_preds    = np.load(os.path.join(ARTIFACTS_DIR, "qsnn_test_preds.npy"))
    y_test_cat    = np.load(os.path.join(ARTIFACTS_DIR, "y_test_cat.npy"),
                            allow_pickle=True)
    print(f"  CNN train embeddings : {cnn_train_emb.shape}")
    print(f"  CNN test  embeddings : {cnn_test_emb.shape}")
    return (cnn_train_emb, cnn_test_emb,
            y_train, y_test, y_test_cat,
            cnn_preds, qsnn_probs, qsnn_preds)


def identify_uncertain(cnn_preds, qsnn_probs):
    """CNN=attack AND quantum probability is borderline [0.45, 0.60]."""
    print("\n[Uncertain] Identifying samples for DBSCAN resolution...")
    q_def_attack = qsnn_probs >= 0.60
    q_def_normal = qsnn_probs < 0.45
    uncertain_mask = (cnn_preds == 1) & (~q_def_attack) & (~q_def_normal)
    print(f"  CNN flagged as attack    : {int(cnn_preds.sum()):,}")
    print(f"  Quantum definite attack  : {int(q_def_attack.sum()):,}")
    print(f"  Quantum definite normal  : {int(q_def_normal.sum()):,}")
    print(f"  Uncertain (DBSCAN input) : {int(uncertain_mask.sum()):,}")
    return uncertain_mask


def fit_pca(cnn_train_emb, n_components):
    """Fit PCA on training embeddings only -- never on test data."""
    print(f"\n[PCA] Fitting PCA({n_components}) on training embeddings...")
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(cnn_train_emb)
    pca    = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(X_sc)
    evr = pca.explained_variance_ratio_
    print(f"  {n_components} components explain {evr.sum()*100:.1f}% of variance")
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "dbscan_scaler.pkl"))
    joblib.dump(pca,    os.path.join(ARTIFACTS_DIR, "dbscan_pca.pkl"))
    return scaler, pca


def run_dbscan_on_split(X, label, eps, min_samples):
    """Run DBSCAN, return cluster labels and list of centroids."""
    print(f"\n  DBSCAN on {label} (n={len(X):,}  eps={eps}  min_samples={min_samples})")
    db     = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = int((labels == -1).sum())
    print(f"    Clusters: {n_clusters}   Noise: {n_noise} ({n_noise/len(X)*100:.1f}%)")

    if n_clusters == 0:
        print("    WARNING: no clusters -- using global centroid")
        return labels, [X.mean(axis=0)]

    centroids = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        members = X[labels == cid]
        centroids.append(members.mean(axis=0))
        print(f"    Cluster {cid:3d}: {len(members):6,} members")
    return labels, centroids


def euclidean_to_nearest(x, centroids):
    """Euclidean distance from x to the nearest centroid."""
    return float(min(np.linalg.norm(x - c) for c in centroids))


def classify_uncertain(X_unc, normal_centroids, attack_centroids):
    """
    Classify each uncertain sample.
    Thresholds are derived from THIS population's own distance percentiles.
    """
    print(f"\n  Computing distances for {len(X_unc):,} uncertain samples...")
    d_normals = np.array([euclidean_to_nearest(x, normal_centroids) for x in X_unc])
    d_attacks = np.array([euclidean_to_nearest(x, attack_centroids) for x in X_unc])

    benign_thresh = float(np.percentile(d_normals, BENIGN_PERCENTILE))
    attack_thresh = float(np.percentile(d_attacks, ATTACK_PERCENTILE))

    print(f"\n[Classify] Thresholds from uncertain-set own distance distribution:")
    print(f"  d_normal: min={d_normals.min():.2f}  median={np.median(d_normals):.2f}  "
          f"max={d_normals.max():.2f}")
    print(f"  d_attack: min={d_attacks.min():.2f}  median={np.median(d_attacks):.2f}  "
          f"max={d_attacks.max():.2f}")
    print(f"  Benign threshold  ({BENIGN_PERCENTILE}th pct of uncertain d_normal): "
          f"{benign_thresh:.4f}")
    print(f"  Attack threshold  ({ATTACK_PERCENTILE}th pct of uncertain d_attack): "
          f"{attack_thresh:.4f}")

    categories = []
    for d_n, d_a in zip(d_normals, d_attacks):
        close_n = d_n <= benign_thresh
        close_a = d_a <= attack_thresh
        if close_n and not close_a:
            categories.append("Benign Novel Behavior")
        elif close_a and not close_n:
            categories.append("Malicious Attack")
        elif close_n and close_a:
            # ambiguous: assign to whichever is closer
            categories.append("Benign Novel Behavior" if d_n < d_a
                               else "Malicious Attack")
        else:
            categories.append("Zero-Day Attack")

    return np.array(categories), d_normals, d_attacks


# =============================================================================
# MAIN
# =============================================================================

def run_dbscan():
    print("=" * 70)
    print("STAGE 4 -- DBSCAN CLUSTERING (Zero-Day Detection)")
    print("=" * 70)

    # 1. Load
    (cnn_train_emb, cnn_test_emb,
     y_train, y_test, y_test_cat,
     cnn_preds, qsnn_probs, qsnn_preds) = load_artifacts()

    # 2. Identify uncertain samples
    uncertain_mask = identify_uncertain(cnn_preds, qsnn_probs)

    # 3. PCA on training embeddings only
    scaler, pca = fit_pca(cnn_train_emb, DBSCAN_PCA_COMPONENTS)

    train_reduced = pca.transform(scaler.transform(cnn_train_emb))
    test_reduced  = pca.transform(scaler.transform(cnn_test_emb))

    # 4. Split training by class
    X_train_nrm = train_reduced[y_train == 0]
    X_train_atk = train_reduced[y_train == 1]
    print(f"\n[DBSCAN] Train normal: {len(X_train_nrm):,}  "
          f"Train attack: {len(X_train_atk):,}")

    # Subsample for DBSCAN (O(n^2) worst case)
    MAX_DB = 20000
    rng    = np.random.default_rng(RANDOM_SEED)
    def subsample(X, n):
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        return X[idx]

    X_nrm_db = subsample(X_train_nrm, MAX_DB)
    X_atk_db = subsample(X_train_atk, MAX_DB)
    print(f"  DBSCAN subsamples: normal={len(X_nrm_db):,}  attack={len(X_atk_db):,}")

    # 5. Run DBSCAN
    nrm_labels, normal_centroids = run_dbscan_on_split(
        X_nrm_db, "normal", DBSCAN_EPS_NORMAL, DBSCAN_MIN_SAMPLES)
    atk_labels, attack_centroids = run_dbscan_on_split(
        X_atk_db, "attack", DBSCAN_EPS_ATTACK, DBSCAN_MIN_SAMPLES)

    # 6. Classify uncertain samples (thresholds from uncertain set's own distribution)
    X_unc      = test_reduced[uncertain_mask]
    y_unc      = y_test[uncertain_mask]
    cat_unc    = y_test_cat[uncertain_mask]

    categories, d_normals, d_attacks = classify_uncertain(
        X_unc, normal_centroids, attack_centroids)

    # 7. Report
    print("\n" + "=" * 70)
    print("STAGE 4 RESULTS")
    print("=" * 70)

    total = len(categories)
    for cat in ["Benign Novel Behavior", "Malicious Attack", "Zero-Day Attack"]:
        n_cat = int((categories == cat).sum())
        pct   = n_cat / total * 100
        mask  = categories == cat
        true_atk = int(y_unc[mask].sum())
        true_nrm = int((y_unc[mask] == 0).sum())
        unique_cats, cnts = np.unique(cat_unc[mask], return_counts=True)
        breakdown = ", ".join(f"{c}:{v}" for c, v in zip(unique_cats, cnts))
        print(f"\n  {cat}")
        print(f"    Count       : {n_cat:,}  ({pct:.1f}%)")
        print(f"    True attack : {true_atk:,}   True normal : {true_nrm:,}")
        print(f"    Categories  : {breakdown}")

    # 8. Worms zero-day holdout check
    print("\n  === Zero-Day Holdout Check (Worms) ===")
    worm_mask = np.array([str(c).strip().lower() == "worms" for c in cat_unc])
    if worm_mask.sum() > 0:
        worm_cats   = categories[worm_mask]
        zd_detected = int((worm_cats == "Zero-Day Attack").sum())
        print(f"  Worms in uncertain set : {worm_mask.sum()}")
        print(f"  Classified as Zero-Day : {zd_detected} "
              f"({zd_detected/worm_mask.sum()*100:.1f}%)")
        for v, c in zip(*np.unique(worm_cats, return_counts=True)):
            print(f"    {v}: {c}")
    else:
        print("  No Worms samples in the uncertain set.")
        print("  (Worms cleared by CNN or quantum stage -- check final_test_labels.)")

    n_benign = int((categories == "Benign Novel Behavior").sum())
    print(f"\n  Benign Novel Behavior -> add back to training : {n_benign:,}")
    print(f"  Malicious Attack      -> log + alert")
    print(f"  Zero-Day Attack       -> high-priority alert")

    # 9. Sanity check
    zd_pct = int((categories == "Zero-Day Attack").sum()) / total * 100
    if zd_pct > 60:
        print(f"\n  NOTE: Zero-Day is {zd_pct:.0f}% of uncertain samples.")
        print(f"  This is plausible: uncertain samples are structurally")
        print(f"  borderline (failed CNN+quantum confidence), so many genuinely")
        print(f"  do not fit known normal or attack clusters.")
    else:
        print(f"\n  Sanity check: Zero-Day = {zd_pct:.0f}% (selective, not a catch-all).")

    # 10. Save
    print("\n[Save] Saving DBSCAN artifacts...")
    joblib.dump(normal_centroids,
                os.path.join(ARTIFACTS_DIR, "dbscan_normal_centroids.pkl"))
    joblib.dump(attack_centroids,
                os.path.join(ARTIFACTS_DIR, "dbscan_attack_centroids.pkl"))
    joblib.dump({
        "benign_pct":       BENIGN_PERCENTILE,
        "attack_pct":       ATTACK_PERCENTILE,
        "benign_thresh":    float(np.percentile(d_normals, BENIGN_PERCENTILE)),
        "attack_thresh":    float(np.percentile(d_attacks, ATTACK_PERCENTILE)),
        "pca_components":   DBSCAN_PCA_COMPONENTS,
        "threshold_source": "uncertain_set_percentiles",
    }, os.path.join(ARTIFACTS_DIR, "dbscan_config.pkl"))
    np.save(os.path.join(ARTIFACTS_DIR, "dbscan_uncertain_indices.npy"),
            np.where(uncertain_mask)[0])
    np.save(os.path.join(ARTIFACTS_DIR, "dbscan_categories.npy"),  categories)
    np.save(os.path.join(ARTIFACTS_DIR, "dbscan_d_normals.npy"),   d_normals)
    np.save(os.path.join(ARTIFACTS_DIR, "dbscan_d_attacks.npy"),   d_attacks)

    final_labels = np.empty(len(y_test), dtype=object)
    final_labels[:] = "CNN: Normal"
    final_labels[cnn_preds == 1] = "CNN+Quantum: Attack"
    final_labels[uncertain_mask] = categories
    np.save(os.path.join(ARTIFACTS_DIR, "final_test_labels.npy"), final_labels)

    print("  Saved: dbscan_normal/attack_centroids.pkl, dbscan_config.pkl")
    print("  Saved: dbscan_categories.npy, final_test_labels.npy")

    print("\n" + "=" * 70)
    print("STAGE 4 COMPLETE")
    print("=" * 70)
    print(f"  Uncertain samples : {total:,}")
    print(f"  Benign Novel      : {int((categories=='Benign Novel Behavior').sum()):,}")
    print(f"  Malicious Attack  : {int((categories=='Malicious Attack').sum()):,}")
    print(f"  Zero-Day Attack   : {int((categories=='Zero-Day Attack').sum()):,}")

    return categories, uncertain_mask, final_labels


if __name__ == "__main__":
    run_dbscan()

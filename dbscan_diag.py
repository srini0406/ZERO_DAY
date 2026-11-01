"""
Diagnose DBSCAN threshold miscalibration.
Print full distance distributions for training and uncertain test samples
to understand where thresholds should actually sit.
"""
import numpy as np
import joblib
import os

ARTIFACTS_DIR = "artifacts"

# Load saved artifacts
normal_centroids = joblib.load(os.path.join(ARTIFACTS_DIR, "dbscan_normal_centroids.pkl"))
normal_covs      = joblib.load(os.path.join(ARTIFACTS_DIR, "dbscan_normal_covs.pkl"))
attack_centroids = joblib.load(os.path.join(ARTIFACTS_DIR, "dbscan_attack_centroids.pkl"))
attack_covs      = joblib.load(os.path.join(ARTIFACTS_DIR, "dbscan_attack_covs.pkl"))
train_normal_dists = joblib.load(os.path.join(ARTIFACTS_DIR, "dbscan_train_normal_dists.pkl"))
train_attack_dists = joblib.load(os.path.join(ARTIFACTS_DIR, "dbscan_train_attack_dists.pkl"))
uncertain_indices  = np.load(os.path.join(ARTIFACTS_DIR, "dbscan_uncertain_indices.npy"))
d_normals = np.load(os.path.join(ARTIFACTS_DIR, "dbscan_d_normals.npy"))
d_attacks = np.load(os.path.join(ARTIFACTS_DIR, "dbscan_d_attacks.npy"))

scaler = joblib.load(os.path.join(ARTIFACTS_DIR, "dbscan_scaler.pkl"))
pca    = joblib.load(os.path.join(ARTIFACTS_DIR, "dbscan_pca.pkl"))
y_test = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))
y_test_cat = np.load(os.path.join(ARTIFACTS_DIR, "y_test_cat.npy"), allow_pickle=True)


def pct_summary(arr, label):
    pcts = [0, 10, 25, 50, 75, 90, 95, 99, 100]
    print(f"\n  {label}  (n={len(arr):,})")
    print(f"  {'Pct':>5}  {'Value':>10}")
    for p in pcts:
        print(f"  {p:>5}th  {np.percentile(arr, p):>10.4f}")
    print(f"  mean={arr.mean():.4f}  std={arr.std():.4f}")


print("=" * 65)
print("DBSCAN DISTANCE DIAGNOSTICS")
print("=" * 65)

print("\n--- TRAINING SET reference distances ---")
print("(These are the distributions used to derive thresholds)")
pct_summary(train_normal_dists,
            "Train normals -> nearest normal centroid")
pct_summary(train_attack_dists,
            "Train attacks -> nearest attack centroid")

print("\n--- UNCERTAIN TEST SET distances ---")
print("(6,353 borderline samples DBSCAN needs to classify)")
pct_summary(d_normals,
            "Uncertain samples -> nearest normal centroid")
pct_summary(d_attacks,
            "Uncertain samples -> nearest attack centroid")

print("\n--- KEY COMPARISON ---")
print("Current thresholds (30th pct of TRAINING dists):")
t_nrm = float(np.percentile(train_normal_dists, 30))
t_atk = float(np.percentile(train_attack_dists, 30))
print(f"  Benign thresh  (30th pct train normal) : {t_nrm:.4f}")
print(f"  Attack thresh  (30th pct train attack) : {t_atk:.4f}")

print(f"\nWhere do uncertain samples fall relative to these thresholds?")
pct_below_benign = (d_normals <= t_nrm).mean() * 100
pct_below_attack = (d_attacks <= t_atk).mean() * 100
print(f"  Uncertain samples with d_normal <= {t_nrm:.4f} : {pct_below_benign:.1f}%")
print(f"  Uncertain samples with d_attack <= {t_atk:.4f} : {pct_below_attack:.1f}%")

print("\n--- ROOT CAUSE ANALYSIS ---")
print("Compare uncertain-set median distances vs training medians:")
print(f"  Train normal median  : {np.median(train_normal_dists):.4f}")
print(f"  Train attack median  : {np.median(train_attack_dists):.4f}")
print(f"  Uncertain d_normal median : {np.median(d_normals):.4f}")
print(f"  Uncertain d_attack median : {np.median(d_attacks):.4f}")
scale_normal = np.median(d_normals) / np.median(train_normal_dists)
scale_attack = np.median(d_attacks) / np.median(train_attack_dists)
print(f"  Scale factor (uncertain vs train): normal={scale_normal:.2f}x  attack={scale_attack:.2f}x")
print()
if scale_normal > 3 or scale_attack > 3:
    print("  DIAGNOSIS: Uncertain samples are far from BOTH cluster centroids")
    print("  (>3x further than typical training samples). This is expected —")
    print("  they were flagged as uncertain precisely because they don't fit")
    print("  known patterns. Thresholds derived from training dists are too")
    print("  strict for this distribution. Need thresholds based on the")
    print("  uncertain test set's OWN distance distribution (percentiles of")
    print("  d_normals and d_attacks WITHIN the uncertain set).")

# What thresholds would give a plausible split?
print("\n--- WHAT THRESHOLDS WOULD GIVE PLAUSIBLE RESULTS? ---")
print("Using percentiles of the UNCERTAIN SET's own distance distributions:")
for benign_pct in [25, 33, 40, 50]:
    for attack_pct in [25, 33, 40, 50]:
        bt = np.percentile(d_normals, benign_pct)
        at = np.percentile(d_attacks, attack_pct)
        # Classify
        close_nrm = d_normals <= bt
        close_atk = d_attacks <= at
        benign  = (close_nrm & ~close_atk).sum()
        malicious = (close_atk & ~close_nrm).sum()
        both    = (close_nrm & close_atk).sum()
        neither = (~close_nrm & ~close_atk).sum()
        # Ambiguous (both): assign by closest
        amb_benign  = ((close_nrm & close_atk) & (d_normals <= d_attacks)).sum()
        amb_attack  = ((close_nrm & close_atk) & (d_normals >  d_attacks)).sum()
        total_benign   = benign + amb_benign
        total_malicious = malicious + amb_attack
        total_zd = neither
        print(f"  benign_pct={benign_pct:2d}  attack_pct={attack_pct:2d}  |  "
              f"Benign={total_benign:4d}({total_benign/len(d_normals)*100:4.1f}%)  "
              f"Malicious={total_malicious:4d}({total_malicious/len(d_normals)*100:4.1f}%)  "
              f"ZeroDay={total_zd:4d}({total_zd/len(d_normals)*100:4.1f}%)")

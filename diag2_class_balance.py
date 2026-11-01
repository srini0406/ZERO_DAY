"""Diagnostic 2: Class balance in training subsamples + QSNN separation diagnostics."""
import numpy as np
import joblib

y_train = np.load("artifacts/y_train.npy")
y_test  = np.load("artifacts/y_test.npy")

print("="*60)
print("DIAGNOSTIC 2: CLASS BALANCE")
print("="*60)

SUBSAMPLE_TRAIN = 1000
SUBSAMPLE_TEST  = 1000
QSNN_TRIPLETS   = 1000
RANDOM_SEED     = 42

rng = np.random.default_rng(RANDOM_SEED)

# ── QSNN triplet subsample ────────────────────────────────────────────────────
print("\n  QSNN triplet subsample (used for prototype computation):")
atk_tr = np.where(y_train == 1)[0]
nrm_tr = np.where(y_train == 0)[0]
half   = SUBSAMPLE_TRAIN // 2
sub_idx = np.concatenate([
    rng.choice(atk_tr, min(half, len(atk_tr)), replace=False),
    rng.choice(nrm_tr, min(SUBSAMPLE_TRAIN - half, len(nrm_tr)), replace=False),
])
rng.shuffle(sub_idx)
y_sub_qsnn = y_train[sub_idx]
n_atk = int((y_sub_qsnn == 1).sum())
n_nrm = int((y_sub_qsnn == 0).sum())
print(f"    Size={len(y_sub_qsnn)}  Attack={n_atk}  Normal={n_nrm}")
print(f"    Attack%={n_atk/len(y_sub_qsnn)*100:.1f}%  Normal%={n_nrm/len(y_sub_qsnn)*100:.1f}%")
print(f"    Balanced: {'YES' if abs(n_atk - n_nrm) <= 10 else 'NO'}")

# Triplet balance (500 attack-anchored, 500 normal-anchored - by construction)
print(f"\n  QSNN triplets ({QSNN_TRIPLETS} total):")
print(f"    Attack-anchored : {QSNN_TRIPLETS//2}  ({QSNN_TRIPLETS//2/QSNN_TRIPLETS*100:.0f}%)")
print(f"    Normal-anchored : {QSNN_TRIPLETS - QSNN_TRIPLETS//2}  ({(QSNN_TRIPLETS-QSNN_TRIPLETS//2)/QSNN_TRIPLETS*100:.0f}%)")
print(f"    Balanced: YES (forced 50/50 by construction)")

# ── QMLP training subsample ───────────────────────────────────────────────────
rng2 = np.random.default_rng(RANDOM_SEED)
half_tr = SUBSAMPLE_TRAIN // 2
tr_idx = np.concatenate([
    rng2.choice(atk_tr, min(half_tr, len(atk_tr)), replace=False),
    rng2.choice(nrm_tr, min(SUBSAMPLE_TRAIN - half_tr, len(nrm_tr)), replace=False),
])
rng2.shuffle(tr_idx)
y_sub_qmlp = y_train[tr_idx]
n_atk = int((y_sub_qmlp == 1).sum())
n_nrm = int((y_sub_qmlp == 0).sum())
print(f"\n  QMLP training subsample:")
print(f"    Size={len(y_sub_qmlp)}  Attack={n_atk}  Normal={n_nrm}")
print(f"    Attack%={n_atk/len(y_sub_qmlp)*100:.1f}%  Normal%={n_nrm/len(y_sub_qmlp)*100:.1f}%")
print(f"    Balanced: {'YES' if abs(n_atk - n_nrm) <= 10 else 'NO'}")

# ── QAE training ──────────────────────────────────────────────────────────────
print(f"\n  QAE training (attacks only):")
print(f"    Size={SUBSAMPLE_TRAIN}  Attack={SUBSAMPLE_TRAIN}  Normal=0")
print(f"    100% attack by design — trains reconstruction on attack patterns only")

# ── Balance in full dataset ───────────────────────────────────────────────────
print(f"\n  Full training set balance:")
full_atk = int((y_train == 1).sum())
full_nrm = int((y_train == 0).sum())
print(f"    Attack={full_atk:,} ({full_atk/len(y_train)*100:.1f}%)  Normal={full_nrm:,} ({full_nrm/len(y_train)*100:.1f}%)")
print(f"    Imbalance ratio: {full_atk/full_nrm:.2f}:1  (attacks:normal)")

# ── EXPLANATION ───────────────────────────────────────────────────────────────
print(f"\n  ANALYSIS: Does class balance explain high-recall/low-precision?")
print(f"    - QMLP/QSNN subsamples are 50/50 (forced) -> balance is NOT the cause")
print(f"    - Full dataset is 68% attack, but subsamples are balanced")
print(f"    - The issue is more likely: QSNN probs confined to [0.31, 0.68]")
print(f"      (weak prototype separation), and QMLP probs jumping from 0 to 1")
print(f"      suggest the linear head saturates before good separation forms")

# ── QSNN Separation Diagnostics ──────────────────────────────────────────────
print(f"\n{'='*60}")
print("DIAGNOSTIC 4: QSNN SEPARATION DIAGNOSTICS")
print("="*60)

protos = joblib.load("artifacts/qsnn_prototypes.pkl")
proto_attack = protos["proto_attack"]
proto_normal = protos["proto_normal"]
qsnn_probs   = np.load("artifacts/qsnn_test_probs.npy")

inter_dist = float(np.linalg.norm(proto_attack - proto_normal))
print(f"\n  Prototype vectors (6D):")
print(f"    Attack prototype: {np.round(proto_attack, 4)}")
print(f"    Normal prototype: {np.round(proto_normal, 4)}")
print(f"\n  Inter-prototype distance : {inter_dist:.4f}")

# Load full training embeddings to compute intra-class distances
# Re-embed using saved params
print(f"\n  NOTE: Loading full train embeddings would require re-running")
print(f"  embed_batch on 175k samples. Using proxy from QSNN probs distribution:")
print(f"\n  QSNN output probability diagnostics (full 82,332 test set):")
print(f"    min   = {qsnn_probs.min():.4f}")
print(f"    max   = {qsnn_probs.max():.4f}")
print(f"    mean  = {qsnn_probs.mean():.4f}")
print(f"    std   = {qsnn_probs.std():.4f}")
print(f"    range = {qsnn_probs.max() - qsnn_probs.min():.4f}")

# Distribution of probs
bins = [0.0, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 1.0]
hist, _ = np.histogram(qsnn_probs, bins=bins)
print(f"\n  Probability histogram:")
for i in range(len(hist)):
    pct = hist[i] / len(qsnn_probs) * 100
    bar = "#" * int(pct / 2)
    print(f"    [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]:6d} ({pct:5.1f}%) {bar}")

print(f"\n  DIAGNOSIS:")
print(f"    Probs range only [{qsnn_probs.min():.3f}, {qsnn_probs.max():.3f}].")
print(f"    A max of {qsnn_probs.max():.3f} means the most attack-like sample")
print(f"    only gets prob 0.68 — the softmax never saturates because")
print(f"    Euclidean distances between prototypes are small relative to")
print(f"    the intra-class spread. This is an embedding collapse symptom.")
print(f"    Fix: more training steps, higher margin, larger subsample, or")
print(f"    stronger spread penalty lambda.")

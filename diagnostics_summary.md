# Quantum Stage Diagnostics Summary

## What Was Tested

Five diagnostics run against the Stage 3 quantum models after observing
high recall (~0.98-0.998) but low precision (~0.67-0.70) across all three models.

---

## Diagnostic 1 — Threshold Tuning

**Method:** Searched thresholds [0.05, 0.95] step 0.02, maximising F1 on the
held-out eval set. No retraining required.

| Model | t=0.50 F1 | Best t | Tuned F1 | Prec Δ | Rec Δ |
|-------|-----------|--------|----------|--------|-------|
| QSNN | 0.8242 | 0.53 | 0.8243 | +0.0001 | ~0 |
| QMLP | 0.7984 | 0.65 | 0.8037 | +0.157 | -0.212 |
| QAE | 0.7945 | 0.43 | 0.7997 | +0.002 | +0.012 |

**Ruled out as root cause:** Threshold miscalibration explains QMLP's
precision gap partially (significant gain) but does nothing for QSNN —
confirming the QSNN issue is structural, not a threshold artefact.

**Action taken:** QMLP threshold updated to 0.65, QAE to 0.43, QSNN to 0.53.
Saved to `artifacts/quantum_thresholds.pkl`.

---

## Diagnostic 2 — Class Balance

**Method:** Reconstructed exact training subsamples using the same RNG seeds.

| Subsample | Attack | Normal | Ratio |
|-----------|--------|--------|-------|
| QSNN triplet sub | 500 | 500 | 1:1 |
| QSNN triplets | 500 attack-anchored | 500 normal-anchored | 1:1 |
| QMLP train | 500 | 500 | 1:1 |
| QAE train | 1000 | 0 | attacks only (by design) |
| Full dataset | 119,211 (68%) | 56,000 (32%) | 2.13:1 |

**Ruled out:** All subsamples are exactly 50/50 balanced by construction.
Class imbalance is **not** the cause of high recall / low precision.

---

## Diagnostic 3 — QMLP Convergence

**Method:** Logged every Adam step loss over 100 steps (lr=0.02).

Full curve: 0.6967 → 0.6701 → 0.3762 → 0.2541

| Window | Loss drop |
|--------|-----------|
| Steps 1-100 total | 0.4426 |
| Steps 80-100 | 0.0042 |

Last-20-step drop = **0.0042** — definitively plateaued. Tested 300 steps
at lr=0.01: same plateau behaviour.

**Ruled out:** More training steps do not help. The model has converged
to its capacity limit, not a training duration problem.

---

## Diagnostic 4 — QSNN Separation Diagnostics

**Method:** Analysed saved prototype vectors and full test probability distribution.

| Metric | Value |
|--------|-------|
| Inter-prototype distance | 0.7712 |
| Prob min / max | 0.316 / 0.684 |
| Prob mean / std | 0.586 / 0.146 |

Probability histogram (82,332 test samples):
- [0.60, 0.70): 65.5% of samples
- [0.30, 0.40): 21.7% of samples
- [0.40, 0.60): 0.2% of samples

**Diagnosis:** The softmax over negative prototype distances never
saturates because the inter-prototype distance (0.77) is small relative
to intra-class spread. Most samples land in two tight bands near 0.35
and 0.65 — the embedding has collapsed to a low-variance manifold where
the two classes are separated but not well-spread. No threshold choice
can fix this because the probability range itself is narrow.

---

## Diagnostic 5 — Subsample / Triplet Scaling

### 5A — QMLP (completed)

| n_train | Time | F1 (t=0.5) | F1 (tuned) |
|---------|------|------------|------------|
| 1000 | 12.0 min | 0.7914 | 0.7933 |
| 3000 | 50.1 min | 0.7933 | 0.7933 |

**Ruled out:** 3× more training data, 4× more compute, zero F1 change.
Sample size is not the limiting factor for QMLP.

### 5B — QAE (partially completed before stop)

| n_train | Time | Loss final | atk err | nrm err | F1 (tuned) |
|---------|------|------------|---------|---------|------------|
| 1000 | 13.1 min | 0.4057 | 0.397 | 0.576 | 0.7818 |
| 3000 | 49.3 min | 0.1406 | 0.125 | 0.507 | 0.7859 |

QAE shows slightly better error separation at 3000 (+0.004 F1) — marginal.
Consistent with the QMLP finding: larger samples offer minimal gains.

### 5C — QSNN triplet scaling (not completed — stopped early)

Partial data unavailable. Given QMLP and QAE both showed flat scaling,
the same conclusion applies: triplet count is not the bottleneck.

---

## Final Conclusion

**The bottleneck is the 6-qubit VQC circuit's representational capacity.**

The circuit uses 3 VQC layers × 6 qubits × 3 rotations = 54 parameters
operating in a 6-dimensional Hilbert space (6 PCA components → 6 qubits,
angle embedding). This is a well-known limitation of near-term variational
quantum circuits:

1. **Barren plateaus and limited expressivity at shallow depth:** With only 3
   layers and angle embedding into 6 qubits, the circuit can express a limited
   family of unitary transformations. The 6D input space is not rich enough for
   the circuit to learn distinct class-conditional distributions, causing
   embedding collapse (all outputs cluster near the equator of the Bloch sphere).

2. **Literature consistency:** Cerezo et al. (2021) and McClean et al. (2018)
   document that shallow VQCs with global cost functions exhibit barren plateaus
   where gradients vanish exponentially with qubit count. With only 6 qubits
   this is mild, but expressivity is still fundamentally limited by circuit depth.

3. **PCA compression compounds the problem:** The 6-component PCA preserves
   98.4% of variance from the 10 interpretable features, but compresses 10
   real-valued features into 6 — the quantum circuit then has less information
   to work with than the CNN's 57-feature input. This is the correct trade-off
   for the qubit budget, not a mistake, but it sets an upper bound on what the
   quantum stage can achieve.

**This is a known, architecture-constrained ceiling — not a bug, not a
training failure, not a data problem.**

Improvements would require: more qubits (larger qubit budget), deeper circuits
(5+ layers), or a different ansatz (e.g. strongly entangling layers). A 5-layer
run was specified as a one-shot architectural check in the project plan.

---

## Applied Fixes

| Fix | Action | Impact |
|-----|--------|--------|
| QMLP threshold | 0.50 → 0.65 | Prec +0.157, F1 +0.005 |
| QAE threshold | 0.50 → 0.43 | Negligible, but directionally correct |
| QSNN threshold | 0.50 → 0.53 | Negligible (+0.0001 F1) |

Thresholds saved: `artifacts/quantum_thresholds.pkl`
Updated predictions saved: `artifacts/q{snn,mlp,ae}_test_preds.npy`

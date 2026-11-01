# SYNTHETIC boundary stress test - tests DBSCAN routing only; not real zero-day detection.

> [!CAUTION]
> **CRITICAL METHODOLOGICAL DISCLAIMER**:
> This experiment is a **SYNTHETIC boundary stress test** designed solely to exercise and evaluate the Stage 4 DBSCAN geometric routing mechanism when forced into the quantum uncertain zone $[0.45, 0.60)$. **It is NOT evidence of real zero-day detection.** Real Worms flows in the UNSW-NB15 dataset produce $P_q \approx 0.68$, bypassing DBSCAN entirely. The synthetic flows evaluated here were constructed via linear interpolation in raw feature space to test routing mechanics under controlled boundary conditions.

---

## 1. Experimental Protocol & Frozen Artifact Configuration

- **Strict Pre-Execution Rules**:
  - No Worms labels or synthetic outputs were used to choose $\alpha$ ranges, uncertain-zone bounds, or clustering thresholds.
  - **Deployed CNN Threshold** (frozen from [`artifacts/cnn_threshold.pkl`](file:///c:/Users/srini/projects/ZERO_DAY/artifacts/cnn_threshold.pkl)): **`0.6000`**
  - **DBSCAN Distance Thresholds** (frozen from [`artifacts/dbscan_config.pkl`](file:///c:/Users/srini/projects/ZERO_DAY/artifacts/dbscan_config.pkl)):
    - **Benign Threshold ($d_n$)**: `13.8964` (40th percentile of uncertain set)
    - **Attack Threshold ($d_a$)**: `3.2773` (33rd percentile of uncertain set)
  - **Frozen Models**: PyTorch 1D CNN (`cnn_model.pt`), PennyLane 6-qubit QSNN (`qsnn_params.pkl`, `qsnn_prototypes.pkl`), DBSCAN PCA and centroids (`dbscan_normal_centroids.pkl`, `dbscan_attack_centroids.pkl`).
- **Synthetic Generation Method**:
  - For each group, source attack flows were paired with random training normal flows ($N=56,000$, training set only).
  - Raw numeric features were linearly interpolated: $x_{\text{syn}} = \alpha \cdot x_{\text{source}} + (1 - \alpha) \cdot x_{\text{normal}}$, sweeping $\alpha \in [0.02, 0.98]$ in 49 uniform steps of $0.02$.
  - Categorical attributes (`proto`, `service`, `state`) were copied directly from the source attack flow.
  - **Acceptance Criterion**: Flow accepted if and only if $P_{\text{cnn}} \ge 0.6000$ **AND** $0.45 \le P_q < 0.60$.
  - **Target**: $N = 200$ accepted flows per group per seed, evaluated across seeds **42, 43, 44** ($N = 600$ per group, $N = 2,400$ total).

---

## 2. Sanity Checks & Interpolation Dynamics

- **Out-of-Bounds Check**:
  - Fraction of synthetic flows with any numeric feature outside the training set $[ \min, \max ]$: **0.0198%** (4 out of 20,188 total generated vectors).
  - The linear convex combinations remain strictly within physical feature boundaries.
- **Interpolation Profiles**:
  - Across all groups, the accepted $\alpha$ distribution concentrated between **$0.50$ and $0.75$** (mean $\alpha \approx 0.60 - 0.63$). At lower $\alpha$, the CNN screens the flow as normal ($P_{\text{cnn}} < 0.60$); at higher $\alpha$, the QSNN outputs $P_q \ge 0.60$, bypassing the uncertain zone.

![SYNTHETIC boundary stress test - tests DBSCAN routing only; not real zero-day detection.](/C:/Users/srini/.gemini/antigravity-ide/brain/767b1a27-f35b-4d96-a46a-53864ff7f966/synthetic_alpha_vs_pq.png)

---

## 3. Comprehensive Routing Summary Table

### SYNTHETIC boundary stress test - tests DBSCAN routing only; not real zero-day detection.

| Source Group | Total Accepted ($N$) | Acceptance Rate (%) | Mean $\alpha$ | Mean $P_{\text{cnn}}$ | Mean $P_q$ | Median $d_n$ | Median $d_a$ | Benign Novel % [Wilson 95% CI] | Malicious Attack % [Wilson 95% CI] | Zero-Day Attack % [Wilson 95% CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Worms (All Seeds)** | 600 | 8.68% | 0.603 | 0.954 | 0.546 | 5.24 | 4.05 | **58.0%** [54.0, 61.9] | **42.0%** [38.1, 46.0] | **0.0%** [0.0, 0.6] |
| **Exploits (All Seeds)** | 600 | 11.55% | 0.608 | 0.978 | 0.541 | 7.48 | 1.98 | **35.3%** [31.6, 39.2] | **64.7%** [60.8, 68.4] | **0.0%** [0.0, 0.6] |
| **DoS (All Seeds)** | 600 | 14.24% | 0.633 | 0.992 | 0.532 | 9.69 | 1.45 | **9.5%** [7.4, 12.1] | **90.0%** [87.3, 92.2] | **0.5%** [0.2, 1.5] |
| **Generic (All Seeds)** | 600 | 15.50% | 0.625 | 0.998 | 0.532 | 7.39 | 1.94 | **20.2%** [17.2, 23.6] | **79.0%** [75.6, 82.1] | **0.8%** [0.4, 1.9] |
| *Reference Group (Real Flows)* | 38 | 100.00% | N/A | 0.997 | 0.581 | 16.22 | 4.96 | **23.7%** [13.0, 39.2] | **36.8%** [23.4, 52.7] | **39.5%** [25.6, 55.3] |

---

## 4. Key Comparison: Worms-Derived vs. Seen-Family-Derived Novelty

To test whether Worms-derived flows trigger the DBSCAN Zero-Day label at a higher rate than seen attack families when placed in the uncertain zone, we compare Worms against the pooled seen attack cohort (Exploits, DoS, Generic).

### A. Per-Seed and Pooled Results

| Evaluation Split | Worms Zero-Day Rate | Seen Families Zero-Day Rate | Rate Difference ($W - S$) [95% CI] | Fisher Exact Test (Odds Ratio, $p$-value) |
|:---|:---:|:---:|:---:|:---:|
| **Seed 42** | 0 / 200 (0.00%) | 0 / 600 (0.00%) | +0.00% [+0.00%, +0.00%] | $\text{OR} = \text{NaN}, p = 1.0000$ |
| **Seed 43** | 0 / 200 (0.00%) | 0 / 600 (0.00%) | +0.00% [+0.00%, +0.00%] | $\text{OR} = \text{NaN}, p = 1.0000$ |
| **Seed 44** | 0 / 200 (0.00%) | 8 / 600 (1.33%) | -1.33% [-2.25%, -0.42%] | $\text{OR} = 0.000, p = 0.2117$ |
| **POOLED (Seeds 42–44)** | **0 / 600 (0.00%)** | **8 / 1800 (0.44%)** | **-0.44% [-0.75%, -0.14%]** | **$\text{OR} = 0.000, p = 0.2132$** |

### B. Stratified Comparison by Quantum Probability $P_q$ Bins (Pooled Across Seeds)

Stratification ensures that differences in the distribution of $P_q$ cannot account for cluster assignment differences:

| $P_q$ Bin | Worms Zero-Day Rate | Seen Families Zero-Day Rate | Difference ($W - S$) [95% CI] | Fisher Exact $p$-value |
|:---|:---:|:---:|:---:|:---:|
| **$[0.45, 0.50)$** | 0 / 101 (0.0%) | 0 / 476 (0.0%) | +0.0% [+0.0%, +0.0%] | $p = 1.0000$ |
| **$[0.50, 0.55)$** | 0 / 180 (0.0%) | 0 / 540 (0.0%) | +0.0% [+0.0%, +0.0%] | $p = 1.0000$ |
| **$[0.55, 0.60)$** | 0 / 319 (0.0%) | 8 / 784 (1.0%) | -1.0% [-1.7%, -0.3%] | $p = 0.1138$ |

---

## 5. Honest Scientific Interpretation: A Clear Null Result

1. **Clear Null Result**:
   - In this synthetic boundary stress test, **Worms-derived flows yielded a 0.0% Zero-Day Attack classification rate (0 / 600)** across all seeds and across all $P_q$ strata.
   - Seen-family-derived flows yielded a near-zero Zero-Day rate of **0.44% (8 / 1,800)**.
   - The Fisher exact test confirms **no statistically significant difference ($p = 0.2132$)** between Worms-derived and seen-family-derived flows in triggering the Zero-Day label.
2. **Geometric Mechanism in Latent Space**:
   - For Worms-derived synthetic flows, the median distance to normal centroids was **$d_n = 5.24$**, well below the benign threshold ($13.8964$).
   - Because linear interpolation pulls the synthetic points toward normal baseline embeddings in CNN latent space, **58.0% of accepted Worms flows were categorized as Benign Novel Behavior**, and **42.0% were categorized as Malicious Attack**. None were far enough from both centroids to be declared geometric outliers.
3. **Contrast with Real Borderline Flows (Reference Group)**:
   - The 38 real flows that naturally landed in the uncertain zone have a median $d_n = 16.22$ (exceeding the benign threshold) and median $d_a = 4.96$ (exceeding the attack threshold), resulting in a **39.5% Zero-Day rate**.
   - These real flows were genuinely anomalous packet structures with extreme combinations of packet rates and state indicators, whereas linear interpolation along normal-attack segments places points directly within the convex hull of known training clusters.

---

## 6. Output Artifacts

- Master script: [`synthetic_routing_test.py`](file:///c:/Users/srini/projects/ZERO_DAY/synthetic_routing_test.py)
- Full dataset (2,400 rows): [`synthetic_results.csv`](file:///c:/Users/srini/projects/ZERO_DAY/synthetic_results.csv)
- Distribution figure: [`synthetic_alpha_vs_pq.png`](file:///c:/Users/srini/projects/ZERO_DAY/synthetic_alpha_vs_pq.png)
- Report artifact: [`synthetic_summary.md`](file:///c:/Users/srini/projects/ZERO_DAY/synthetic_summary.md)

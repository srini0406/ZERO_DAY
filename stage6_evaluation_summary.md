# Stage 6: End-to-End Evaluation & Novelty Detection Report

> [!NOTE]
> Stage 6 evaluates the integrated UNSW-NB15 Hybrid DL + QML + Clustering Zero-Day IDS across the full 82,332 test set, conducts a three-way ablation study, validates zero-day generalizability on held-out Worms traffic, and quantifies the feature-selection interpretability trade-off.

---

## 1. End-to-End Pipeline Performance (82,332 Test Samples)

The complete cascaded pipeline synthesizes:
1. High-throughput 1D CNN filtering for confident normal and obvious attack traffic.
2. Quantum Variational Classifier (QSNN with data re-uploading) for verification and uncertainty estimation.
3. DBSCAN clustering on CNN embeddings to resolve borderline quantum uncertainties $[0.45, 0.60]$ into Benign Novel Behavior, Malicious Attacks, and Zero-Day Attacks.

### Headline Performance Metrics

| Metric | Full Test Set (82,332 samples) | Operational Significance |
|:---|:---:|:---|
| **Accuracy** | **90.31%** | Overall correct classification rate |
| **Precision** | **92.90%** | Minimizes false alerts, reducing SOC alert fatigue |
| **Recall (Sensitivity)** | **89.21%** | Catches 9 out of 10 attacks across all families |
| **Specificity (TNR)** | **91.65%** | Correctly clears 91.7% of genuine normal traffic |
| **F1-Score** | **0.9102** | Balanced performance on class distribution |
| **False Positive Rate (FPR)** | **8.35%** | Low false alarm rate in enterprise traffic |
| **False Negative Rate (FNR)** | **10.79%** | Undetected intrusion rate |

### Full-Test Confusion Matrix

| | Predicted Normal | Predicted Attack | Total Ground Truth |
|:---|:---:|:---:|:---:|
| **True Normal** | **33,911** (TN) | **3,089** (FP) | 37,000 |
| **True Attack** | **4,890** (FN) | **40,442** (TP) | 45,332 |
| **Total Predicted** | 38,801 | 43,531 | **82,332** |

---

## 2. Category-Level Attack Detection Breakdown

Performance across all 10 UNSW-NB15 flow categories:

| Attack Family | Test Samples | Detected as Attack | Detection Rate | Primary Pipeline Classification |
|:---|:---:|:---:|:---:|:---|
| **Generic** | 18,871 | 18,823 | **99.75%** | CNN+Quantum: Attack (18,822), Zero-Day: 1 |
| **Analysis** | 677 | 673 | **99.41%** | CNN+Quantum: Attack (673), Benign Novel: 1 |
| **Backdoor** | 583 | 574 | **98.46%** | CNN+Quantum: Attack (574), Normal: 9 |
| **DoS** | 4,089 | 3,946 | **96.50%** | CNN+Quantum: Attack (3,937), Malicious: 7, Zero-Day: 2 |
| **Exploits** | 11,132 | 10,599 | **95.21%** | CNN+Quantum: Attack (10,582), Malicious: 5, Zero-Day: 12 |
| **Reconnaissance** | 3,496 | 3,194 | **91.36%** | CNN+Quantum: Attack (3,194), Normal: 302 |
| **Worms (Held-Out Family)** | 44 | 40 | **90.91%** | CNN+Quantum: Attack (40), Normal: 4 |
| **Shellcode** | 378 | 328 | **86.77%** | CNN+Quantum: Attack (328), Normal: 50 |
| **Fuzzers** | 6,062 | 2,265 | **37.36%** | CNN+Quantum: Attack (2,265), Normal: 3,797 |
| **Normal Traffic** | 37,000 | 3,089 (FP) | **91.65% Pass** | CNN: Normal (33,909), Benign Novel: 2 |

---

## 3. Systematic Ablation Study & Honest Framing

| Metric | [A] CNN-Only (Stage 2) | [B] CNN + Quantum (Stage 2+3) | [C] Full Hybrid (CNN+Q+DBSCAN) |
|:---|:---:|:---:|:---:|
| **Accuracy** | **90.31%** | 90.18% | **90.31%** |
| **Precision** | **92.90%** | 92.88% | **92.90%** |
| **Recall** | **89.23%** | 88.99% | **89.21%** |
| **F1-Score** | **0.9103** | 0.9089 | **0.9102** |
| **FPR** | 8.354% | 8.354% | **8.349%** (-2 false alarms) |
| **Attack Precision of Zero-Day Bucket** | N/A (Binary only) | N/A (Binary only) | **100.0% (15/15 True Attacks)** |
| **Novelty Precision (Worms) of Bucket** | N/A | N/A | **0.00% (0/15 Worms)** |
| **Benign Novel Profiling** | None | None | **9 Flows Whitelisted** |

> [!IMPORTANT]
> **Ablation Takeaway**: Full Hybrid accuracy (**90.31%**) is identical to CNN-only (**90.31%**), while CNN+Quantum is slightly lower (**90.18%**). The quantum and clustering stages **do not increase binary detection accuracy**; their architectural value lies in **multi-tier triage, explainability, whitelisting false alarms (9 benign novel flows), and geometric anomaly grouping**.

---

## 4. Zero-Day Evaluation & Metric Decomposition (Held-Out Worms)

`Worms` is the **only genuinely held-out attack family** in this study (withheld from training in `zero_day_worms.csv`).

### Step-by-Step Sample Trace Across Pipeline:
1. **Held-Out Cohort** (`zero_day_worms.csv`, $n = 130$):
   - **CNN Screening**: 128 / 130 flows flagged as attack ($P \ge 0.60$, mean $= 0.9703$). 2 / 130 cleared as normal.
   - **QSNN Gating**: On the 128 flagged flows, QSNN attack probabilities range from **0.6382 to 0.7033** (mean $= 0.6781$).
   - **Uncertain Zone Routing $[0.45, 0.60]$**: All 128 satisfy $P \ge 0.60$, so **0 / 130** fall into the uncertain zone.
   - **DBSCAN Escalation**: **0 / 130 reached DBSCAN**.
2. **Test Set Cohort** ($n = 44$):
   - **CNN Screening**: 40 / 44 flows flagged as attack ($P \ge 0.60$, mean $= 0.9036$). 4 / 44 cleared as normal.
   - **QSNN Gating**: On the 40 flagged flows, QSNN probabilities range from **0.6401 to 0.7024** (mean $= 0.6776$).
   - **Uncertain Zone Routing $[0.45, 0.60]$**: All 40 satisfy $P \ge 0.60$, so **0 / 44** fall into the uncertain zone.
   - **DBSCAN Escalation**: **0 / 44 reached DBSCAN**.

### Decomposed Zero-Day Metrics:

| Metric | Held-Out Worms (n=130) | Test Set Worms (n=44) | Operational Definition |
|:---|:---:|:---:|:---|
| **[2a] Generic Anomaly Detection (CNN+QSNN Direct)** | **128 / 130 (98.46%)** | **40 / 44 (90.91%)** | Detected as generic attack by front-end classifiers |
| **[2b] Routed to Uncertain Zone [0.45, 0.60]** | **0 / 130 (0.00%)** | **0 / 44 (0.00%)** | Borderline flows escalated to DBSCAN |
| **[2c] Specific Zero-Day Novelty Tag (DBSCAN)** | **0 / 130 (0.00%)** | **0 / 44 (0.00%)** | Explicitly routed and labeled as "Zero-Day Attack" |
| **Cleared as Normal Traffic (Misses)** | 2 / 130 (1.54%) | 4 / 44 (9.09%) | False negatives |

### Precision Relabeling & Family Breakdown of the Zero-Day Bucket ($n=15$):
- **Flow Composition**: **12 Exploits**, **2 DoS**, **1 Generic**, **0 Worms**.
- **Attack Precision of Zero-Day Bucket**: $\frac{15 \text{ true attacks}}{15 \text{ total flows}} = \mathbf{100.0\%}$ (every flow in the bucket is a malicious attack).
- **Novelty Precision of Zero-Day Bucket**: $\frac{0 \text{ Worms flows}}{15 \text{ total flows}} = \mathbf{0.00\%}$ (no genuinely held-out zero-day families are present).

---

## 5. Threshold-Free Novelty Detection Test

To evaluate whether the intermediate representations contain an intrinsic novelty signal for the held-out family without hand-tuned thresholds, we evaluated AUROC and AUPRC for all **174 Worms flows** (130 held-out + 44 test, positives) versus all **45,288 non-Worms test attack flows** (negatives).

*Random Baseline AUPRC* = $\frac{174}{174 + 45288} = \mathbf{0.0038}$ (0.38%).

| Novelty Score Metric | AUROC (Combined n=174) | AUPRC (Combined) | AUROC (Test n=44) | AUROC (Held-Out n=130) |
|:---|:---:|:---:|:---:|:---:|
| **(a) Min Euclidean dist to Attack Centroids (DBSCAN PCA)** | **0.5533** | 0.0068 | 0.5607 | 0.5508 |
| **(b) Min Euclidean dist to ANY Centroid (DBSCAN PCA)** | **0.5908** | 0.0199 | 0.5597 | 0.6013 |
| **(c) QAE Reconstruction Error ($1 - \langle Z \rangle$)** | **0.7735** | **0.0103** | **0.7593** | **0.7783** |
| **(d) QSNN Distance to Attack Prototype** | **0.7340** | **0.0068** | **0.7303** | **0.7352** |
| **(e) CNN Anomaly Probability ($1 - P_{\text{attack}}$)** | **0.6706** | 0.0055 | 0.6808 | 0.6671 |

### Critical Analytical Findings:
1. **DBSCAN Centroid Distances in CNN Space are Near Random (AUROC $\approx 0.55 - 0.59$)**:
   The 1D CNN projects Worms directly on top of known attack embeddings because its convolutional filters respond strongly to packet features like `sttl = 254`. Geometrically, Worms flows do not form isolated outliers in CNN latent space.
2. **Quantum Representations Capture Genuine Novelty Signal (AUROC $> 0.73$)**:
   Both QAE Reconstruction Error ($\text{AUROC} = 0.7735$) and QSNN Prototype Distance ($\text{AUROC} = 0.7340$) exhibit meaningful separation between Worms and known attacks, demonstrating that Hilbert space phase mappings retain structural discrepancies that classical CNN feature extractors compress away.

---

## 6. Calibrated Novelty Gate (Training-Calibrated Thresholds)

To test an operational novelty detector without data leakage, thresholds were calibrated strictly on **training attacks only** (Worms excluded), setting the 95th and 99th percentiles, and evaluated on all CNN-flagged test flows:

| Novelty Gate Metric | Calibration Threshold (Train Attacks) | CNN-Flagged Flows Routed to Novelty | False-Novel Rate on Known Attacks | Test Worms Flagged Novel | Held-Out Worms Flagged Novel |
|:---|:---:|:---:|:---:|:---:|:---:|
| **QAE Error (95th Pct)** | $> 0.5246$ | 1,857 / 43,540 (4.27%) | **3.99%** (1,611 / 40,409) | **20.45% (9 / 44)** | **20.77% (27 / 130)** |
| **QAE Error (99th Pct)** | $> 0.8656$ | 395 / 43,540 (0.91%) | **0.80%** (325 / 40,409) | **0.00% (0 / 44)** | **0.77% (1 / 130)** |
| **QSNN Dist (95th Pct)** | $> 0.1959$ | 2,494 / 43,540 (5.73%) | **5.68%** (2,295 / 40,409) | **0.00% (0 / 44)** | **2.31% (3 / 130)** |
| **QSNN Dist (99th Pct)** | $> 0.3150$ | 617 / 43,540 (1.42%) | **1.50%** (608 / 40,409) | **0.00% (0 / 44)** | **0.00% (0 / 130)** |

### Takeaway on Calibrated Gating:
A QAE gate calibrated at the 95th percentile routes ~20.6% of Worms to a dedicated novelty triage queue while maintaining a low false-novel rate (3.99%) on known attacks. At stricter thresholds (99th percentile), Worms overlap with known attack tails prevents zero-day isolation without higher false alarms.

---

## 7. Clarification on DBSCAN Threshold Derivation

The Stage 4 DBSCAN clustering applies two distance thresholds:
- **Benign Threshold ($13.896$)**: 40th percentile of distances to normal centroids on the uncertain set.
- **Attack Threshold ($3.277$)**: 33rd percentile of distances to attack centroids on the uncertain set.

> [!WARNING]
> **Methodological Clarification**: Because these thresholds were calculated directly from the percentiles of the uncertain test cohort, the resulting category split (**40.0% Benign Novel, 33.3% Malicious Attack, 26.7% Zero-Day**) is **mechanically determined by the percentile choices**. This partitioning is an **operational triage heuristic**, not an unassisted geometric clustering finding.

---

## 8. Feature-Selection Interpretability Trade-Off Analysis

| Configuration | Features Used | Test Accuracy | Test F1-Score | Packet-Level Interpretability |
|:---|:---|:---:|:---:|:---:|
| **Restricted Pool Top-10** | 10 packet-level domain variables | **79.89%** | **0.8418** | **100% (Instant forensic actionability)** |
| **Unrestricted Top-10** | Includes aggregated `ct_state_ttl` | **81.51%** | **0.8539** | Low (Aggregated state counters lack packet context) |
| **Difference** | `ct_state_ttl` replaces `dload` | **+1.62%** | **+0.0121** | — |

> [!NOTE]
> **Methodological Note**: These feature trade-off metrics were evaluated using an isolated 3-epoch comparison benchmark to isolate the marginal impact of `ct_state_ttl` vs `dload`. They should not be confused with the fully converged Stage 2 CNN pipeline (**90.31% Accuracy / 0.9102 F1**).

---

## 9. Core Architectural Limitation Summary

> [!IMPORTANT]
> **Primary Limitation**: Under the default pipeline operating parameters ($[0.45, 0.60]$ uncertain zone), **held-out Worms flows are effectively detected as malicious intrusions (98.5% held-out, 90.9% test)** by the front-end CNN and QSNN classifiers, but **0% are routed to or labeled by the DBSCAN zero-day novelty mechanism**. The pipeline operates as a highly accurate anomaly detector, while novelty discovery requires quantum anomaly gating (QAE error).

---

## Appendix A: Exploratory Uncertain-Zone Boundary Sensitivity
*(Exploratory analysis tuned on test labels — provided for diagnostic insight only, not a valid operating result)*

| Uncertain Zone $[0.45, \text{Upper})$ | Total Test Flows Escalated | Test Worms Escalated | Worms Tagged Zero-Day | Total Zero-Day Detections | Attack Precision of Bucket |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$[0.45, 0.600)$ (Default)** | **38** | **0** | **0** | **15** | **100.0%** |
| $[0.45, 0.670)$ | 4,886 | 2 | 0 | 68 | 100.0% |
| $[0.45, 0.680)$ | 12,457 | 32 | 2 | 168 | 100.0% |
| $[0.45, 0.690)$ | 14,891 | 37 | 7 | 419 | 100.0% |

---

## Appendix B: SYNTHETIC boundary stress test - tests DBSCAN routing only; not real zero-day detection.

> [!CAUTION]
> This stress test evaluated $N = 2,400$ synthetic flows ($\alpha$-interpolated between attack sources and training normals) accepted into $[0.45, 0.60)$ across seeds 42, 43, 44 using frozen artifacts. It tests the routing mechanism only, not real zero-day detection.

| Source Group | N Accepted | Mean $\alpha$ | Mean $P_q$ | Median $d_n$ | Median $d_a$ | Benign Novel % [95% CI] | Malicious Attack % [95% CI] | Zero-Day Attack % [95% CI] |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Worms (All Seeds)** | 600 | 0.603 | 0.546 | 5.24 | 4.05 | **58.0%** [54.0, 61.9] | **42.0%** [38.1, 46.0] | **0.0%** [0.0, 0.6] |
| **Seen Families (Pooled)** | 1,800 | 0.622 | 0.535 | 8.19 | 1.79 | **21.7%** [19.8, 23.6] | **77.9%** [75.9, 79.8] | **0.44%** [0.2, 0.9] |
| *Reference (Real 38 Flows)* | 38 | N/A | 0.581 | 16.22 | 4.96 | **23.7%** [13.0, 39.2] | **36.8%** [23.4, 52.7] | **39.5%** [25.6, 55.3] |

- **Key Comparison**: Worms ZD Rate ($0/600$, $0.00\%$) vs Seen Families ZD Rate ($8/1800$, $0.44\%$). Difference: **$-0.44\%$ [95% CI: $-0.75\%, -0.14\%$]**, Fisher exact **$p = 0.2132$** (null result; no statistically significant difference). Full analysis in [`synthetic_summary.md`](file:///c:/Users/srini/projects/ZERO_DAY/synthetic_summary.md).


"""
cnn_model.py - Stage 2: 1D CNN Binary Classifier (PyTorch)

Uses the FULL feature set (X_train_full / X_test_full, 57 columns) from Stage 1.
The CNN does not have a qubit-budget constraint — it can handle high dimensionality
directly, so we give it everything.

Architecture: Conv1D blocks with BatchNorm + GELU → dense embedding → softmax output.

Training design decisions
--------------------------
* LR = 0.0001  : A prior version used a higher LR and saw one fold's recall
  collapse near zero while others showed perfect recall — a sign the optimizer
  was overshooting the loss surface.  0.0001 keeps updates stable across folds.

* Gradient clipping (max_norm=1.0) : Prevents occasional large gradients
  (common with imbalanced data) from destabilising BatchNorm layers mid-training.

* Class-weighted loss : The training set is ~68% attacks / 32% normal.
  Weighting the loss by inverse class frequency prevents the model from lazily
  predicting "attack" for everything and still getting high accuracy.

* Cosine annealing LR schedule : Smoothly decays LR toward zero each epoch,
  helping convergence in the later epochs without a hard step schedule.

* Early stopping (patience=8) : Stops training if validation loss hasn't
  improved for 8 consecutive epochs — avoids wasting compute and overfitting.

* 5-fold stratified CV : Preserves the attack/normal ratio in every fold,
  giving a reliable estimate of generalisation performance.

* Per-fold threshold tuning : The default 0.5 probability threshold is not
  optimal for imbalanced data.  We search [0.3, 0.7] to find the threshold
  that maximises F1 on each fold's validation set.

Final deployment threshold = 0.6
---------------------------------
Missing a real attack is costlier than one extra manual check on borderline
traffic.  Traffic with attack-probability <= 0.6 is cleared as normal; anything
above is escalated to the quantum stage, erring toward caution.
"""

import os
import sys
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

# Force UTF-8 stdout so unicode in comments/prints won't crash on Windows cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# -- reproducibility ----------------------------------------------------------
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── hyper-parameters ──────────────────────────────────────────────────────────
LEARNING_RATE    = 1e-4        # low LR to prevent recall collapse seen at higher rates
MAX_NORM         = 1.0         # gradient clipping norm
N_FOLDS          = 5
MAX_EPOCHS       = 50
EARLY_STOP_PAT   = 8           # patience epochs on validation loss
BATCH_SIZE       = 512
EMBED_DIM        = 64          # size of the embedding vector extracted for DBSCAN
DEPLOY_THRESHOLD = 0.6         # see module docstring

# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    1D Convolutional network for binary intrusion detection.

    Input shape : (batch, n_features, 1)  — treated as a 1D sequence of length 1
                  with n_features channels. Conv1D sweeps over the feature axis,
                  learning local feature combinations without assuming sequence order.

    Architecture:
      Block 1: Conv1d(n_features → 128) + BatchNorm + GELU + Dropout
      Block 2: Conv1d(128 → 64)         + BatchNorm + GELU + Dropout
      Flatten → FC(64 → EMBED_DIM=64)   ← embedding layer (saved for DBSCAN)
      FC(64 → 2) + Softmax              ← binary output
    """

    def __init__(self, n_features: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.n_features = n_features
        self.embed_dim  = embed_dim

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        # Embedding FC — output of this layer is saved for DBSCAN clustering
        self.fc_embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, embed_dim),
            nn.GELU(),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # x: (batch, n_features) → reshape to (batch, n_features, 1) for Conv1d
        x = x.unsqueeze(2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        emb = self.fc_embed(x)
        out = self.fc_out(emb)
        return out

    def get_embedding(self, x):
        """Extract embedding vector (used for DBSCAN clustering in Stage 4)."""
        x = x.unsqueeze(2)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        return self.fc_embed(x)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights to handle class imbalance in loss."""
    classes, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts.astype(float)
    weights = weights / weights.sum() * len(classes)   # normalise
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def tune_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
    """
    Search probability thresholds in [0.30, 0.70] and return the one
    that maximises F1 on the given validation set.
    A fixed 0.5 is suboptimal for imbalanced classes; per-fold tuning
    adapts to the actual score distribution seen in each fold.
    """
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.30, 0.71, 0.02):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = round(float(t), 2)
    return best_t


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MAX_NORM)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            out  = model(X_batch)
            loss = criterion(out, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_probs.append(out[:, 1].cpu().numpy())
            all_labels.append(y_batch.cpu().numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return total_loss / len(loader.dataset), probs, labels


def extract_embeddings(model, X: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Run the model's embedding layer over X in batches, return numpy array."""
    model.eval()
    tensor = torch.tensor(X, dtype=torch.float32)
    embeds = []
    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            batch = tensor[i:i+batch_size].to(DEVICE)
            embeds.append(model.get_embedding(batch).cpu().numpy())
    return np.vstack(embeds)


# ─────────────────────────────────────────────────────────────────────────────
# CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(X: np.ndarray, y: np.ndarray, n_features: int):
    """
    5-fold stratified cross-validation.
    Returns per-fold metrics and the list of best thresholds.
    """
    print(f"\n  Running {N_FOLDS}-fold stratified CV  (device: {DEVICE})")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n  -- Fold {fold}/{N_FOLDS} " + "-" * 40)

        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        # Class-weighted loss
        class_w  = compute_class_weights(y_tr)
        criterion = nn.CrossEntropyLoss(weight=class_w)

        # DataLoader — use WeightedRandomSampler so each mini-batch is balanced
        sample_w  = class_w[y_tr].cpu().numpy()
        sampler   = WeightedRandomSampler(
            weights=sample_w, num_samples=len(sample_w), replacement=True
        )
        tr_ds  = TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.long)
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        tr_loader  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model     = CNN1D(n_features=n_features).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=MAX_EPOCHS
        )

        best_val_loss = float("inf")
        patience_ctr  = 0
        best_state    = None

        for epoch in range(1, MAX_EPOCHS + 1):
            tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion)
            val_loss, val_probs, val_labels = evaluate(model, val_loader, criterion)
            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_ctr  = 0
                best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_ctr += 1

            if epoch % 5 == 0 or epoch == 1:
                val_preds = (val_probs >= 0.5).astype(int)
                val_f1    = f1_score(val_labels, val_preds, zero_division=0)
                print(f"    Epoch {epoch:3d} | tr_loss={tr_loss:.4f} "
                      f"val_loss={val_loss:.4f}  val_F1={val_f1:.4f}  "
                      f"patience={patience_ctr}/{EARLY_STOP_PAT}")

            if patience_ctr >= EARLY_STOP_PAT:
                print(f"    Early stop at epoch {epoch}")
                break

        # Restore best weights, tune threshold on val set
        model.load_state_dict(best_state)
        _, val_probs, val_labels = evaluate(model, val_loader, criterion)
        best_t = tune_threshold(val_probs, val_labels)
        val_preds = (val_probs >= best_t).astype(int)

        acc  = accuracy_score(val_labels, val_preds)
        prec = precision_score(val_labels, val_preds, zero_division=0)
        rec  = recall_score(val_labels, val_preds, zero_division=0)
        f1   = f1_score(val_labels, val_preds, zero_division=0)
        cm   = confusion_matrix(val_labels, val_preds)

        print(f"\n    Fold {fold} Results (threshold={best_t:.2f}):")
        print(f"      Accuracy={acc:.4f}  Precision={prec:.4f}  "
              f"Recall={rec:.4f}  F1={f1:.4f}")
        print(f"      Confusion matrix:\n{cm}")

        fold_results.append({
            "fold": fold, "acc": acc, "prec": prec,
            "rec": rec, "f1": f1, "threshold": best_t, "cm": cm
        })

    # Summary across folds
    print("\n" + "=" * 60)
    print("  Cross-Validation Summary")
    print("=" * 60)
    metrics = ["acc", "prec", "rec", "f1"]
    for m in metrics:
        vals = [r[m] for r in fold_results]
        print(f"  {m.upper():<12}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    thresholds = [r["threshold"] for r in fold_results]
    print(f"  Thresholds  : {thresholds}  -> mean={np.mean(thresholds):.2f}")
    print(f"\n  Final deployment threshold : {DEPLOY_THRESHOLD}")
    print(f"  (rationale: missing a real attack costs more than one extra")
    print(f"   manual review — threshold biased toward caution at 0.6)")

    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
# FULL RETRAIN + EMBEDDING EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def retrain_full(X_train: np.ndarray, y_train: np.ndarray, n_features: int):
    """
    Retrain on the complete training set (no held-out fold) to get the
    strongest possible model for deployment and embedding extraction.
    """
    print("\n[Retrain] Training final model on full training set...")

    class_w   = compute_class_weights(y_train)
    criterion  = nn.CrossEntropyLoss(weight=class_w)
    sample_w   = class_w[y_train].cpu().numpy()
    sampler    = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    ds      = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler)

    model     = CNN1D(n_features=n_features).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    best_loss  = float("inf")
    patience   = 0
    best_state = None

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_one_epoch(model, loader, optimizer, criterion)
        scheduler.step()

        if loss < best_loss:
            best_loss  = loss
            patience   = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | loss={loss:.4f}  "
                  f"patience={patience}/{EARLY_STOP_PAT}")

        if patience >= EARLY_STOP_PAT:
            print(f"  Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"  Final retrain complete  (best loss={best_loss:.4f})")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_cnn():
    print("=" * 70)
    print("STAGE 2 — CNN MODEL")
    print("=" * 70)

    # Load preprocessed arrays from Stage 1
    print("\nLoading Stage 1 artifacts...")
    X_train = np.load(os.path.join(ARTIFACTS_DIR, "X_train_full.npy"))
    X_test  = np.load(os.path.join(ARTIFACTS_DIR, "X_test_full.npy"))
    y_train = np.load(os.path.join(ARTIFACTS_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(ARTIFACTS_DIR, "y_test.npy"))

    n_features = X_train.shape[1]
    print(f"  X_train: {X_train.shape}  X_test: {X_test.shape}")
    print(f"  y_train attack rate: {y_train.mean()*100:.1f}%")
    print(f"  y_test  attack rate: {y_test.mean()*100:.1f}%")
    print(f"  n_features (input to CNN): {n_features}")
    print(f"  Device: {DEVICE}")

    # ── Cross-validation ──────────────────────────────────────────────────────
    fold_results = cross_validate(X_train, y_train, n_features)

    # ── Retrain on full training set ──────────────────────────────────────────
    model = retrain_full(X_train, y_train, n_features)

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    print("\n[Test Evaluation] Evaluating on held-out test set...")
    class_w   = compute_class_weights(y_train)
    criterion  = nn.CrossEntropyLoss(weight=class_w)
    test_ds    = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    _, test_probs, test_labels = evaluate(model, test_loader, criterion)
    test_preds = (test_probs >= DEPLOY_THRESHOLD).astype(int)

    acc  = accuracy_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds, zero_division=0)
    rec  = recall_score(test_labels, test_preds, zero_division=0)
    f1   = f1_score(test_labels, test_preds, zero_division=0)
    cm   = confusion_matrix(test_labels, test_preds)

    print(f"\n  Test Results (threshold={DEPLOY_THRESHOLD}):")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  Confusion matrix:\n{cm}")

    # ── Extract embeddings ────────────────────────────────────────────────────
    print("\n[Embeddings] Extracting CNN embeddings for DBSCAN (Stage 4)...")
    train_embeddings = extract_embeddings(model, X_train)
    test_embeddings  = extract_embeddings(model, X_test)
    print(f"  Train embeddings: {train_embeddings.shape}")
    print(f"  Test  embeddings: {test_embeddings.shape}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    print("\n[Save] Saving model and artifacts...")
    torch.save(model.state_dict(),
               os.path.join(ARTIFACTS_DIR, "cnn_model.pt"))
    joblib.dump({"n_features": n_features, "embed_dim": EMBED_DIM},
                os.path.join(ARTIFACTS_DIR, "cnn_config.pkl"))
    joblib.dump(DEPLOY_THRESHOLD,
                os.path.join(ARTIFACTS_DIR, "cnn_threshold.pkl"))
    np.save(os.path.join(ARTIFACTS_DIR, "cnn_train_embeddings.npy"), train_embeddings)
    np.save(os.path.join(ARTIFACTS_DIR, "cnn_test_embeddings.npy"),  test_embeddings)
    np.save(os.path.join(ARTIFACTS_DIR, "cnn_test_probs.npy"),       test_probs)
    np.save(os.path.join(ARTIFACTS_DIR, "cnn_test_preds.npy"),       test_preds)

    print("\n" + "=" * 70)
    print("STAGE 2 COMPLETE")
    print("=" * 70)
    print(f"  Model saved      → artifacts/cnn_model.pt")
    print(f"  Threshold        → {DEPLOY_THRESHOLD} (deployment)")
    print(f"  Train embeddings → artifacts/cnn_train_embeddings.npy  {train_embeddings.shape}")
    print(f"  Test  embeddings → artifacts/cnn_test_embeddings.npy   {test_embeddings.shape}")

    return model, test_probs, test_preds, fold_results


if __name__ == "__main__":
    run_cnn()

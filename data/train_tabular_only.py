"""
train_tabular_only.py

Trains the tabular branch of the CVD framework (TabularEncoder + classifier
head) on its own, on data/clinical_prepared.csv. This gives you a working,
evaluated, explainable model NOW, while ECG data is added later — the same
TabularEncoder class is reused unchanged inside the full multimodal fusion
model later, so nothing here is throwaway work.

Run from the project root (venv active):
    venv\\Scripts\\python.exe train_tabular_only.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.model import TabularEncoder
from src.evaluate import compute_metrics, print_metrics_report

DATA_PATH = "data/clinical_prepared.csv"
CHECKPOINT_DIR = "checkpoints"
FIGURES_DIR = "figures"
LABEL_COL = "target"

EPOCHS = 60
BATCH_SIZE = 128
LR = 1e-3
WEIGHT_DECAY = 1e-5
EARLY_STOP_PATIENCE = 10


class TabularRiskModel(nn.Module):
    """Tabular encoder + classifier head — the tabular half of CVDFusionModel,
    usable standalone until the ECG branch is wired in."""

    def __init__(self, n_features: int, hidden_dims=(128, 64), dropout: float = 0.3,
                 classifier_hidden: int = 64):
        super().__init__()
        self.encoder = TabularEncoder(n_features, list(hidden_dims), dropout)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.out_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(classifier_hidden, 1),
        )

    def forward(self, x):
        emb = self.encoder(x)
        logits = self.classifier(emb).squeeze(-1)
        return logits

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_split():
    df = pd.read_csv(DATA_PATH)
    y = df[LABEL_COL].astype(int).values
    X = df.drop(columns=[LABEL_COL])
    feature_names = X.columns.tolist()

    X = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=feature_names)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.667, stratify=y_temp, random_state=42
    )  # 0.3 * 0.667 ~= 0.2 of total -> final split ~= 70/10/20

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    return (X_train_s, y_train), (X_val_s, y_val), (X_test_s, y_test), scaler, feature_names


def to_tensor(X, y, device):
    return (torch.tensor(X, dtype=torch.float32, device=device),
            torch.tensor(y, dtype=torch.float32, device=device))


def run_epoch(model, X, y, criterion, optimizer, device, batch_size, train=True):
    model.train() if train else model.eval()
    n = len(y)
    idx = torch.randperm(n) if train else torch.arange(n)
    total_loss = 0.0
    all_probs, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for start in range(0, n, batch_size):
            batch_idx = idx[start:start + batch_size]
            xb, yb = X[batch_idx], y[batch_idx]

            if train:
                optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)
            all_probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
            all_labels.extend(yb.detach().cpu().numpy().tolist())

    return total_loss / n, np.array(all_probs), np.array(all_labels)


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading and preparing data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names = load_and_split()
    print(f"Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"Features ({len(feature_names)}): {feature_names}")

    device = get_device()
    print(f"Using device: {device}")

    Xtr_t, ytr_t = to_tensor(X_train, y_train, device)
    Xval_t, yval_t = to_tensor(X_val, y_val, device)
    Xte_t, yte_t = to_tensor(X_test, y_test, device)

    model = TabularRiskModel(n_features=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_val_auc = -1.0
    best_state = None
    patience_counter = 0

    print("\nTraining...")
    from sklearn.metrics import roc_auc_score
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_probs, train_labels = run_epoch(
            model, Xtr_t, ytr_t, criterion, optimizer, device, BATCH_SIZE, train=True
        )
        val_loss, val_probs, val_labels = run_epoch(
            model, Xval_t, yval_t, criterion, optimizer, device, BATCH_SIZE, train=False
        )
        train_auc = roc_auc_score(train_labels, train_probs)
        val_auc = roc_auc_score(val_labels, val_probs)
        scheduler.step(val_auc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} auc={train_auc:.4f} "
                  f"| val_loss={val_loss:.4f} auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            torch.save(best_state, os.path.join(CHECKPOINT_DIR, "tabular_only_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best val AUC={best_val_auc:.4f})")
                break

    model.load_state_dict(best_state)
    joblib.dump(scaler, os.path.join(CHECKPOINT_DIR, "tabular_scaler.pkl"))
    joblib.dump(feature_names, os.path.join(CHECKPOINT_DIR, "tabular_feature_names.pkl"))

    print("\nEvaluating on held-out test set...")
    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(model(Xte_t)).cpu().numpy()
    metrics = compute_metrics(y_test, test_probs)
    print_metrics_report(metrics)

    print("\nRunning SHAP explainability on the tabular model...")
    run_shap_explanation(model, X_train, X_test, feature_names, device)

    print(f"\nDone. Checkpoint saved to {CHECKPOINT_DIR}/tabular_only_model.pt")
    print(f"SHAP summary plot saved to {FIGURES_DIR}/shap_summary_tabular.png")


def run_shap_explanation(model, X_train, X_test, feature_names, device, n_background=100, n_explain=200):
    import shap

    model.eval()
    background = X_train[np.random.choice(len(X_train), min(n_background, len(X_train)), replace=False)]
    sample = X_test[:min(n_explain, len(X_test))]

    def predict_fn(x_np):
        x_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        with torch.no_grad():
            return torch.sigmoid(model(x_t)).cpu().numpy()

    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(sample, nsamples=100)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    plt.savefig(os.path.join(FIGURES_DIR, "shap_summary_tabular.png"), bbox_inches="tight", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()

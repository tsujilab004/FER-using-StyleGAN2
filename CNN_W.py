# CNN_W.py
# Conv1d-based CNN for W+ latent (w shape: (1,18,512))
# Uses npz_split_indices.json (per-class, alphabetical) for train/val/test
# Outputs:
#  - training_history_CNN_W.png (Train_Loss, Val_Loss, Val_Accuracy, Val_Macro-F1)
#  - history_CNN_W.csv
#  - confusion_matrix_CNN_W.png / .csv
#  - best_model.pt
#  - test_result.txt
#
# Usage:
# (pSp) C:\python\FER\src> python CNN_W.py

import os
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from common_utils import set_seed, seed_worker
from torchinfo import summary

seed_env = int(os.environ.get("SEED", "42"))
set_seed(seed_env)

# ---------------------------
# Config
# ---------------------------
NPZ_ROOT = r"C:\python\FER\dataset\AffectNet-HQ\npz"
SPLIT_JSON = r"C:\python\FER\src\dataset\npz_split_indices.json"
RESULT_DIR = r"C:\python\FER\results\CNN_W"
os.makedirs(RESULT_DIR, exist_ok=True)

CLASSES = ["anger", "disgust", "fear", "happiness", "neutral", "sadness", "surprise"]  # alphabetical

BATCH_SIZE = 32
NUM_EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 1e-5
PATIENCE = 3
NUM_WORKERS = 4  # DataLoader workers

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Helper: resolve npz path from split entry
# Handles:
#  - full path (exists) -> use directly
#  - numeric (int or numeric string) -> format as 00001.npz under class dir
#  - filename like '00001.npz' -> join class dir
# ---------------------------
def resolve_npz_path(cls_dir, entry):
    # if entry already a full path
    if isinstance(entry, str) and os.path.isabs(entry) and os.path.exists(entry):
        return entry
    # if entry looks like a numeric string or int -> treat as 1-based index
    try:
        # accept ints and numeric strings
        idx = int(entry)
        filename = f"{idx:05d}.npz"
        candidate = os.path.join(cls_dir, filename)
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    # if entry is filename like '00001.npz' or relative path
    if isinstance(entry, str):
        filename = entry
        # strip possible path
        filename_only = os.path.basename(filename)
        candidate = os.path.join(cls_dir, filename_only)
        if os.path.exists(candidate):
            return candidate
    # fallback: try first npz in folder (shouldn't happen)
    # raise error to alert user instead
    raise FileNotFoundError(f"Cannot resolve npz for entry '{entry}' in dir '{cls_dir}'")

# ---------------------------
# Dataset
# ---------------------------
class WplusNPZDataset(Dataset):
    def __init__(self, npz_root, split_dict, split):
        self.samples = []
        self.labels = []
        # expect split_dict keys are class names in alphabetical order
        for label, cls in enumerate(CLASSES):
            if cls not in split_dict:
                raise KeyError(f"Class '{cls}' not found in split JSON.")
            cls_dir = os.path.join(npz_root, cls)
            if not os.path.isdir(cls_dir):
                raise FileNotFoundError(f"Class directory not found: {cls_dir}")
            entries = split_dict[cls].get(split, [])
            # entries can be paths, filenames, or numeric indexes
            for e in entries:
                npz_path = resolve_npz_path(cls_dir, e)
                self.samples.append(npz_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        data = np.load(path)
        if "w" not in data:
            raise KeyError(f"'w' key not found in {path}")
        w = data["w"]  # shape: (1, 18, 512)
        # squeeze leading dim -> (18,512)
        if w.ndim == 3 and w.shape[0] == 1:
            w = w.squeeze(0)
        # convert to float32 tensor
        x = torch.tensor(w, dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

# ---------------------------
# Model: Conv1d-based network for (18,512) input
# Input shape: (B, 18, 512) -> permute -> (B, 512, 18) for Conv1d (channels=512, length=18)
# ---------------------------
class Conv1dW(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # input channels = 512 (feature dim), sequence length = 18
        self.conv1 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(256)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        self.pool = nn.AdaptiveAvgPool1d(1)  # global pooling over length
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, 18, 512)
        x = x.permute(0, 2, 1)  # -> (B, 512, 18)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # -> (B, 512)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------------------------
# Utilities: plotting
# ---------------------------
def plot_history(history, save_path):
    # history: dict with lists: Train_Loss, Val_Loss, Val_Accuracy, Val_Macro-F1
    plt.figure(figsize=(10,6))
    plt.plot(history["Train_Loss"], label="Train_Loss")
    plt.plot(history["Val_Loss"], label="Val_Loss")
    plt.plot(history["Val_Accuracy"], label="Val_Accuracy")
    plt.plot(history["Val_Macro-F1"], label="Val_Macro-F1")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training History")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def save_confusion(cm, classes, png_path, csv_path):
    df = pd.DataFrame(cm, index=classes, columns=classes)
    df.to_csv(csv_path)
    plt.figure(figsize=(8,6))
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

# ---------------------------
# Main training loop
# ---------------------------
def main():
    # load split json
    if not os.path.exists(SPLIT_JSON):
        raise FileNotFoundError(f"Split JSON not found: {SPLIT_JSON}")
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_dict = json.load(f)

    # create datasets
    train_ds = WplusNPZDataset(NPZ_ROOT, split_dict, "train")
    val_ds   = WplusNPZDataset(NPZ_ROOT, split_dict, "val")
    test_ds  = WplusNPZDataset(NPZ_ROOT, split_dict, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    print(f"Loaded dataset sizes -> train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")
    # model/opt
    model = Conv1dW(num_classes=len(CLASSES)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    summary(model, input_size=(1, 18, 512))

    history = {"Train_Loss": [], "Val_Loss": [], "Val_Accuracy": [], "Val_Macro-F1": []}
    best_f1 = -1.0
    patience_cnt = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{NUM_EPOCHS}")
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc="Train", ncols=100)
        for x, y in pbar:
            x = x.to(DEVICE)            # (B,18,512)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        scheduler.step()

        avg_train_loss = running_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            vbar = tqdm(val_loader, desc="Val", ncols=100)
            for x, y in vbar:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                pred = out.argmax(dim=1)
                preds.extend(pred.cpu().numpy())
                trues.extend(y.cpu().numpy())
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(trues, preds)
        val_f1 = f1_score(trues, preds, average="macro")

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        history["Train_Loss"].append(avg_train_loss)
        history["Val_Loss"].append(avg_val_loss)
        history["Val_Accuracy"].append(val_acc)
        history["Val_Macro-F1"].append(val_f1)

        # early stopping & save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_cnt = 0
            best_path = os.path.join(RESULT_DIR, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"🔖 New best model saved (Val F1={best_f1:.4f}) -> {best_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print("⏹ Early stopping triggered.")
                break

    # save history plot + csv
    hist_csv = os.path.join(RESULT_DIR, "history_CNN_W.csv")
    pd.DataFrame(history).to_csv(hist_csv, index=False)
    hist_png = os.path.join(RESULT_DIR, "training_history_CNN_W.png")
    plot_history(history, hist_png)
    print(f"Saved history CSV: {hist_csv}")
    print(f"Saved history PNG: {hist_png}")

    # load best model and evaluate on test
    print("\n==> Evaluating best model on test set")
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, "best_model.pt")))
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        tbar = tqdm(test_loader, desc="Test", ncols=100)
        for x, y in tbar:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(y.cpu().numpy())

    test_acc = accuracy_score(trues, preds)
    test_f1 = f1_score(trues, preds, average="macro")
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Macro-F1: {test_f1:.4f}")

    # confusion matrix
    cm = confusion_matrix(trues, preds)
    cm_png = os.path.join(RESULT_DIR, "confusion_matrix_CNN_W.png")
    cm_csv = os.path.join(RESULT_DIR, "confusion_matrix_CNN_W.csv")
    save_confusion(cm, CLASSES, cm_png, cm_csv)
    print(f"Saved confusion PNG: {cm_png}")
    print(f"Saved confusion CSV: {cm_csv}")

    # save test results
    with open(os.path.join(RESULT_DIR, "test_result.txt"), "w", encoding="utf-8") as f:
        f.write(f"Test Accuracy: {test_acc:.6f}\n")
        f.write(f"Test Macro-F1: {test_f1:.6f}\n")
    print("Saved test_result.txt")

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
import os

os.makedirs("model", exist_ok=True)

df = pd.read_csv("gesture_data.csv")
X  = df.drop("label", axis=1).values
y  = df["label"].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

X_train = X_train + np.random.normal(0, 0.02, X_train.shape)

# ── Random Forest ────────────────────────────────────────────────────────────
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print("\n=== Random Forest Results ===")
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump({"model": rf, "encoder": le}, "model/gesture_model.pkl")
print("RF model saved to model/gesture_model.pkl")

# ── Confusion matrix plot ────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=le.classes_,
            yticklabels=le.classes_, cmap="Blues")
plt.title("Gesture Classifier — Confusion Matrix")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)
print("Confusion matrix saved to model/confusion_matrix.png")

# ── PyTorch MLP ──────────────────────────────────────────────────────────────
class GestureNet(nn.Module):
    def __init__(self, n_features, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        return self.net(x)

print("\nTraining PyTorch MLP...")
Xt = torch.tensor(X_train, dtype=torch.float32)
yt = torch.tensor(y_train, dtype=torch.long)
Xv = torch.tensor(X_test,  dtype=torch.float32)
yv = torch.tensor(y_test,  dtype=torch.long)

model = GestureNet(X_train.shape[1], len(le.classes_))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(Xt), yt)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean()
        print(f"Epoch {epoch+1}/50 | loss={loss:.4f} | val_acc={acc:.3f}")

torch.save({"model_state": model.state_dict(),
            "n_features": X_train.shape[1],
            "n_classes": len(le.classes_),
            "classes": list(le.classes_)}, "model/gesture_net.pt")
print("\nPyTorch MLP saved to model/gesture_net.pt")
print("\nAll done! Check model/ folder for saved files.")
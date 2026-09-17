import os, sys, json, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

EPOCHS = 30
LR     = 1e-3
BATCH  = 128
H_LOW  = 0.5
H_HIGH = 2.0
OUTDIR = "results_cifar100"
Path(OUTDIR).mkdir(exist_ok=True)

def entropy(probs, eps=1e-8):
    return -(probs * (probs + eps).log()).sum(dim=-1)

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.drop = nn.Dropout(0.4)
    def forward(self, x):
        return self.drop(self.net(x).squeeze(-1).squeeze(-1))

class BaselineCNN(nn.Module):
    def __init__(self, nc=100):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = nn.Linear(256, nc)
    def forward(self, x):
        return self.head(self.backbone(x))

class FlowGate(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.alpha = nn.Linear(d, d)
        self.bias  = nn.Parameter(torch.zeros(d))
        self.gamma = nn.Linear(d, d)
    def forward(self, x):
        m    = self.alpha(x) + self.bias
        mask = torch.sigmoid(m)
        sup  = x * mask
        attn = torch.sigmoid(self.gamma(sup))
        return sup * attn, mask, attn

class Encephalon(nn.Module):
    def __init__(self, nc=100, H_low=0.5, H_high=2.0):
        super().__init__()
        self.H_low, self.H_high = H_low, H_high
        self.backbone = CNNBackbone()
        self.gate     = FlowGate(256)
        self.head     = nn.Linear(256, nc)
    def forward(self, x):
        feats = self.backbone(x)
        with torch.no_grad():
            H     = entropy(F.softmax(self.head(feats), dim=-1))
            fmask = ((H > self.H_low) & (H < self.H_high)).float()
        gated, _, _ = self.gate(feats)
        m2  = fmask.unsqueeze(-1)
        out = self.head(m2 * gated + (1 - m2) * feats)
        return out, fmask

class AlwaysOnCNN(nn.Module):
    def __init__(self, nc=100):
        super().__init__()
        self.backbone = CNNBackbone()
        self.gate     = FlowGate(256)
        self.head     = nn.Linear(256, nc)
    def forward(self, x):
        feats       = self.backbone(x)
        gated, _, _ = self.gate(feats)
        return self.head(gated)

mean, std = (0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761)
train_tf = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ColorJitter(0.2,0.2,0.2), T.ToTensor(), T.Normalize(mean, std)
])
test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

train_ds = torchvision.datasets.CIFAR100("data/", train=True,  download=True, transform=train_tf)
test_ds  = torchvision.datasets.CIFAR100("data/", train=False, download=True, transform=test_tf)
train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=2, pin_memory=True)

print("CIFAR-100 loaded")

def train_one(model, loader, epochs, label):
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"\n--- {label} ---")
    for ep in range(1, epochs+1):
        model.train(); ls=co=n=0; t0=time.time()
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            loss = crit(logits, labs); loss.backward(); opt.step()
            ls += loss.item()*len(imgs)
            co += (logits.argmax(1)==labs).sum().item(); n+=len(imgs)
        sch.step()
        print(f"  Ep {ep:>2}/{epochs}  loss={ls/n:.4f}  acc={co/n:.4f}  ({time.time()-t0:.1f}s)")

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); co=n=0
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        out = model(imgs)
        logits = out[0] if isinstance(out, tuple) else out
        co += (logits.argmax(1)==labs).sum().item(); n+=len(imgs)
    return co/n

@torch.no_grad()
def flow_rate(model, loader):
    model.eval(); tot=n=0
    for imgs,_ in loader:
        _, fm = model(imgs.to(device))
        tot+=fm.sum().item(); n+=len(imgs)
    return tot/n

torch.manual_seed(42); np.random.seed(42)
baseline  = BaselineCNN(100).to(device)
always_on = AlwaysOnCNN(100).to(device)
enceph    = Encephalon(100, H_LOW, H_HIGH).to(device)

train_one(baseline,  train_loader, EPOCHS, "CIFAR100_Baseline")
train_one(always_on, train_loader, EPOCHS, "CIFAR100_AlwaysOn")
train_one(enceph,    train_loader, EPOCHS, "CIFAR100_Encephalon")

bc = evaluate(baseline,  test_loader)
ac = evaluate(always_on, test_loader)
fc = evaluate(enceph,    test_loader)
fr = flow_rate(enceph,   test_loader)

print(f"\n{'='*50}")
print(f"  CIFAR-100 RESULTS")
print(f"{'='*50}")
print(f"  Baseline   = {bc*100:.2f}%")
print(f"  AlwaysOn   = {ac*100:.2f}%")
print(f"  Encephalon = {fc*100:.2f}%")
print(f"  Flow rate  = {fr*100:.2f}%")
print(f"  Delta Enc-Base     = {(fc-bc)*100:+.2f}pp")
print(f"  Delta Enc-AlwaysOn = {(fc-ac)*100:+.2f}pp")

results = {"dataset":"CIFAR-100","baseline":bc,"always_on":ac,
           "encephalon":fc,"flow_rate":fr,
           "delta_base":(fc-bc),"delta_alwayson":(fc-ac)}
with open(f"{OUTDIR}/cifar100_results.json","w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"\nSaved -> {OUTDIR}/cifar100_results.json")

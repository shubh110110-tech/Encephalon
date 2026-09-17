import os, sys, json, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'}")

SEEDS  = [42, 123, 456]
EPOCHS = 30
LR     = 1e-3
BATCH  = 128
H_LOW  = 0.5
H_HIGH = 2.0
OUTDIR = "results_3seed"
Path(OUTDIR).mkdir(exist_ok=True)

C_BASE, C_FLOW, C_GATE = "#4A90D9", "#E8534A", "#6ABF69"

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
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = nn.Linear(256, 10)
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
    def __init__(self, H_low=0.5, H_high=2.0):
        super().__init__()
        self.H_low, self.H_high = H_low, H_high
        self.backbone = CNNBackbone()
        self.gate     = FlowGate(256)
        self.head     = nn.Linear(256, 10)
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
    def __init__(self):
        super().__init__()
        self.backbone = CNNBackbone()
        self.gate     = FlowGate(256)
        self.head     = nn.Linear(256, 10)
    def forward(self, x):
        feats       = self.backbone(x)
        gated, _, _ = self.gate(feats)
        return self.head(gated)

mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
train_tf = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ColorJitter(0.2,0.2,0.2), T.ToTensor(), T.Normalize(mean, std)
])
test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
test_ds = torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=test_tf)
test_loader = DataLoader(test_ds, BATCH, shuffle=False, num_workers=2, pin_memory=True)

def get_train_loader(seed):
    g = torch.Generator(); g.manual_seed(seed)
    train_ds = torchvision.datasets.CIFAR10("data/", train=True, download=True, transform=train_tf)
    return DataLoader(train_ds, BATCH, shuffle=True, num_workers=2, pin_memory=True, generator=g)

def train_one(model, loader, epochs, mode, label):
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

all_results = []

for seed in SEEDS:
    print(f"\n{'#'*50}\n  SEED {seed}\n{'#'*50}")
    torch.manual_seed(seed); np.random.seed(seed)
    train_loader = get_train_loader(seed)

    baseline  = BaselineCNN().to(device)
    always_on = AlwaysOnCNN().to(device)
    enceph    = Encephalon(H_low=H_LOW, H_high=H_HIGH).to(device)

    train_one(baseline,  train_loader, EPOCHS, "base",   f"Seed{seed}_Baseline")
    train_one(always_on, train_loader, EPOCHS, "always", f"Seed{seed}_AlwaysOn")
    train_one(enceph,    train_loader, EPOCHS, "flow",   f"Seed{seed}_Encephalon")

    bc = evaluate(baseline,  test_loader)
    ac = evaluate(always_on, test_loader)
    fc = evaluate(enceph,    test_loader)
    fr = flow_rate(enceph,   test_loader)

    print(f"\n  Seed {seed} Results:")
    print(f"  Baseline   = {bc*100:.2f}%")
    print(f"  AlwaysOn   = {ac*100:.2f}%")
    print(f"  Encephalon = {fc*100:.2f}%")
    print(f"  Flow rate  = {fr*100:.2f}%")
    print(f"  Delta Enc-Base     = {(fc-bc)*100:+.2f}pp")
    print(f"  Delta Enc-AlwaysOn = {(fc-ac)*100:+.2f}pp")

    all_results.append({"seed":seed,"baseline":bc,"always_on":ac,"encephalon":fc,"flow_rate":fr})

bc_arr = np.array([r["baseline"]   for r in all_results])
ac_arr = np.array([r["always_on"]  for r in all_results])
fc_arr = np.array([r["encephalon"] for r in all_results])
fr_arr = np.array([r["flow_rate"]  for r in all_results])

print(f"\n{'='*55}")
print(f"  FINAL SUMMARY")
print(f"{'='*55}")
print(f"  Baseline   = {bc_arr.mean()*100:.2f}% +/- {bc_arr.std()*100:.2f}%")
print(f"  AlwaysOn   = {ac_arr.mean()*100:.2f}% +/- {ac_arr.std()*100:.2f}%")
print(f"  Encephalon = {fc_arr.mean()*100:.2f}% +/- {fc_arr.std()*100:.2f}%")
print(f"  Flow rate  = {fr_arr.mean()*100:.2f}% +/- {fr_arr.std()*100:.2f}%")
print(f"  Delta Enc-Baseline   = {(fc_arr-bc_arr).mean()*100:+.2f}pp +/- {(fc_arr-bc_arr).std()*100:.2f}pp")
print(f"  Delta Enc-AlwaysOn   = {(fc_arr-ac_arr).mean()*100:+.2f}pp +/- {(fc_arr-ac_arr).std()*100:.2f}pp")

with open(f"{OUTDIR}/results.json","w") as f:
    json.dump(all_results, f, indent=2, default=float)
print(f"\nSaved -> {OUTDIR}/results.json")

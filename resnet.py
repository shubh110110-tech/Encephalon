import os, sys, json, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - switch to T4!'}")

EPOCHS = 30
LR     = 1e-3
BATCH  = 128
H_LOW  = 0.5
H_HIGH = 2.0
OUTDIR = "results_resnet"
Path(OUTDIR).mkdir(exist_ok=True)

def entropy(probs, eps=1e-8):
    return -(probs * (probs + eps).log()).sum(dim=-1)

class FlowGate(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.alpha = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, d))
        self.bias  = nn.Parameter(torch.zeros(d))
        self.gamma = nn.Sequential(nn.Linear(d, d//2), nn.ReLU(), nn.Linear(d//2, d))
    def forward(self, x):
        m    = self.alpha(x) + self.bias
        mask = torch.sigmoid(m)
        sup  = x * mask
        attn = torch.sigmoid(self.gamma(sup))
        return sup * attn, mask, attn

def make_backbone():
    b = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    b.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    b.maxpool = nn.Identity()
    feat_dim  = b.fc.in_features
    b.fc      = nn.Identity()
    return b, feat_dim

class ResNetBaseline(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.backbone, d = make_backbone()
        self.head = nn.Linear(d, nc)
    def forward(self, x):
        return self.head(self.backbone(x))

class ResNetAlwaysOn(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.backbone, d = make_backbone()
        self.gate = FlowGate(d)
        self.head = nn.Linear(d, nc)
    def forward(self, x):
        feats       = self.backbone(x)
        gated, _, _ = self.gate(feats)
        return self.head(gated)

class ResNetEncephalon(nn.Module):
    def __init__(self, nc=10, H_low=0.5, H_high=2.0):
        super().__init__()
        self.H_low, self.H_high = H_low, H_high
        self.backbone, d = make_backbone()
        self.gate = FlowGate(d)
        self.head = nn.Linear(d, nc)
    def forward(self, x):
        feats = self.backbone(x)
        with torch.no_grad():
            H     = entropy(F.softmax(self.head(feats), dim=-1))
            fmask = ((H > self.H_low) & (H < self.H_high)).float()
        gated, _, _ = self.gate(feats)
        m2  = fmask.unsqueeze(-1)
        out = self.head(m2 * gated + (1 - m2) * feats)
        return out, fmask

mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
train_tf = T.Compose([
    T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
    T.ColorJitter(0.2,0.2,0.2), T.ToTensor(), T.Normalize(mean, std)
])
test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
train_ds = torchvision.datasets.CIFAR10("data/", train=True,  download=True, transform=train_tf)
test_ds  = torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=test_tf)
train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=2, pin_memory=True)
print("Data loaded")

def train_one(model, loader, epochs, label):
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    backbone_params = list(model.backbone.parameters())
    new_params      = [p for n,p in model.named_parameters() if "backbone" not in n]
    opt = optim.AdamW([
        {"params": backbone_params, "lr": LR * 0.1},
        {"params": new_params,      "lr": LR}
    ], weight_decay=1e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"\n--- {label} ---")
    for ep in range(1, epochs+1):
        model.train(); ls=co=n=0; t0=time.time()
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            out    = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            loss   = crit(logits, labs); loss.backward(); opt.step()
            ls += loss.item()*len(imgs)
            co += (logits.argmax(1)==labs).sum().item(); n+=len(imgs)
        sch.step()
        print(f"  Ep {ep:>2}/{epochs}  loss={ls/n:.4f}  acc={co/n:.4f}  ({time.time()-t0:.1f}s)")

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); co=n=0
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        out    = model(imgs)
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
baseline  = ResNetBaseline().to(device)
always_on = ResNetAlwaysOn().to(device)
enceph    = ResNetEncephalon(H_low=H_LOW, H_high=H_HIGH).to(device)

print(f"\nParams - Baseline: {sum(p.numel() for p in baseline.parameters()):,}")
print(f"Params - AlwaysOn: {sum(p.numel() for p in always_on.parameters()):,}")
print(f"Params - Encephalon: {sum(p.numel() for p in enceph.parameters()):,}")

train_one(baseline,  train_loader, EPOCHS, "ResNet_Baseline")
train_one(always_on, train_loader, EPOCHS, "ResNet_AlwaysOn")
train_one(enceph,    train_loader, EPOCHS, "ResNet_Encephalon")

bc = evaluate(baseline,  test_loader)
ac = evaluate(always_on, test_loader)
fc = evaluate(enceph,    test_loader)
fr = flow_rate(enceph,   test_loader)

print(f"\n{'='*50}")
print(f"  RESNET-18 RESULTS")
print(f"{'='*50}")
print(f"  Baseline   = {bc*100:.2f}%")
print(f"  AlwaysOn   = {ac*100:.2f}%")
print(f"  Encephalon = {fc*100:.2f}%")
print(f"  Flow rate  = {fr*100:.2f}%")
print(f"  Delta Enc-Base     = {(fc-bc)*100:+.2f}pp")
print(f"  Delta Enc-AlwaysOn = {(fc-ac)*100:+.2f}pp")
print(f"\n  If Delta Enc-AlwaysOn > 0 = mechanism works on ResNet-18")
print(f"  = architecture agnostic claim proven")

torch.save(baseline.state_dict(),  f"{OUTDIR}/resnet_baseline.pt")
torch.save(always_on.state_dict(), f"{OUTDIR}/resnet_alwayson.pt")
torch.save(enceph.state_dict(),    f"{OUTDIR}/resnet_encephalon.pt")

results = {"backbone":"ResNet-18","baseline":bc,"always_on":ac,
           "encephalon":fc,"flow_rate":fr,
           "delta_base":(fc-bc),"delta_alwayson":(fc-ac)}
with open(f"{OUTDIR}/resnet_results.json","w") as f:
    json.dump(results, f, indent=2, default=float)
print(f"Saved -> {OUTDIR}/resnet_results.json")

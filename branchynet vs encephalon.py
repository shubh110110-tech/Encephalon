import os, json, time
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
THRESHOLD = 0.5  # BranchyNet single threshold — exit early if confidence > this
OUTDIR = "results_branchynet"
Path(OUTDIR).mkdir(exist_ok=True)

def entropy(probs, eps=1e-8):
    return -(probs * (probs + eps).log()).sum(dim=-1)

# ── shared backbone ──────────────────────────────────────────
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.drop = nn.Dropout(0.4)

    def forward(self, x, return_intermediate=False):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        feat = self.drop(x3.squeeze(-1).squeeze(-1))
        if return_intermediate:
            # early exit features from block1
            early = x1.mean(dim=[2,3])
            return feat, early
        return feat

# ── BranchyNet ───────────────────────────────────────────────
class BranchyNet(nn.Module):
    """
    Single threshold early exit network.
    If confidence at early exit > threshold, exit early.
    Otherwise continue to full network.
    This is the direct prior work Encephalon improves upon.
    """
    def __init__(self, nc=10, threshold=0.5):
        super().__init__()
        self.threshold = threshold
        self.backbone  = CNNBackbone()
        # early exit head after block1 (64 channels, 16x16)
        self.early_head = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, nc)
        )
        # final head
        self.final_head = nn.Linear(256, nc)

    def forward(self, x, force_full=False):
        feat, early_feat = self.backbone(x, return_intermediate=True)
        early_logits = self.early_head(early_feat)
        early_probs  = F.softmax(early_logits, dim=-1)
        early_conf   = early_probs.max(dim=-1).values

        if not self.training and not force_full:
            # at inference: exit early if confident enough
            exit_mask = (early_conf > self.threshold)
            final_logits = self.final_head(feat)
            # use early exit for confident samples, full for uncertain
            out = torch.where(
                exit_mask.unsqueeze(-1).expand_as(early_logits),
                early_logits, final_logits
            )
            return out, exit_mask
        else:
            # during training: always use final head + auxiliary loss on early
            final_logits = self.final_head(feat)
            return final_logits, early_logits

# ── Baseline ─────────────────────────────────────────────────
class BaselineCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = nn.Linear(256, nc)
    def forward(self, x):
        return self.head(self.backbone(x))

# ── Encephalon ───────────────────────────────────────────────
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
    def __init__(self, nc=10, H_low=0.5, H_high=2.0):
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

# ── data ─────────────────────────────────────────────────────
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

# ── train ─────────────────────────────────────────────────────
def train_baseline(model, loader, epochs, label):
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
            ls+=loss.item()*len(imgs)
            co+=(logits.argmax(1)==labs).sum().item(); n+=len(imgs)
        sch.step()
        print(f"  Ep {ep:>2}/{epochs}  loss={ls/n:.4f}  acc={co/n:.4f}  ({time.time()-t0:.1f}s)")

def train_branchynet(model, loader, epochs, label):
    """BranchyNet needs auxiliary loss on early exit head."""
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"\n--- {label} ---")
    for ep in range(1, epochs+1):
        model.train(); ls=co=n=0; t0=time.time()
        for imgs, labs in loader:
            imgs, labs = imgs.to(device), labs.to(device)
            opt.zero_grad()
            final_logits, early_logits = model(imgs)
            # weighted combination: 0.4 early exit + 0.6 final
            loss = 0.6*crit(final_logits, labs) + 0.4*crit(early_logits, labs)
            loss.backward(); opt.step()
            ls+=loss.item()*len(imgs)
            co+=(final_logits.argmax(1)==labs).sum().item(); n+=len(imgs)
        sch.step()
        print(f"  Ep {ep:>2}/{epochs}  loss={ls/n:.4f}  acc={co/n:.4f}  ({time.time()-t0:.1f}s)")

# ── eval ──────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, mode="base"):
    model.eval(); co=n=0
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        out = model(imgs)
        logits = out[0] if isinstance(out, tuple) else out
        co+=(logits.argmax(1)==labs).sum().item(); n+=len(imgs)
    return co/n

@torch.no_grad()
def eval_noisy(model, loader, sigma=0.3):
    model.eval(); co=n=0
    for imgs, labs in loader:
        imgs, labs = imgs.to(device), labs.to(device)
        noisy  = imgs + sigma*torch.randn_like(imgs)
        out    = model(noisy)
        logits = out[0] if isinstance(out, tuple) else out
        co+=(logits.argmax(1)==labs).sum().item(); n+=len(imgs)
    return co/n

@torch.no_grad()
def branchy_exit_rate(model, loader):
    model.eval(); early=n=0
    for imgs,_ in loader:
        imgs = imgs.to(device)
        _, exit_mask = model(imgs)
        early+=exit_mask.sum().item(); n+=len(imgs)
    return early/n

@torch.no_grad()
def enceph_flow_rate(model, loader):
    model.eval(); tot=n=0
    for imgs,_ in loader:
        _, fm = model(imgs.to(device))
        tot+=fm.sum().item(); n+=len(imgs)
    return tot/n

# ── run ───────────────────────────────────────────────────────
torch.manual_seed(42); np.random.seed(42)
baseline  = BaselineCNN().to(device)
branchynet = BranchyNet(threshold=THRESHOLD).to(device)
enceph    = Encephalon(H_low=H_LOW, H_high=H_HIGH).to(device)

train_baseline(baseline,   train_loader, EPOCHS, "Baseline")
train_branchynet(branchynet, train_loader, EPOCHS, "BranchyNet")
train_baseline(enceph,     train_loader, EPOCHS, "Encephalon")

bc  = evaluate(baseline,   test_loader)
brc = evaluate(branchynet, test_loader)
fc  = evaluate(enceph,     test_loader)

bn_noisy = eval_noisy(branchynet, test_loader, 0.3)
enc_noisy = eval_noisy(enceph,    test_loader, 0.3)
base_noisy = eval_noisy(baseline, test_loader, 0.3)

ber = branchy_exit_rate(branchynet, test_loader)
efr = enceph_flow_rate(enceph,     test_loader)

print(f"\n{'='*55}")
print(f"  HEAD-TO-HEAD: BranchyNet vs Encephalon")
print(f"{'='*55}")
print(f"\n  {'Model':<15} {'Clean':>8} {'Noisy(0.3)':>12} {'Gate Rate':>12}")
print(f"  {'-'*50}")
print(f"  {'Baseline':<15} {bc*100:>7.2f}%  {base_noisy*100:>10.2f}%  {'N/A':>12}")
print(f"  {'BranchyNet':<15} {brc*100:>7.2f}%  {bn_noisy*100:>10.2f}%  {ber*100:>10.2f}%")
print(f"  {'Encephalon':<15} {fc*100:>7.2f}%  {enc_noisy*100:>10.2f}%  {efr*100:>10.2f}%")
print(f"\n  Encephalon vs BranchyNet clean:      {(fc-brc)*100:+.2f}pp")
print(f"  Encephalon vs BranchyNet noisy(0.3): {(enc_noisy-bn_noisy)*100:+.2f}pp")
print(f"\n  Key: BranchyNet = single threshold (prior work)")
print(f"       Encephalon  = dual threshold (our method)")
print(f"       positive delta = Encephalon wins")

with open(f"{OUTDIR}/comparison_results.json","w") as f:
    json.dump({
        "baseline":   {"clean":bc,  "noisy":base_noisy},
        "branchynet": {"clean":brc, "noisy":bn_noisy, "exit_rate":ber},
        "encephalon": {"clean":fc,  "noisy":enc_noisy, "flow_rate":efr},
        "delta_clean": fc-brc, "delta_noisy": enc_noisy-bn_noisy
    }, f, indent=2, default=float)
print(f"Saved -> {OUTDIR}/comparison_results.json")

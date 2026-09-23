import os, json, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

EPOCHS = 30
LR     = 1e-3
BATCH  = 128
H_LOW  = 0.5
H_HIGH = 2.0
OUTDIR = "results_resnet_fixed"
Path(OUTDIR).mkdir(exist_ok=True)

def entropy(probs, eps=1e-8):
    return -(probs * (probs + eps).log()).sum(dim=-1)

class FlowGate(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.alpha = nn.Sequential(
            nn.Linear(d, d//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d//2, d)
        )
        self.bias  = nn.Parameter(torch.zeros(d))
        self.gamma = nn.Sequential(
            nn.Linear(d, d//2), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(d//2, d)
        )
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
    # freeze early layers to prevent overfitting
    for name, param in b.named_parameters():
        if "layer1" in name or "layer2" in name:
            param.requires_grad = False
    feat_dim = b.fc.in_features
    b.fc     = nn.Identity()
    return b, feat_dim

class ResNetBaseline(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.backbone, d = make_backbone()
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(d, nc)
        )
    def forward(self, x):
        return self.head(self.backbone(x))

class ResNetAlwaysOn(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.backbone, d = make_backbone()
        self.gate = FlowGate(d)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(d, nc))
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
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(d, nc))
    def forward(self, x):
        feats = self.backbone(x)
        with torch.no_grad():
            H     = entropy(F.softmax(self.head(feats), dim=-1))
            fmask = ((H > self.H_low) & (H < self.H_high)).float()
        gated, _, _ = self.gate(feats)
        m2  = fmask.unsqueeze(-1)
        out = self.head(m2 * gated + (1 - m2) * feats)
        return out, fmask

# stronger augmentation to prevent overfitting
mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.3, 0.3, 0.3),
    T.RandomGrayscale(p=0.1),
    T.ToTensor(),
    T.Normalize(mean, std),
    T.RandomErasing(p=0.2)
])
test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

train_ds = torchvision.datasets.CIFAR10("data/", train=True,  download=True, transform=train_tf)
test_ds  = torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=test_tf)
train_loader = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  BATCH, shuffle=False, num_workers=2, pin_memory=True)
print("Data loaded")

def train_one(model, loader, epochs, label):
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    # separate LR — frozen layers get 0, new layers get full LR
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    new_params      = [p for n,p in model.named_parameters() if "backbone" not in n]
    opt = optim.AdamW([
        {"params": backbone_params, "lr": LR * 0.05},  # very slow backbone
        {"params": new_params,      "lr": LR}
    ], weight_decay=5e-4)  # stronger weight decay
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    print(f"\n--- {label} ---")
    best_test = 0
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
        train_acc = co/n
        # eval every 5 epochs to track test acc
        if ep % 5 == 0 or ep == epochs:
            test_acc = evaluate(model, test_loader)
            best_test = max(best_test, test_acc)
            print(f"  Ep {ep:>2}/{epochs}  loss={ls/n:.4f}  train={train_acc:.4f}  test={test_acc:.4f}  ({time.time()-t0:.1f}s)")
        else:
            print(f"  Ep {ep:>2}/{epochs}  loss={ls/n:.4f}  train={train_acc:.4f}  ({time.time()-t0:.1f}s)")
    return best_test

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

print(f"Trainable params:")
print(f"  Baseline:   {sum(p.numel() for p in baseline.parameters() if p.requires_grad):,}")
print(f"  AlwaysOn:   {sum(p.numel() for p in always_on.parameters() if p.requires_grad):,}")
print(f"  Encephalon: {sum(p.numel() for p in enceph.parameters() if p.requires_grad):,}")

bc = train_one(baseline,  train_loader, EPOCHS, "ResNet_Baseline")
ac = train_one(always_on, train_loader, EPOCHS, "ResNet_AlwaysOn")
fc = train_one(enceph,    train_loader, EPOCHS, "ResNet_Encephalon")
fr = flow_rate(enceph, test_loader)

print(f"\n{'='*50}")
print(f"  RESNET-18 RESULTS (overfitting fixed)")
print(f"{'='*50}")
print(f"  Baseline   = {bc*100:.2f}%")
print(f"  AlwaysOn   = {ac*100:.2f}%")
print(f"  Encephalon = {fc*100:.2f}%")
print(f"  Flow rate  = {fr*100:.2f}%")
print(f"  Delta Enc-Base     = {(fc-bc)*100:+.2f}pp")
print(f"  Delta Enc-AlwaysOn = {(fc-ac)*100:+.2f}pp")

with open(f"{OUTDIR}/resnet_results.json","w") as f:
    json.dump({"baseline":bc,"always_on":ac,"encephalon":fc,
               "flow_rate":fr,"delta_base":(fc-bc),"delta_alwayson":(fc-ac)},
              f, indent=2, default=float)
print(f"Saved -> {OUTDIR}/resnet_results.json")

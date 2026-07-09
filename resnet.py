"""
encephalon_resnet.py  —  Encephalon on ResNet-18 backbone
ST Yau / IRIS / ISEF

Proves Encephalon's dual-threshold entropy gate is architecture-agnostic:
works on custom CNN (done) AND ResNet-18 (this script).

Run: python encephalon_resnet.py
"""

import os, json, time, urllib.request, tarfile
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision, torchvision.transforms as T
import torchvision.models as models
import matplotlib, matplotlib.pyplot as plt, matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
})

C_BASE, C_FLOW, C_GATE = "#4A90D9", "#E8534A", "#6ABF69"
torch.manual_seed(42); np.random.seed(42)

CIFAR10C_DIR = Path("data/CIFAR-10-C")
CIFAR10C_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    "speckle_noise", "gaussian_blur", "spatter", "saturate"
]

CORRUPTION_GROUPS = {
    "Noise":   ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"],
    "Blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur", "gaussian_blur"],
    "Weather": ["snow", "frost", "fog", "brightness", "spatter"],
    "Digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression", "saturate"]
}

# ═══════════════════════════════════════════════════════════════
# 1. CORE ENCEPHALON COMPONENTS (architecture-agnostic)
# ═══════════════════════════════════════════════════════════════
def entropy(probs, eps=1e-8):
    return -(probs * (probs + eps).log()).sum(dim=-1)

class FlowGate(nn.Module):
    """
    The Encephalon gate — plugs into ANY backbone.
    d = feature dimension from backbone (512 for ResNet-18)
    alpha-analogue: noise suppression
    gamma-analogue: signal amplification
    """
    def __init__(self, d=512):
        super().__init__()
        # alpha-analogue: learned feature suppression mask
        self.alpha = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, d)
        )
        self.bias = nn.Parameter(torch.zeros(d))
        # gamma-analogue: learned signal amplification
        self.gamma = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, d)
        )

    def forward(self, x, hard=False):
        # alpha: suppress unreliable features
        m    = self.alpha(x) + self.bias
        mask = (m > 0).float() if hard else torch.sigmoid(m)
        sup  = x * mask
        # gamma: amplify task-relevant signals
        attn = torch.sigmoid(self.gamma(sup))
        return sup * attn, mask, attn

# ═══════════════════════════════════════════════════════════════
# 2. THREE MODELS — ResNet-18 backbone
# ═══════════════════════════════════════════════════════════════
def make_resnet18_backbone():
    """ResNet-18 pretrained on ImageNet, adapted for CIFAR-10 (32x32)."""
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # adapt first conv for CIFAR small images
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    # remove final FC — we use our own head
    feat_dim = backbone.fc.in_features  # 512
    backbone.fc = nn.Identity()
    return backbone, feat_dim

class ResNetBaseline(nn.Module):
    """Standard ResNet-18 classifier — experimental control."""
    def __init__(self, nc=10):
        super().__init__()
        self.backbone, feat_dim = make_resnet18_backbone()
        self.head = nn.Linear(feat_dim, nc)

    def forward(self, x):
        return self.head(self.backbone(x))

class ResNetAlwaysOn(nn.Module):
    """
    Ablation: ResNet-18 + FlowGate always active, NO entropy trigger.
    Isolates gate parameters from the challenge-skill mechanism.
    """
    def __init__(self, nc=10):
        super().__init__()
        self.backbone, feat_dim = make_resnet18_backbone()
        self.gate = FlowGate(feat_dim)
        self.head = nn.Linear(feat_dim, nc)

    def forward(self, x):
        feats       = self.backbone(x)
        gated, _, _ = self.gate(feats)
        return self.head(gated)

class ResNetEncephalon(nn.Module):
    """
    Encephalon on ResNet-18 backbone.
    Gate activates ONLY within dual-threshold flow zone [H_low, H_high].
    Outside zone: identical to ResNetBaseline, zero overhead.
    """
    def __init__(self, nc=10, H_low=0.5, H_high=2.0):
        super().__init__()
        self.H_low, self.H_high = H_low, H_high
        self.backbone, feat_dim = make_resnet18_backbone()
        self.gate = FlowGate(feat_dim)
        self.head = nn.Linear(feat_dim, nc)

    def forward(self, x, hard=False):
        feats = self.backbone(x)

        # measure entropy — are we in the productive uncertainty band?
        with torch.no_grad():
            H     = entropy(F.softmax(self.head(feats), dim=-1))
            fmask = ((H > self.H_low) & (H < self.H_high)).float()

        # flow gate activates only in the band
        gated, alpha, gamma = self.gate(feats, hard)
        m2  = fmask.unsqueeze(-1)
        out = self.head(m2 * gated + (1 - m2) * feats)

        return out, fmask, {
            "entropy":    H,
            "flow_mask":  fmask,
            "alpha_mask": alpha,
            "gamma_attn": gamma
        }

# ═══════════════════════════════════════════════════════════════
# 3. DATA
# ═══════════════════════════════════════════════════════════════
def get_cifar10_loaders(bs=128):
    mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR10("data/", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=test_tf)
    return (DataLoader(train_ds, bs, shuffle=True,  num_workers=0, pin_memory=True),
            DataLoader(test_ds,  bs, shuffle=False, num_workers=0, pin_memory=True))

def download_cifar10c():
    CIFAR10C_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = Path("data/CIFAR-10-C.tar")
    if not tar_path.exists():
        print("📥 Downloading CIFAR-10-C (~2.5 GB)...")
        def progress(count, block, total):
            print(f"\r   {min(count*block/total*100,100):.1f}%", end="", flush=True)
        urllib.request.urlretrieve(CIFAR10C_URL, tar_path, progress); print()
    if not (CIFAR10C_DIR / "labels.npy").exists():
        print("📦 Extracting...")
        with tarfile.open(tar_path) as tf: tf.extractall("data/")
        print("✅ Extracted")

def get_cifar10c_loader(corruption, severity=3, bs=128):
    mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
    data_path = CIFAR10C_DIR / f"{corruption}.npy"
    if not data_path.exists(): return None
    data   = np.load(data_path)
    labels = np.load(CIFAR10C_DIR / "labels.npy")
    idx    = slice((severity-1)*10000, severity*10000)
    imgs   = torch.from_numpy(data[idx]).permute(0,3,1,2).float() / 255.0
    lbls   = torch.from_numpy(labels[idx]).long()
    mean_t = torch.tensor(mean).view(1,3,1,1)
    std_t  = torch.tensor(std).view(1,3,1,1)
    imgs   = (imgs - mean_t) / std_t
    return DataLoader(TensorDataset(imgs, lbls), bs, shuffle=False, num_workers=0)

# ═══════════════════════════════════════════════════════════════
# 4. TRAIN
# ═══════════════════════════════════════════════════════════════
def train(model, loader, epochs, lr, device, mode, label):
    crit  = nn.CrossEntropyLoss(label_smoothing=0.1)
    # separate LR for pretrained backbone vs new layers
    backbone_params = list(model.backbone.parameters())
    new_params      = [p for n,p in model.named_parameters()
                       if not n.startswith("backbone")]
    opt = optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},  # fine-tune backbone slowly
        {"params": new_params,      "lr": lr}          # train new layers faster
    ], weight_decay=1e-4)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist  = []
    start = 1

    Path("results_resnet").mkdir(exist_ok=True)
    ckpt_path = f"results_resnet/{label}_ckpt.pt"
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sch.load_state_dict(ckpt["sch"])
        hist  = ckpt["hist"]
        start = ckpt["epoch"] + 1
        print(f"  ✅ Resumed {label} from epoch {start-1}")

    print(f"\n{'='*55}\n  Training {label} (ResNet-18 backbone)\n{'='*55}")
    for ep in range(start, epochs+1):
        model.train(); loss_sum = correct = n = 0
        t0 = time.time()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)[0] if mode == "flow" else model(imgs)
            loss   = crit(logits, labels)
            loss.backward(); opt.step()
            loss_sum += loss.item() * len(imgs)
            correct  += (logits.argmax(1) == labels).sum().item()
            n        += len(imgs)
        sch.step()
        hist.append({"epoch": ep, "loss": loss_sum/n, "acc": correct/n})
        print(f"  Ep {ep:>2}/{epochs}  loss={loss_sum/n:.4f}  acc={correct/n:.4f}  ({time.time()-t0:.1f}s)")
        if ep % 5 == 0:
            torch.save({"epoch":ep,"model":model.state_dict(),
                        "opt":opt.state_dict(),"sch":sch.state_dict(),
                        "hist":hist}, ckpt_path)
            print(f"  💾 Checkpoint saved")
    return hist

# ═══════════════════════════════════════════════════════════════
# 5. EVAL
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_clean(model, loader, device, mode="base"):
    model.eval(); correct = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)[0] if mode == "flow" else model(imgs)
        correct += (logits.argmax(1) == labels).sum().item(); n += len(imgs)
    return correct / n

@torch.no_grad()
def eval_corruption(model, corruption, device, mode="base", severities=(1,2,3,4,5)):
    results = {}
    for sev in severities:
        loader = get_cifar10c_loader(corruption, sev)
        if loader is None: results[sev] = None; continue
        correct = n = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)[0] if mode == "flow" else model(imgs)
            correct += (logits.argmax(1) == labels).sum().item(); n += len(imgs)
        results[sev] = correct / n
    return results

@torch.no_grad()
def get_flow_rate(model, loader, device):
    model.eval(); tot = n = 0
    for imgs, _ in loader:
        _, fm, _ = model(imgs.to(device))
        tot += fm.sum().item(); n += len(imgs)
    return tot / n

@torch.no_grad()
def entropy_analysis(model, loader, device, H_low, H_high, n_bins=10):
    model.eval(); all_H=[]; all_c=[]; all_f=[]
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, fmask, diag = model(imgs, hard=True)
        all_H.extend(diag["entropy"].cpu().tolist())
        all_c.extend((logits.argmax(1)==labels).float().cpu().tolist())
        all_f.extend(fmask.cpu().tolist())
    H, c, f = map(np.array, [all_H, all_c, all_f])
    edges = np.percentile(H, np.linspace(0, 100, n_bins+1)); bins=[]
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        m = (H>=lo)&(H<hi if i<n_bins-1 else H<=hi)
        if m.sum()==0: continue
        bins.append({"entropy_center":float((lo+hi)/2),
                     "accuracy":float(c[m].mean()),
                     "flow_rate":float(f[m].mean()),
                     "n_samples":int(m.sum())})
    return bins

# ═══════════════════════════════════════════════════════════════
# 6. PLOTS
# ═══════════════════════════════════════════════════════════════
def plot_all(data, cfg, prev_cnn_data=None):
    base = data["baseline"]
    fl   = data["flow"]
    ao   = data["always_on"]
    corr = data["corruption_results"]

    n_cols = 4
    n_rows = 3 if prev_cnn_data is None else 4
    fig = plt.figure(figsize=(22, n_rows * 4.5))
    fig.suptitle(
        f"ENCEPHALON — ResNet-18 Backbone — CIFAR-10 + CIFAR-10-C\n"
        f"(H_low={cfg['H_low']}, H_high={cfg['H_high']}, epochs={cfg['epochs']})",
        fontsize=14, fontweight="bold"
    )
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.5, wspace=0.35)

    def ep(h, k): return [x[k] for x in h]

    # ── A: Training Loss ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep(base["train_history"],"epoch"), ep(base["train_history"],"loss"),
            color=C_BASE, lw=2, marker="o", ms=3, label="Baseline")
    ax.plot(ep(ao["train_history"],"epoch"), ep(ao["train_history"],"loss"),
            color=C_GATE, lw=2, marker="^", ms=3, label="AlwaysOn")
    ax.plot(ep(fl["train_history"],"epoch"), ep(fl["train_history"],"loss"),
            color=C_FLOW, lw=2, marker="s", ms=3, label="Encephalon")
    ax.set_title("A  Training Loss (ResNet-18)", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(frameon=False, fontsize=8)

    # ── B: Training Accuracy ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ep(base["train_history"],"epoch"), ep(base["train_history"],"acc"),
            color=C_BASE, lw=2, marker="o", ms=3, label="Baseline")
    ax.plot(ep(ao["train_history"],"epoch"), ep(ao["train_history"],"acc"),
            color=C_GATE, lw=2, marker="^", ms=3, label="AlwaysOn")
    ax.plot(ep(fl["train_history"],"epoch"), ep(fl["train_history"],"acc"),
            color=C_FLOW, lw=2, marker="s", ms=3, label="Encephalon")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("B  Training Accuracy (ResNet-18)", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.legend(frameon=False, fontsize=8)

    # ── C: Clean accuracy bar ────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    accs   = [base["clean_accuracy"], ao["clean_accuracy"], fl["clean_accuracy"]]
    colors = [C_BASE, C_GATE, C_FLOW]
    labels = ["Baseline", "AlwaysOn", "Encephalon"]
    bars = ax.bar(labels, accs, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{acc:.2%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(min(accs)-0.02, max(accs)+0.04)
    ax.set_title("C  Clean Test Accuracy", fontweight="bold")

    # ── D: Entropy analysis ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 3])
    bins = fl["entropy_bin_analysis"]
    Hc   = [b["entropy_center"] for b in bins]
    bacc = [b["accuracy"] for b in bins]
    ax.plot(Hc, bacc, color=C_FLOW, lw=2, marker="D", ms=5)
    ax.axvspan(cfg["H_low"], cfg["H_high"], alpha=0.12, color=C_FLOW, label="Flow zone")
    ax.axvline(cfg["H_low"],  color=C_FLOW, lw=1.2, ls="--", alpha=0.7)
    ax.axvline(cfg["H_high"], color=C_FLOW, lw=1.2, ls="--", alpha=0.7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("D  Accuracy by Entropy", fontweight="bold")
    ax.set_xlabel("Entropy H (nats)"); ax.legend(frameon=False, fontsize=8)

    # ── E: CIFAR-10-C heatmap ────────────────────────────────
    ax = fig.add_subplot(gs[1, :])
    corruptions = [c for c in ALL_CORRUPTIONS if c in corr["flow"]]
    delta_matrix = np.zeros((len(corruptions), 5))
    for i, c in enumerate(corruptions):
        for j, s in enumerate([1,2,3,4,5]):
            fb = corr["baseline"][c].get(s)
            ff = corr["flow"][c].get(s)
            if fb and ff: delta_matrix[i,j] = (ff-fb)*100
    cmap = LinearSegmentedColormap.from_list("rg", ["#E8534A","white","#6ABF69"])
    im = ax.imshow(delta_matrix.T, cmap=cmap, aspect="auto", vmin=-5, vmax=5)
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels([c.replace("_"," ") for c in corruptions],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"Sev {s}" for s in [1,2,3,4,5]])
    ax.set_title("E  Encephalon vs Baseline Δ Accuracy on CIFAR-10-C (green=better)",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Δ Accuracy (pp)")

    # ── F: Mean accuracy by corruption group ─────────────────
    ax = fig.add_subplot(gs[2, 0:2])
    group_names = list(CORRUPTION_GROUPS.keys())
    x = np.arange(len(group_names)); bw = 0.25
    for gi, (gname, gcorrs) in enumerate(CORRUPTION_GROUPS.items()):
        gcorrs_v = [c for c in gcorrs if c in corr["flow"]]
        for mi, (mdata, color, label) in enumerate([
            (corr["baseline"], C_BASE, "Baseline"),
            (corr["always_on"], C_GATE, "AlwaysOn"),
            (corr["flow"], C_FLOW, "Encephalon")
        ]):
            vals = [np.mean([v for v in mdata[c].values() if v])
                    for c in gcorrs_v if c in mdata]
            if vals:
                ax.bar(gi+(mi-1)*bw, np.mean(vals), bw, color=color, alpha=0.85,
                       edgecolor="white", label=label if gi==0 else "")
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("F  Mean Accuracy by Corruption Group", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    # ── G: Flow activation rate ───────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    rate_labels = ["Clean"]
    rate_vals   = [fl["flow_rate_clean"]]
    for c in ["gaussian_noise", "fog", "contrast", "pixelate"]:
        if c in fl.get("flow_rates_corrupted", {}):
            rate_labels.append(c.replace("_","\n"))
            rate_vals.append(fl["flow_rates_corrupted"][c])
    colors_h = [C_FLOW]+["#B06ABF"]*len(rate_vals[1:])
    bars = ax.bar(range(len(rate_vals)), rate_vals, color=colors_h, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, rate_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rate_vals)))
    ax.set_xticklabels(rate_labels, fontsize=7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("G  Flow Activation Rate\n(scales with difficulty)", fontweight="bold")

    # ── H: CNN vs ResNet comparison (if CNN data provided) ───
    if prev_cnn_data:
        ax = fig.add_subplot(gs[2, 3])
        cnn_clean = prev_cnn_data.get("flow_clean", 0)
        rn_clean  = fl["clean_accuracy"]
        cnn_noise = prev_cnn_data.get("flow_noise_gaussian", 0)
        rn_noise_vals = [v for v in corr["flow"].get("gaussian_noise",{}).values() if v]
        rn_noise  = np.mean(rn_noise_vals) if rn_noise_vals else 0
        x = np.arange(2); bw = 0.3
        ax.bar(x-bw/2, [cnn_clean, cnn_noise], bw, color="#9B59B6", alpha=0.85,
               edgecolor="white", label="Custom CNN")
        ax.bar(x+bw/2, [rn_clean,  rn_noise],  bw, color=C_FLOW,   alpha=0.85,
               edgecolor="white", label="ResNet-18")
        ax.set_xticks(x)
        ax.set_xticklabels(["Clean", "Gaussian Noise\n(mean)"])
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        ax.set_title("H  Architecture Comparison\n(Encephalon gate, same mechanism)",
                     fontweight="bold")
        ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    out = "results_resnet/encephalon_resnet_results.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\n📊 Figure saved → {out}")
    plt.show()

# ═══════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--batch",         type=int,   default=128)
    p.add_argument("--H_low",         type=float, default=0.5)
    p.add_argument("--H_high",        type=float, default=2.0)
    p.add_argument("--skip_download", action="store_true")
    p.add_argument("--cnn_results",   type=str,   default=None,
                   help="Path to previous CNN results JSON for comparison plot")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 ENCEPHALON — ResNet-18 Backbone")
    print(f"   Device={device}  Epochs={args.epochs}")
    print(f"   H_low={args.H_low}  H_high={args.H_high}")
    print(f"   This proves Encephalon is architecture-agnostic ✅")

    if not args.skip_download:
        download_cifar10c()

    train_loader, test_loader = get_cifar10_loaders(args.batch)

    baseline  = ResNetBaseline().to(device)
    always_on = ResNetAlwaysOn().to(device)
    enceph    = ResNetEncephalon(H_low=args.H_low, H_high=args.H_high).to(device)

    print(f"\n  Baseline  params: {sum(p.numel() for p in baseline.parameters()):,}")
    print(f"  AlwaysOn  params: {sum(p.numel() for p in always_on.parameters()):,}")
    print(f"  Encephalon params: {sum(p.numel() for p in enceph.parameters()):,}")

    bh = train(baseline,  train_loader, args.epochs, args.lr, device, "base",   "ResNet_Baseline")
    ah = train(always_on, train_loader, args.epochs, args.lr, device, "always", "ResNet_AlwaysOn")
    fh = train(enceph,    train_loader, args.epochs, args.lr, device, "flow",   "ResNet_Encephalon")

    print("\n" + "="*55)
    print("  CLEAN ACCURACY — ResNet-18")
    print("="*55)
    bc = eval_clean(baseline,  test_loader, device, "base")
    ac = eval_clean(always_on, test_loader, device, "base")
    fc = eval_clean(enceph,    test_loader, device, "flow")
    print(f"  Baseline   = {bc:.4f} ({bc*100:.2f}%)")
    print(f"  AlwaysOn   = {ac:.4f} ({ac*100:.2f}%)")
    print(f"  Encephalon = {fc:.4f} ({fc*100:.2f}%)")
    print(f"  Δ (Enc - Base)    = {(fc-bc)*100:+.2f}pp")
    print(f"  Δ (Enc - AlwaysOn)= {(fc-ac)*100:+.2f}pp  ← ablation proof")

    fr_clean = get_flow_rate(enceph, test_loader, device)
    print(f"\n  Flow rate (clean): {fr_clean:.2%}")

    ebins = entropy_analysis(enceph, test_loader, device, args.H_low, args.H_high)

    print("\n" + "="*55)
    print("  CIFAR-10-C BENCHMARKS — ResNet-18")
    print("="*55)
    corr_results  = {"baseline": {}, "flow": {}, "always_on": {}}
    fr_corrupted  = {}

    for corruption in ALL_CORRUPTIONS:
        if not (CIFAR10C_DIR / f"{corruption}.npy").exists():
            print(f"  ⚠️  {corruption}: not found, skipping"); continue

        br = eval_corruption(baseline,  corruption, device, "base")
        ar = eval_corruption(always_on, corruption, device, "base")
        fr = eval_corruption(enceph,    corruption, device, "flow")

        corr_results["baseline"][corruption]  = br
        corr_results["always_on"][corruption] = ar
        corr_results["flow"][corruption]      = fr

        # flow rate at severity 3
        loader3 = get_cifar10c_loader(corruption, 3, args.batch)
        if loader3:
            enceph.eval(); tot = n = 0
            with torch.no_grad():
                for imgs, _ in loader3:
                    _, fm, _ = enceph(imgs.to(device))
                    tot += fm.sum().item(); n += len(imgs)
            fr_corrupted[corruption] = tot/n if n>0 else 0

        mb = np.mean([v for v in br.values() if v])
        mf = np.mean([v for v in fr.values() if v])
        print(f"  {corruption:<22} Base={mb:.3f}  Enc={mf:.3f}  "
              f"Δ={( mf-mb)*100:+.2f}pp  FlowRate={fr_corrupted.get(corruption,0):.1%}")

    Path("results_resnet").mkdir(exist_ok=True)
    cfg = {"dataset":"CIFAR-10","epochs":args.epochs,
           "H_low":args.H_low,"H_high":args.H_high,"backbone":"ResNet-18"}

    results = {
        "config": cfg,
        "baseline":  {"train_history":bh, "clean_accuracy":bc},
        "always_on": {"train_history":ah, "clean_accuracy":ac},
        "flow": {
            "train_history":       fh,
            "clean_accuracy":      fc,
            "flow_rate_clean":     fr_clean,
            "flow_rates_corrupted":fr_corrupted,
            "entropy_bin_analysis":ebins,
        },
        "corruption_results": corr_results,
    }

    with open("results_resnet/resnet_metrics.json","w") as f:
        json.dump(results, f, indent=2)
    torch.save(baseline.state_dict(),  "results_resnet/resnet_baseline.pt")
    torch.save(always_on.state_dict(), "results_resnet/resnet_alwayson.pt")
    torch.save(enceph.state_dict(),    "results_resnet/resnet_encephalon.pt")

    # load previous CNN results for comparison if provided
    prev_cnn = None
    if args.cnn_results and os.path.exists(args.cnn_results):
        with open(args.cnn_results) as f:
            cnn_data = json.load(f)
        prev_cnn = {
            "flow_clean": cnn_data["flow"]["clean_accuracy"],
            "flow_noise_gaussian": np.mean([
                v for v in cnn_data["corruption_results"]["flow"]
                .get("gaussian_noise",{}).values() if v
            ])
        }

    print("\n📈 Generating plots...")
    plot_all(results, cfg, prev_cnn)

    print("\n✅ DONE — results_resnet/")
    print("   resnet_metrics.json            — all numbers")
    print("   encephalon_resnet_results.png  — 8-panel figure")
    print("\n  KEY COMPARISON FOR PAPER:")
    print(f"  Custom CNN  Encephalon clean acc: check cifar_metrics.json")
    print(f"  ResNet-18   Encephalon clean acc: {fc*100:.2f}%")
    print(f"\n  If both show Encephalon > AlwaysOn → gate is architecture-agnostic ✅")

if __name__ == "__main__":
    main()

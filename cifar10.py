"""
encephalon_cifar.py  —  Encephalon on CIFAR-10 + CIFAR-10-C
ST Yau / IRIS / ISEF

Trains all 3 models on CIFAR-10, then benchmarks on all 19 CIFAR-10-C corruptions.
Auto-downloads CIFAR-10-C if not present.

Run: python encephalon_cifar.py
"""

import os, json, time, urllib.request, tarfile
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision, torchvision.transforms as T
import matplotlib, matplotlib.pyplot as plt, matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

matplotlib.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
    "font.family": "sans-serif"
})

C_BASE, C_FLOW, C_GATE = "#4A90D9", "#E8534A", "#6ABF69"
torch.manual_seed(42); np.random.seed(42)

CIFAR10C_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
CIFAR10C_DIR = Path("data/CIFAR-10-C")

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
# 1. MODELS  (RGB backbone for CIFAR)
# ═══════════════════════════════════════════════════════════════
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
    def __init__(self, nc=10):
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
    def forward(self, x, hard=False):
        m    = self.alpha(x) + self.bias
        mask = (m > 0).float() if hard else torch.sigmoid(m)
        sup  = x * mask
        attn = torch.sigmoid(self.gamma(sup))
        return sup * attn, mask, attn

class FlowStateCNN(nn.Module):
    def __init__(self, nc=10, H_low=0.5, H_high=2.0):
        super().__init__()
        self.H_low, self.H_high = H_low, H_high
        self.backbone = CNNBackbone()
        self.gate     = FlowGate(256)
        self.head     = nn.Linear(256, nc)

    def forward(self, x, hard=False):
        feats = self.backbone(x)
        with torch.no_grad():
            H     = entropy(F.softmax(self.head(feats), dim=-1))
            fmask = ((H > self.H_low) & (H < self.H_high)).float()
        gated, alpha, gamma = self.gate(feats, hard)
        m2  = fmask.unsqueeze(-1)
        out = self.head(m2 * gated + (1 - m2) * feats)
        return out, fmask, {"entropy": H, "flow_mask": fmask,
                            "alpha_mask": alpha, "gamma_attn": gamma}

class AlwaysOnCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.backbone = CNNBackbone()
        self.gate     = FlowGate(256)
        self.head     = nn.Linear(256, nc)
    def forward(self, x):
        feats       = self.backbone(x)
        gated, _, _ = self.gate(feats)
        return self.head(gated)

# ═══════════════════════════════════════════════════════════════
# 2. DATA
# ═══════════════════════════════════════════════════════════════
def get_cifar10_loaders(bs=128):
    mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train_ds = torchvision.datasets.CIFAR10("data/", train=True,  download=True, transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10("data/", train=False, download=True, transform=test_tf)
    return (DataLoader(train_ds, bs, shuffle=True,  num_workers=0),
            DataLoader(test_ds,  bs, shuffle=False, num_workers=0))

def download_cifar10c():
    CIFAR10C_DIR.mkdir(parents=True, exist_ok=True)
    tar_path = Path("data/CIFAR-10-C.tar")
    if not tar_path.exists():
        print("📥 Downloading CIFAR-10-C (~2.5 GB) ...")
        def progress(count, block, total):
            pct = count * block / total * 100
            print(f"\r   {min(pct,100):.1f}%", end="", flush=True)
        urllib.request.urlretrieve(CIFAR10C_URL, tar_path, progress)
        print()
    if not (CIFAR10C_DIR / "labels.npy").exists():
        print("📦 Extracting CIFAR-10-C ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall("data/")
        print("✅ Extracted")

def get_cifar10c_loader(corruption, severity=3, bs=128):
    mean, std = (0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)
    data_path = CIFAR10C_DIR / f"{corruption}.npy"
    label_path = CIFAR10C_DIR / "labels.npy"
    if not data_path.exists():
        return None
    data   = np.load(data_path)
    labels = np.load(label_path)
    # severity 1-5, each is 10000 samples
    idx    = slice((severity-1)*10000, severity*10000)
    imgs   = torch.from_numpy(data[idx]).permute(0,3,1,2).float() / 255.0
    lbls   = torch.from_numpy(labels[idx]).long()
    mean_t = torch.tensor(mean).view(1,3,1,1)
    std_t  = torch.tensor(std).view(1,3,1,1)
    imgs   = (imgs - mean_t) / std_t
    ds     = TensorDataset(imgs, lbls)
    return DataLoader(ds, bs, shuffle=False, num_workers=0)

# ═══════════════════════════════════════════════════════════════
# 3. TRAIN
# ═══════════════════════════════════════════════════════════════
def train(model, loader, epochs, lr, device, mode, label):
    crit  = nn.CrossEntropyLoss()
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sch   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist  = []
    start = 1

    Path("results").mkdir(exist_ok=True)
    ckpt_path = f"results/{label}_ckpt.pt"
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sch.load_state_dict(ckpt["sch"])
        hist  = ckpt["hist"]
        start = ckpt["epoch"] + 1
        print(f"  ✅ Resumed {label} from epoch {start-1}")

    print(f"\n{'='*52}\n  Training {label}\n{'='*52}")
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
        elapsed = time.time() - t0
        hist.append({"epoch": ep, "loss": loss_sum/n, "acc": correct/n})
        print(f"  Ep {ep:>2}/{epochs}  loss={loss_sum/n:.4f}  acc={correct/n:.4f}  ({elapsed:.1f}s)")
        if ep % 5 == 0:
            torch.save({"epoch":ep,"model":model.state_dict(),
                        "opt":opt.state_dict(),"sch":sch.state_dict(),
                        "hist":hist}, ckpt_path)
            print(f"  💾 Checkpoint saved")
    return hist

# ═══════════════════════════════════════════════════════════════
# 4. EVAL
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
        if loader is None:
            results[sev] = None; continue
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
def entropy_analysis(flow_model, loader, device, n_bins=10):
    flow_model.eval(); all_H=[]; all_c=[]; all_f=[]
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, fmask, diag = flow_model(imgs, hard=True)
        all_H.extend(diag["entropy"].cpu().tolist())
        all_c.extend((logits.argmax(1)==labels).float().cpu().tolist())
        all_f.extend(fmask.cpu().tolist())
    H, c, f = map(np.array, [all_H, all_c, all_f])
    edges = np.percentile(H, np.linspace(0, 100, n_bins+1)); bins=[]
    for i in range(n_bins):
        lo, hi = edges[i], edges[i+1]
        m = (H >= lo) & (H < hi if i < n_bins-1 else H <= hi)
        if m.sum() == 0: continue
        bins.append({"entropy_center": float((lo+hi)/2),
                     "accuracy": float(c[m].mean()),
                     "flow_rate": float(f[m].mean()),
                     "n_samples": int(m.sum())})
    return bins

# ═══════════════════════════════════════════════════════════════
# 5. PLOTS
# ═══════════════════════════════════════════════════════════════
def plot_all(data, cfg):
    base = data["baseline"]
    fl   = data["flow"]
    ao   = data["always_on"]
    corr = data["corruption_results"]

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle(
        f"ENCEPHALON — CIFAR-10 + CIFAR-10-C Results\n"
        f"(H_low={cfg['H_low']}, H_high={cfg['H_high']}, epochs={cfg['epochs']})",
        fontsize=15, fontweight="bold", y=1.01
    )

    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    def ep(h, k): return [x[k] for x in h]

    # ── A: Training Loss ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep(base["train_history"],"epoch"), ep(base["train_history"],"loss"),
            color=C_BASE, lw=2, marker="o", ms=3, label="Baseline")
    ax.plot(ep(ao["train_history"],"epoch"), ep(ao["train_history"],"loss"),
            color=C_GATE, lw=2, marker="^", ms=3, label="AlwaysOn")
    ax.plot(ep(fl["train_history"],"epoch"), ep(fl["train_history"],"loss"),
            color=C_FLOW, lw=2, marker="s", ms=3, label="Flow")
    ax.set_title("A  Training Loss", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(frameon=False, fontsize=8)

    # ── B: Training Accuracy ─────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ep(base["train_history"],"epoch"), ep(base["train_history"],"acc"),
            color=C_BASE, lw=2, marker="o", ms=3, label="Baseline")
    ax.plot(ep(ao["train_history"],"epoch"), ep(ao["train_history"],"acc"),
            color=C_GATE, lw=2, marker="^", ms=3, label="AlwaysOn")
    ax.plot(ep(fl["train_history"],"epoch"), ep(fl["train_history"],"acc"),
            color=C_FLOW, lw=2, marker="s", ms=3, label="Flow")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("B  Training Accuracy", fontweight="bold")
    ax.set_xlabel("Epoch"); ax.legend(frameon=False, fontsize=8)

    # ── C: Clean accuracy bar ────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    models = ["Baseline", "AlwaysOn", "Flow"]
    accs   = [base["clean_accuracy"], ao["clean_accuracy"], fl["clean_accuracy"]]
    colors = [C_BASE, C_GATE, C_FLOW]
    bars = ax.bar(models, accs, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{acc:.2%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    bottom = min(accs) - 0.02
    ax.set_ylim(bottom, max(accs) + 0.03)
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

    # ── E: Corruption heatmap (19 corruptions × 5 severities) ─
    ax = fig.add_subplot(gs[1, :])
    corruptions = [c for c in ALL_CORRUPTIONS if c in corr["flow"]]
    sev_labels  = [1, 2, 3, 4, 5]

    delta_matrix = np.zeros((len(corruptions), 5))
    for i, c in enumerate(corruptions):
        for j, s in enumerate(sev_labels):
            fb = corr["baseline"][c].get(s)
            ff = corr["flow"][c].get(s)
            if fb is not None and ff is not None:
                delta_matrix[i, j] = (ff - fb) * 100  # percentage points

    cmap = LinearSegmentedColormap.from_list("rg", ["#E8534A", "white", "#6ABF69"])
    im   = ax.imshow(delta_matrix.T, cmap=cmap, aspect="auto",
                     vmin=-5, vmax=5)
    ax.set_xticks(range(len(corruptions)))
    ax.set_xticklabels([c.replace("_"," ") for c in corruptions],
                       rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"Sev {s}" for s in sev_labels])
    ax.set_title("E  Encephalon vs Baseline Δ Accuracy on CIFAR-10-C (percentage points, green=better)",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Δ Accuracy (pp)")

    # ── F: Mean accuracy by corruption group ─────────────────
    ax = fig.add_subplot(gs[2, 0:2])
    group_names = list(CORRUPTION_GROUPS.keys())
    x = np.arange(len(group_names)); bw = 0.25
    for gi, (gname, gcorrs) in enumerate(CORRUPTION_GROUPS.items()):
        gcorrs_valid = [c for c in gcorrs if c in corr["flow"]]
        for mi, (mdata, color, label) in enumerate(
            [(corr["baseline"], C_BASE, "Baseline"),
             (corr["always_on"], C_GATE, "AlwaysOn"),
             (corr["flow"], C_FLOW, "Flow")]):
            vals = [np.mean([v for v in mdata[c].values() if v is not None])
                    for c in gcorrs_valid if c in mdata]
            if vals:
                ax.bar(gi + (mi-1)*bw, np.mean(vals), bw,
                       color=color, alpha=0.85, edgecolor="white",
                       label=label if gi == 0 else "")
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("F  Mean Accuracy by Corruption Group", fontweight="bold")
    ax.set_ylabel("Accuracy"); ax.legend(frameon=False, fontsize=8)

    # ── G: Severity curve for worst corruption ────────────────
    ax = fig.add_subplot(gs[2, 2])
    # find corruption where baseline drops the most at severity 5
    worst = min(
        [c for c in ALL_CORRUPTIONS if c in corr["baseline"]],
        key=lambda c: corr["baseline"][c].get(5, 1.0) or 1.0
    )
    sevs = [1,2,3,4,5]
    for mdata, color, label in [
        (corr["baseline"], C_BASE, "Baseline"),
        (corr["always_on"], C_GATE, "AlwaysOn"),
        (corr["flow"], C_FLOW, "Flow")
    ]:
        vals = [mdata[worst].get(s) for s in sevs]
        vals = [v for v in vals if v is not None]
        ax.plot(sevs[:len(vals)], vals, color=color, lw=2, marker="o", ms=5, label=label)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title(f"G  Severity Curve\n({worst.replace('_',' ')})", fontweight="bold")
    ax.set_xlabel("Severity"); ax.legend(frameon=False, fontsize=8)

    # ── H: Flow activation rate across datasets ───────────────
    ax = fig.add_subplot(gs[2, 3])
    rate_labels = ["CIFAR-10\nClean"]
    rate_vals   = [fl["flow_rate_clean"]]
    for c in ["gaussian_noise", "fog", "contrast", "pixelate"]:
        if c in fl.get("flow_rates_corrupted", {}):
            rate_labels.append(c.replace("_","\n"))
            rate_vals.append(fl["flow_rates_corrupted"][c])
    colors_h = [C_FLOW] + ["#B06ABF"]*len(rate_vals[1:])
    bars = ax.bar(range(len(rate_vals)), rate_vals, color=colors_h, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, rate_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{val:.1%}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(len(rate_vals)))
    ax.set_xticklabels(rate_labels, fontsize=7)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("H  Flow Activation Rate\n(higher on harder inputs)", fontweight="bold")

    fig.tight_layout()
    out = "results/encephalon_cifar_results.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    print(f"\n📊 Figure saved → {out}")
    plt.show()

# ═══════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int,   default=30)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--batch",  type=int,   default=128)
    p.add_argument("--H_low",  type=float, default=0.5)
    p.add_argument("--H_high", type=float, default=2.0)
    p.add_argument("--severities", type=int, nargs="+", default=[1,2,3,4,5])
    p.add_argument("--skip_download", action="store_true",
                   help="Skip CIFAR-10-C download (if already present)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🧠 ENCEPHALON — CIFAR-10 + CIFAR-10-C")
    print(f"   Device={device}  Epochs={args.epochs}")
    print(f"   H_low={args.H_low}  H_high={args.H_high}")

    # download CIFAR-10-C
    if not args.skip_download:
        download_cifar10c()
    else:
        print("⏭  Skipping CIFAR-10-C download")

    train_loader, test_loader = get_cifar10_loaders(args.batch)

    baseline   = BaselineCNN().to(device)
    flow_model = FlowStateCNN(H_low=args.H_low, H_high=args.H_high).to(device)
    always_on  = AlwaysOnCNN().to(device)

    bh = train(baseline,   train_loader, args.epochs, args.lr, device, "base",   "Baseline")
    fh = train(flow_model, train_loader, args.epochs, args.lr, device, "flow",   "FlowState")
    ah = train(always_on,  train_loader, args.epochs, args.lr, device, "always", "AlwaysOn")

    print("\n" + "="*52)
    print("  CLEAN ACCURACY")
    print("="*52)
    bc = eval_clean(baseline,   test_loader, device, "base")
    fc = eval_clean(flow_model, test_loader, device, "flow")
    ac = eval_clean(always_on,  test_loader, device, "base")
    print(f"  Baseline  = {bc:.4f} ({bc*100:.2f}%)")
    print(f"  AlwaysOn  = {ac:.4f} ({ac*100:.2f}%)")
    print(f"  Flow      = {fc:.4f} ({fc*100:.2f}%)")

    fr_clean = get_flow_rate(flow_model, test_loader, device)
    print(f"\n  Flow rate (clean): {fr_clean:.2%}")

    ebins = entropy_analysis(flow_model, test_loader, device)

    # eval on all CIFAR-10-C corruptions
    print("\n" + "="*52)
    print("  CIFAR-10-C CORRUPTION BENCHMARKS")
    print("="*52)

    corr_results = {"baseline": {}, "flow": {}, "always_on": {}}
    fr_corrupted = {}

    for corruption in ALL_CORRUPTIONS:
        loader_sev3 = get_cifar10c_loader(corruption, severity=3, bs=args.batch)
        if loader_sev3 is None:
            print(f"  ⚠️  {corruption}: not found, skipping")
            continue

        br = eval_corruption(baseline,   corruption, device, "base",   args.severities)
        fr = eval_corruption(flow_model, corruption, device, "flow",   args.severities)
        ar = eval_corruption(always_on,  corruption, device, "base",   args.severities)

        corr_results["baseline"][corruption]  = br
        corr_results["flow"][corruption]      = fr
        corr_results["always_on"][corruption] = ar

        # flow rate at severity 3
        _, fmask_sum, n_sum = 0, 0, 0
        flow_model.eval()
        with torch.no_grad():
            for imgs, _ in loader_sev3:
                _, fm, _ = flow_model(imgs.to(device))
                fmask_sum += fm.sum().item(); n_sum += len(imgs)
        fr_corrupted[corruption] = fmask_sum / n_sum if n_sum > 0 else 0

        mean_b = np.mean([v for v in br.values() if v])
        mean_f = np.mean([v for v in fr.values() if v])
        delta  = (mean_f - mean_b) * 100
        print(f"  {corruption:<22} Base={mean_b:.3f}  Flow={mean_f:.3f}  Δ={delta:+.2f}pp  FlowRate={fr_corrupted[corruption]:.1%}")

    # save everything
    Path("results").mkdir(exist_ok=True)
    cfg = {"dataset": "CIFAR-10", "epochs": args.epochs,
           "H_low": args.H_low, "H_high": args.H_high}

    results = {
        "config": cfg,
        "baseline": {
            "train_history": bh,
            "clean_accuracy": bc,
        },
        "flow": {
            "train_history": fh,
            "clean_accuracy": fc,
            "flow_rate_clean": fr_clean,
            "flow_rates_corrupted": fr_corrupted,
            "entropy_bin_analysis": ebins,
        },
        "always_on": {
            "train_history": ah,
            "clean_accuracy": ac,
        },
        "corruption_results": corr_results,
    }

    with open("results/cifar_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    torch.save(baseline.state_dict(),   "results/cifar_baseline.pt")
    torch.save(flow_model.state_dict(), "results/cifar_flow.pt")
    torch.save(always_on.state_dict(),  "results/cifar_alwayson.pt")

    print("\n📈 Generating plots...")
    plot_all(results, cfg)

    print("\n✅ DONE — results/")
    print("   cifar_metrics.json       — all numbers for your paper")
    print("   encephalon_cifar_results.png — 8-panel figure")

if __name__ == "__main__":
    main()

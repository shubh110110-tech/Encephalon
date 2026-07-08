"""
flow_state.py  —  Flow-State Inspired Neural Computation
ST Yau / IRIS High School Science Award

Just double-click or run: python "flow state.py"
Automatically resumes from checkpoint if crashed.
"""

import os, argparse, json, time
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision, torchvision.transforms as T
import matplotlib, matplotlib.pyplot as plt, matplotlib.ticker as mticker

matplotlib.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,
                             "axes.grid":True,"grid.alpha":0.3,"figure.dpi":150})

C_BASE, C_FLOW, C_GATE = "#4A90D9", "#E8534A", "#6ABF69"
torch.manual_seed(42); np.random.seed(42)

# ═══════════════════════════════════════════════════════════════
# 1. MODEL
# ═══════════════════════════════════════════════════════════════
def entropy(probs, eps=1e-8):
    return -(probs * (probs + eps).log()).sum(dim=-1)

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.AdaptiveAvgPool2d(1),
        )
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        return self.drop(self.net(x).squeeze(-1).squeeze(-1))

class BaselineCNN(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = nn.Linear(64, nc)
    def forward(self, x):
        return self.head(self.backbone(x))

class FlowGate(nn.Module):
    """Alpha-analogue (noise suppression) + Gamma-analogue (amplification)."""
    def __init__(self, d=64):
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
    """Challenge-triggered adaptive network."""
    def __init__(self, nc=10, H_low=0.2, H_high=1.5):
        super().__init__()
        self.H_low, self.H_high = H_low, H_high
        self.backbone = CNNBackbone()
        self.gate     = FlowGate()
        self.head     = nn.Linear(64, nc)

    def forward(self, x, hard=False):
        feats = self.backbone(x)
        with torch.no_grad():
            H     = entropy(F.softmax(self.head(feats), dim=-1))
            fmask = ((H > self.H_low) & (H < self.H_high)).float()
        gated, alpha, gamma = self.gate(feats, hard)
        m2  = fmask.unsqueeze(-1)
        out = self.head(m2 * gated + (1 - m2) * feats)
        return out, fmask, {"entropy":H, "flow_mask":fmask,
                            "alpha_mask":alpha, "gamma_attn":gamma}

class AlwaysOnCNN(nn.Module):
    """Ablation: gate always active, no entropy trigger."""
    def __init__(self, nc=10):
        super().__init__()
        self.backbone = CNNBackbone()
        self.gate     = FlowGate()
        self.head     = nn.Linear(64, nc)
    def forward(self, x):
        feats        = self.backbone(x)
        gated, _, _  = self.gate(feats)
        return self.head(gated)

# ═══════════════════════════════════════════════════════════════
# 2. DATA
# ═══════════════════════════════════════════════════════════════
def get_loaders(name, bs=128):
    tf = T.Compose([T.ToTensor(), T.Normalize((0.5,),(0.5,))])
    DS = getattr(torchvision.datasets, name)
    return (DataLoader(DS("data/",train=True, download=True,transform=tf),
                       bs,shuffle=True, num_workers=0),
            DataLoader(DS("data/",train=False,download=True,transform=tf),
                       bs,shuffle=False,num_workers=0))

# ═══════════════════════════════════════════════════════════════
# 3. TRAIN
# ═══════════════════════════════════════════════════════════════
def train(model, loader, epochs, lr, device, is_flow, label):
    crit  = nn.CrossEntropyLoss()
    opt   = optim.Adam(model.parameters(), lr=lr)
    sch   = optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
    hist  = []
    start = 1

    # ── resume from checkpoint if it exists ──────────────────
    Path("results").mkdir(exist_ok=True)
    ckpt_path = f"results/{label}_checkpoint.pt"
    if os.path.exists(ckpt_path):
        ckpt  = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sch.load_state_dict(ckpt["sch"])
        hist  = ckpt["hist"]
        start = ckpt["epoch"] + 1
        print(f"  ✅ Resumed {label} from epoch {start-1}")

    print(f"\n{'='*50}\n  Training {label}\n{'='*50}")
    for ep in range(start, epochs+1):
        model.train(); loss_sum = correct = n = 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            if is_flow == "flow":
                logits = model(imgs)[0]
            elif is_flow == "always":
                logits = model(imgs)
            else:
                logits = model(imgs)
            loss = crit(logits, labels)
            loss.backward(); opt.step()
            loss_sum += loss.item()*len(imgs)
            correct  += (logits.argmax(1)==labels).sum().item()
            n        += len(imgs)
        sch.step()
        hist.append({"epoch":ep,"loss":loss_sum/n,"acc":correct/n})
        print(f"  Ep {ep:>2}/{epochs}  loss={loss_sum/n:.4f}  acc={correct/n:.4f}")

        # save checkpoint every 5 epochs
        if ep % 5 == 0:
            torch.save({"epoch":ep,"model":model.state_dict(),
                        "opt":opt.state_dict(),"sch":sch.state_dict(),
                        "hist":hist}, ckpt_path)
            print(f"  💾 Checkpoint saved at epoch {ep}")

    return hist

# ═══════════════════════════════════════════════════════════════
# 4. EVAL
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def eval_clean(model, loader, device, mode="base"):
    model.eval(); correct = n = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)[0] if mode=="flow" else model(imgs)
        correct += (logits.argmax(1)==labels).sum().item(); n+=len(imgs)
    return correct/n

@torch.no_grad()
def eval_noisy(model, loader, device, mode="base", sigmas=(0.1,0.3,0.5,0.7,1.0)):
    model.eval(); res={}
    for s in sigmas:
        correct=n=0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            noisy  = imgs + s*torch.randn_like(imgs)
            logits = model(noisy)[0] if mode=="flow" else model(noisy)
            correct += (logits.argmax(1)==labels).sum().item(); n+=len(imgs)
        res[s] = correct/n
    return res

@torch.no_grad()
def entropy_bins(flow_model, loader, device, n_bins=10):
    flow_model.eval(); all_H=[]; all_c=[]; all_f=[]
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits, fmask, diag = flow_model(imgs, hard=True)
        all_H.extend(diag["entropy"].cpu().tolist())
        all_c.extend((logits.argmax(1)==labels).float().cpu().tolist())
        all_f.extend(fmask.cpu().tolist())
    H,c,f = map(np.array,[all_H,all_c,all_f])
    edges  = np.percentile(H, np.linspace(0,100,n_bins+1)); bins=[]
    for i in range(n_bins):
        lo,hi = edges[i],edges[i+1]
        m = (H>=lo)&(H<hi if i<n_bins-1 else H<=hi)
        if m.sum()==0: continue
        bins.append({"entropy_center":float((lo+hi)/2),
                     "accuracy":float(c[m].mean()),
                     "flow_rate":float(f[m].mean()),
                     "n_samples":int(m.sum())})
    return bins

@torch.no_grad()
def get_flow_rate(model, loader, device):
    model.eval(); tot=n=0
    for imgs,_ in loader:
        _,fm,_ = model(imgs.to(device))
        tot+=fm.sum().item(); n+=len(imgs)
    return tot/n

# ═══════════════════════════════════════════════════════════════
# 5. VISUALISE
# ═══════════════════════════════════════════════════════════════
def plot(data, out):
    cfg,base,fl,ao = (data["config"],data["baseline"],
                      data["flow"],data["always_on"])
    fig,axes = plt.subplots(2,3,figsize=(16,9))
    fig.suptitle(
        f"Flow-State Neural Computation — {cfg['dataset']}  "
        f"(H_low={cfg['H_low']}, H_high={cfg['H_high']})",
        fontsize=13,fontweight="bold",y=1.01)

    def ep(h,k): return [x[k] for x in h]

    # A — training loss
    ax=axes[0,0]
    ax.plot(ep(base["train_history"],"epoch"),ep(base["train_history"],"loss"),
            color=C_BASE,lw=2,marker="o",ms=4,label="Baseline")
    ax.plot(ep(ao["train_history"],"epoch"),ep(ao["train_history"],"loss"),
            color=C_GATE,lw=2,marker="^",ms=4,label="AlwaysOn")
    ax.plot(ep(fl["train_history"],"epoch"),ep(fl["train_history"],"loss"),
            color=C_FLOW,lw=2,marker="s",ms=4,label="Flow")
    ax.set_title("A  Training Loss",fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(frameon=False)

    # B — training accuracy
    ax=axes[0,1]
    ax.plot(ep(base["train_history"],"epoch"),ep(base["train_history"],"acc"),
            color=C_BASE,lw=2,marker="o",ms=4,label="Baseline")
    ax.plot(ep(ao["train_history"],"epoch"),ep(ao["train_history"],"acc"),
            color=C_GATE,lw=2,marker="^",ms=4,label="AlwaysOn")
    ax.plot(ep(fl["train_history"],"epoch"),ep(fl["train_history"],"acc"),
            color=C_FLOW,lw=2,marker="s",ms=4,label="Flow")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("B  Training Accuracy",fontweight="bold")
    ax.set_xlabel("Epoch"); ax.legend(frameon=False)

    # C — robustness
    ax=axes[0,2]
    sigs=[float(k) for k in sorted(base["noisy_accuracy"])]
    ax.plot(sigs,[base["noisy_accuracy"][str(s)] for s in sigs],
            color=C_BASE,lw=2,marker="o",ms=6,label="Baseline")
    ax.plot(sigs,[ao["noisy_accuracy"][str(s)]   for s in sigs],
            color=C_GATE,lw=2,marker="^",ms=6,label="AlwaysOn")
    ax.plot(sigs,[fl["noisy_accuracy"][str(s)]   for s in sigs],
            color=C_FLOW,lw=2,marker="s",ms=6,label="Flow")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("C  Robustness (AWGN)",fontweight="bold")
    ax.set_xlabel("Noise σ"); ax.legend(frameon=False)

    # D — entropy binned accuracy
    ax=axes[1,0]; bins=fl["entropy_bin_analysis"]
    Hc  =[b["entropy_center"] for b in bins]
    bacc=[b["accuracy"]        for b in bins]
    ax.plot(Hc,bacc,color=C_FLOW,lw=2,marker="D",ms=5)
    ax.axvspan(cfg["H_low"],cfg["H_high"],alpha=0.12,color=C_FLOW,label="Flow zone")
    ax.axvline(cfg["H_low"], color=C_FLOW,lw=1.2,ls="--",alpha=0.6)
    ax.axvline(cfg["H_high"],color=C_FLOW,lw=1.2,ls="--",alpha=0.6)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("D  Accuracy by Entropy",fontweight="bold")
    ax.set_xlabel("Entropy H (nats)"); ax.legend(frameon=False,fontsize=8)

    # E — flow activation rate
    ax=axes[1,1]; bfr=[b["flow_rate"] for b in bins]
    w = np.diff(Hc+[Hc[-1]]).mean()*0.8 if len(Hc)>1 else 0.1
    ax.bar(Hc,bfr,width=w,color=C_GATE,alpha=0.8,edgecolor="white")
    ax.axvline(cfg["H_low"], color=C_GATE,lw=1.2,ls="--",alpha=0.7,
               label=f"H_low={cfg['H_low']}")
    ax.axvline(cfg["H_high"],color=C_GATE,lw=1.2,ls="--",alpha=0.7,
               label=f"H_high={cfg['H_high']}")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_title("E  Flow Activation Rate",fontweight="bold")
    ax.set_xlabel("Entropy H (nats)"); ax.legend(frameon=False,fontsize=8)

    # F — summary bar chart (3 models)
    ax=axes[1,2]
    lbls=["Clean","σ=0.3","σ=0.5","σ=0.7"]; keys=[None,"0.3","0.5","0.7"]
    vb=[base["clean_accuracy"] if k is None else base["noisy_accuracy"].get(k,0) for k in keys]
    va=[ao["clean_accuracy"]   if k is None else ao["noisy_accuracy"].get(k,0)   for k in keys]
    vf=[fl["clean_accuracy"]   if k is None else fl["noisy_accuracy"].get(k,0)   for k in keys]
    x=np.arange(4); bw=0.25
    b1=ax.bar(x-bw,  vb,bw,color=C_BASE,alpha=0.85,edgecolor="white",label="Baseline")
    b2=ax.bar(x,     va,bw,color=C_GATE,alpha=0.85,edgecolor="white",label="AlwaysOn")
    b3=ax.bar(x+bw,  vf,bw,color=C_FLOW,alpha=0.85,edgecolor="white",
              label=f"Flow ({fl['overall_flow_rate']:.1%})")
    for bar in list(b1)+list(b2)+list(b3):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                f"{bar.get_height():.1%}",ha="center",va="bottom",fontsize=6)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xticks(x); ax.set_xticklabels(lbls)
    ax.set_title("F  3-Model Comparison",fontweight="bold")
    ax.legend(frameon=False,fontsize=8)

    fig.tight_layout()
    fig.savefig(out,bbox_inches="tight")
    print(f"\n📊 Figure saved → {out}")
    plt.show()

# ═══════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════
def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset",default="FashionMNIST",choices=["MNIST","FashionMNIST"])
    p.add_argument("--epochs", type=int,   default=20)
    p.add_argument("--lr",     type=float, default=1e-3)
    p.add_argument("--batch",  type=int,   default=128)
    p.add_argument("--H_low",  type=float, default=0.2)
    p.add_argument("--H_high", type=float, default=1.5)
    return p.parse_args()

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Flow-State Neural Computation")
    print(f"   Device={device}  Dataset={args.dataset}")
    print(f"   H_low={args.H_low}  H_high={args.H_high}  Epochs={args.epochs}")

    train_loader, test_loader = get_loaders(args.dataset, args.batch)

    baseline   = BaselineCNN().to(device)
    flow_model = FlowStateCNN(H_low=args.H_low, H_high=args.H_high).to(device)
    always_on  = AlwaysOnCNN().to(device)

    # train all 3 models (auto-resumes from checkpoint if crashed)
    bh = train(baseline,   train_loader, args.epochs, args.lr, device, "base",   "BaselineCNN")
    fh = train(flow_model, train_loader, args.epochs, args.lr, device, "flow",   "FlowStateCNN")
    ah = train(always_on,  train_loader, args.epochs, args.lr, device, "always", "AlwaysOnCNN")

    # evaluate all 3
    print("\n" + "="*50)
    print("  EVALUATION RESULTS")
    print("="*50)

    bc = eval_clean(baseline,   test_loader, device, "base")
    fc = eval_clean(flow_model, test_loader, device, "flow")
    ac = eval_clean(always_on,  test_loader, device, "base")
    print(f"\n  Clean Accuracy:")
    print(f"    Baseline  = {bc:.4f} ({bc*100:.2f}%)")
    print(f"    AlwaysOn  = {ac:.4f} ({ac*100:.2f}%)")
    print(f"    Flow      = {fc:.4f} ({fc*100:.2f}%)")

    bn = eval_noisy(baseline,   test_loader, device, "base")
    fn = eval_noisy(flow_model, test_loader, device, "flow")
    an = eval_noisy(always_on,  test_loader, device, "base")
    print(f"\n  Noisy Accuracy (AWGN):")
    print(f"  {'σ':<6} {'Baseline':>10} {'AlwaysOn':>10} {'Flow':>10} {'Δ(F-B)':>10}")
    print(f"  {'-'*50}")
    for s in bn:
        delta = fn[s]-bn[s]
        print(f"  {s:<6} {bn[s]:>10.4f} {an[s]:>10.4f} {fn[s]:>10.4f} {delta:>+10.4f}")

    ebins = entropy_bins(flow_model, test_loader, device)
    fr    = get_flow_rate(flow_model, test_loader, device)
    print(f"\n  Flow Activation Rate: {fr:.2%}")
    print(f"  (% of test samples that triggered the flow gate)")

    # save everything
    Path("results").mkdir(exist_ok=True)
    results = {
        "config":    {"dataset":args.dataset,"epochs":args.epochs,
                      "H_low":args.H_low,"H_high":args.H_high},
        "baseline":  {"train_history":bh,"clean_accuracy":bc,
                      "noisy_accuracy":{str(k):v for k,v in bn.items()}},
        "flow":      {"train_history":fh,"clean_accuracy":fc,
                      "noisy_accuracy":{str(k):v for k,v in fn.items()},
                      "overall_flow_rate":fr,"entropy_bin_analysis":ebins},
        "always_on": {"train_history":ah,"clean_accuracy":ac,
                      "noisy_accuracy":{str(k):v for k,v in an.items()}},
    }
    with open("results/metrics.json","w") as f:
        json.dump(results,f,indent=2)
    torch.save(baseline.state_dict(),   "results/baseline.pt")
    torch.save(flow_model.state_dict(), "results/flow_model.pt")
    torch.save(always_on.state_dict(),  "results/always_on.pt")
    plot(results, "results/flow_results.png")

    print("\n✅ DONE! All results saved to results/")
    print("   metrics.json  — all numbers for your paper")
    print("   flow_results.png — your 6-panel figure")
    print("   *.pt — model checkpoints")

if __name__ == "__main__":
    main()

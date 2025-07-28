# ========================= AVALANCHE BACKEND (DROP-IN) =========================
# This section assumes your data arrays are already in memory from your original loader:
# X_train, X_val, X_test : np.ndarray, shape (N, T, C) or (N, T) -> will be reshaped to (N, T, 1)
# y_*_super, y_*_sub     : np.ndarray, either 1-D int labels or 2-D multi-hot (0/1)
# classes_super, classes_sub : lists/arrays of class names

import math
import time
import copy
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset

# Avalanche
from avalanche.training.strategies import Naive, EWC, SynapticIntelligence
from avalanche.training.plugins import EvaluationPlugin, MultiHeadPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics, timing_metrics
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_generic_benchmark

# Metrics
from sklearn.metrics import f1_score

# ---------------- Reproducibility ----------------
def set_seed(seed: int = 1337):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- Data Adapters ----------------
def _ensure_3d(x: np.ndarray) -> np.ndarray:
    # Expect (N, T, C). If (N, T), add channel dim.
    if x.ndim == 2:
        return x[..., None]
    return x

def _is_multilabel(y: np.ndarray) -> bool:
    return (y.ndim == 2) and (set(np.unique(y)) <= {0, 1})

class NumpyECGDataset(Dataset):
    """
    Works for both single-label (y: 1-D ints) and multi-label (y: 2-D one-hot).
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        X = _ensure_3d(X)
        self.X = torch.from_numpy(X.astype(np.float32)).contiguous()
        self.y = torch.from_numpy(y.astype(np.float32 if _is_multilabel(y) else np.int64)).contiguous()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]  # (T, C)
        # Avalanche expects channel-first for conv1d in torch: (C, T)
        x = x.permute(1, 0)  # (C, T)
        y = self.y[idx]
        return x, y

# --------------- Backbones (1D) -----------------
class CNNBackbone(nn.Module):
    def __init__(self, in_ch=1, width=64, depth=3, dropout=0.1):
        super().__init__()
        layers_ = []
        ch = in_ch
        for i in range(depth):
            out = width * (2 ** i)
            layers_ += [
                nn.Conv1d(ch, out, kernel_size=7, padding=3, stride=1),
                nn.BatchNorm1d(out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
            ]
            if dropout and dropout > 0:
                layers_.append(nn.Dropout(dropout))
            ch = out
        self.feat = nn.Sequential(*layers_)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.feat(x)       # (B, C, T')
        x = self.gap(x).squeeze(-1)  # (B, C)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.skip  = None
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        s = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.skip is not None:
            s = self.skip(s)
        x = F.relu(x + s)
        return x

class ResNet1D(nn.Module):
    def __init__(self, in_ch=1, width=64, n_blocks=6, dropout=0.1):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        ch = width
        blocks = []
        for i in range(n_blocks):
            stride = 2 if i in {n_blocks // 3, 2 * (n_blocks // 3)} else 1
            out_ch = width * (2 ** (i // (n_blocks // 3 + 1)))
            blocks.append(ResBlock(ch, out_ch, stride))
            ch = out_ch
            if dropout and (i % 2 == 1):
                blocks.append(nn.Dropout(dropout))
        self.blocks = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return x

# Minimal 1D ViT (patchify 1D signal)
class PatchExtract1D(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.ps = patch_size
    def forward(self, x):  # x: (B, C, T)
        B, C, T = x.shape
        pad = (self.ps - (T % self.ps)) % self.ps
        if pad:
            x = F.pad(x, (0, pad))
            T = T + pad
        n = T // self.ps
        x = x.view(B, C, n, self.ps).permute(0, 2, 1, 3).contiguous()  # (B, n, C, ps)
        x = x.view(B, n, C * self.ps)  # (B, n, C*ps)
        return x

class MiniViT1D(nn.Module):
    def __init__(self, in_ch=1, patch_size=16, dim=128, depth=4, n_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch = PatchExtract1D(patch_size)
        self.proj = nn.Linear(in_ch * patch_size, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = None  # lazily create after seeing n_patches
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=n_heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # x: (B, C, T)
        x = self.patch(x)          # (B, n, C*ps)
        x = self.proj(x)           # (B, n, dim)
        B, n, d = x.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, n+1, dim)
        if (self.pos is None) or (self.pos.shape[1] != n + 1):
            self.pos = nn.Parameter(torch.randn(1, n + 1, d, device=x.device))
        x = x + self.pos
        x = self.encoder(x)        # (B, n+1, dim)
        x = self.norm(x[:, 0, :])  # CLS
        return x

# --------------- Heads & Multihead Model ----------------
class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim, multilabel: bool):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.multilabel = multilabel
    def forward(self, z):
        return self.fc(z)  # logits

class MultiHeadModel(nn.Module):
    """
    Shared backbone + per-task heads (for TiL). Task IDs: 0, 1.
    """
    def __init__(self, backbone: nn.Module, head_dims: Dict[int, int], feat_dim: int, multilabel: bool):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({str(t): LinearHead(feat_dim, out_dim, multilabel) for t, out_dim in head_dims.items()})
        self.current_task = 0
        self.multilabel = multilabel

    def set_task(self, task_id: int):
        self.current_task = task_id

    def forward(self, x):
        z = self.backbone(x)
        logits = self.heads[str(self.current_task)](z)
        return logits

# --------------- Specs ----------------
@dataclass
class BackboneSpec:
    family: str   # 'cnn' | 'resnet' | 'vit'
    size: str     # 'Small' | 'Medium' | 'Large' | 'Huge'
    params: dict
    feat_dim: int

def _infer_in_ch(X: np.ndarray) -> int:
    x = _ensure_3d(X)
    return x.shape[-1]

def build_backbone(spec: BackboneSpec, in_ch: int) -> nn.Module:
    fam = spec.family.lower()
    if fam == 'cnn':
        return CNNBackbone(in_ch=in_ch, **spec.params)
    if fam == 'resnet':
        return ResNet1D(in_ch=in_ch, **spec.params)
    if fam == 'vit':
        return MiniViT1D(in_ch=in_ch, **spec.params)
    raise ValueError(f"Unknown family: {spec.family}")

# --------------- Metrics helpers ----------------
def _sigmoid_logits_to_pred(y_logits: torch.Tensor) -> torch.Tensor:
    return (torch.sigmoid(y_logits) >= 0.5).long()

def _softmax_logits_to_pred(y_logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(y_logits, dim=1)

def eval_numpy(model: nn.Module, X: np.ndarray, y: np.ndarray, task_id: Optional[int], multilabel: bool) -> Dict[str, float]:
    model.eval()
    X = _ensure_3d(X)
    X_t = torch.from_numpy(X.astype(np.float32)).permute(0, 2, 1).to(DEVICE)  # (N, C, T)
    y_np = y.copy()
    with torch.no_grad():
        if isinstance(model, MultiHeadModel) and task_id is not None:
            model.set_task(task_id)
        logits = []
        bs = 512
        for i in range(0, len(X_t), bs):
            out = model(X_t[i:i+bs])
            logits.append(out.cpu())
        logits = torch.cat(logits, dim=0)
    if multilabel:
        y_true = y_np
        y_pred = _sigmoid_logits_to_pred(logits).numpy()
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        acc = (y_pred == y_true).mean()  # strict multilabel exact match
        return {"macro_f1": float(macro_f1), "acc": float(acc)}
    else:
        y_true = y_np.astype(np.int64)
        y_pred = _softmax_logits_to_pred(logits).numpy()
        acc = (y_pred == y_true).mean()
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return {"macro_f1": float(macro_f1), "acc": float(acc)}

# --------------- Scenario builders ----------------
def _to_avalanche_dataset(X: np.ndarray, y: np.ndarray, task_label: int) -> AvalancheDataset:
    ds = NumpyECGDataset(X, y)
    ads = AvalancheDataset(ds, task_labels=task_label)
    return ads

def make_til_benchmark(
    X0_tr, y0_tr, X0_te, y0_te,
    X1_tr, y1_tr, X1_te, y1_te
):
    """
    TiL: two experiences with distinct task labels, their own heads.
    Validation is folded into train_stream evaluation each epoch (Avalanche-style).
    """
    ex0_train = _to_avalanche_dataset(X0_tr, y0_tr, task_label=0)
    ex1_train = _to_avalanche_dataset(X1_tr, y1_tr, task_label=1)
    ex0_test  = _to_avalanche_dataset(X0_te, y0_te, task_label=0)
    ex1_test  = _to_avalanche_dataset(X1_te, y1_te, task_label=1)

    benchmark = create_generic_benchmark(
        train_datasets=[ex0_train, ex1_train],
        test_datasets=[ex0_test, ex1_test],
        task_labels=[0, 1],
        complete_test_set_only=False
    )
    return benchmark

def make_cil_benchmark(
    XA_tr, yA_tr, XA_te, yA_te,
    XB_tr, yB_tr, XB_te, yB_te
):
    """
    CiL: new classes appear in ex1. If multi-label, we 'expand' space by concatenating columns (A+B)
    and shifting yB into indices [nA..nA+nB-1].
    For single-label, map B labels += nA.
    """
    multilabel = _is_multilabel(yA_tr)
    if multilabel:
        nA = yA_tr.shape[1]; nB = yB_tr.shape[1]
        def shift_B(y):
            y2 = np.zeros((y.shape[0], nA + nB), dtype=y.dtype)
            y2[:, nA:] = y
            return y2
        yA_tr2, yA_te2 = yA_tr, yA_te
        yB_tr2, yB_te2 = shift_B(yB_tr), shift_B(yB_te)
    else:
        nA = int(np.max(yA_tr)) + 1
        yA_tr2, yA_te2 = yA_tr, yA_te
        yB_tr2, yB_te2 = yB_tr + nA, yB_te + nA

    ex0_train = _to_avalanche_dataset(XA_tr, yA_tr2, task_label=0)
    ex1_train = _to_avalanche_dataset(XB_tr, yB_tr2, task_label=0)  # same task label (single head)
    ex0_test  = _to_avalanche_dataset(XA_te, yA_te2, task_label=0)
    ex1_test  = _to_avalanche_dataset(XB_te, yB_te2, task_label=0)

    benchmark = create_generic_benchmark(
        train_datasets=[ex0_train, ex1_train],
        test_datasets=[ex0_test, ex1_test],
        task_labels=[0, 0],
        complete_test_set_only=False
    )
    return benchmark

# --------------- Strategy builder ----------------
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    ewc_lambda: float = 2.0
    si_c: float = 2.0

def make_strategy(
    method: str,
    model: nn.Module,
    multilabel: bool,
    lr: float,
    ewc_lambda: float,
    si_c: float,
    plugins=None,
    eval_mb_size: int = 256
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
    evaluator = EvaluationPlugin(
        accuracy_metrics(epoch=True, experience=True, stream=True),
        loss_metrics(epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True, experience=True, stream=True),
        loggers=[],
    )
    if method.lower() == "naive":
        return Naive(model, optimizer, criterion, train_mb_size=eval_mb_size, train_epochs=1, eval_mb_size=eval_mb_size, evaluator=evaluator, device=DEVICE, plugins=plugins)
    if method.lower() == "ewc":
        return EWC(model, optimizer, criterion, ewc_lambda=ewc_lambda, train_mb_size=eval_mb_size, train_epochs=1, eval_mb_size=eval_mb_size, evaluator=evaluator, device=DEVICE, plugins=plugins)
    if method.lower() == "si":
        return SynapticIntelligence(model, optimizer, criterion, si_lambda=si_c, train_mb_size=eval_mb_size, train_epochs=1, eval_mb_size=eval_mb_size, evaluator=evaluator, device=DEVICE, plugins=plugins)
    raise ValueError(f"Unknown method: {method}")

# --------------- Public APIs (called by ablation) ----------------
def run_task_incremental_avalanche(
    spec: BackboneSpec,
    config: TrainConfig,
    # Task 0 (super)
    X0_train, y0_train, X0_val, y0_val, X0_test, y0_test, classes0,
    # Task 1 (sub)
    X1_train, y1_train, X1_val, y1_val, X1_test, y1_test, classes1,
    method: str,
) -> Dict[str, Dict[str, float]]:
    set_seed(1337)
    in_ch = _infer_in_ch(X0_train)
    multilabel = _is_multilabel(y0_train)

    # Merge train+val per task (Avalanche trains epoch-wise; we keep eval separate below)
    X0_tr = np.concatenate([X0_train, X0_val], axis=0)
    y0_tr = np.concatenate([y0_train, y0_val], axis=0)
    X1_tr = np.concatenate([X1_train, X1_val], axis=0)
    y1_tr = np.concatenate([y1_train, y1_val], axis=0)

    bench = make_til_benchmark(X0_tr, y0_tr, X0_test, y0_test, X1_tr, y1_tr, X1_test, y1_test)

    # Backbone & heads
    backbone = build_backbone(spec, in_ch=in_ch).to(DEVICE)
    feat_dim = spec.feat_dim
    head_dims = {0: (y0_train.shape[1] if multilabel else len(classes0)),
                 1: (y1_train.shape[1] if multilabel else len(classes1))}
    model = MultiHeadModel(backbone, head_dims=head_dims, feat_dim=feat_dim, multilabel=multilabel).to(DEVICE)

    # Multi-head plugin selects the proper head by task label automatically
    mhp = MultiHeadPlugin()
    strat = make_strategy(method, model, multilabel, config.lr, config.ewc_lambda, config.si_c, plugins=[mhp], eval_mb_size=config.batch_size)

    # ---- Train on Task 0 ----
    model.set_task(0)
    for _ in range(config.epochs):
        strat.train(bench.train_stream[0])
    # Evaluate Task 0 after Task 0
    t0_after_t0 = eval_numpy(model, X0_test, y0_test, task_id=0, multilabel=multilabel)

    # ---- Train on Task 1 ----
    model.set_task(1)
    for _ in range(config.epochs):
        strat.train(bench.train_stream[1])
    # Evaluate Task 1 after Task 1
    t1_after_t1 = eval_numpy(model, X1_test, y1_test, task_id=1, multilabel=multilabel)
    # Retention on Task 0 after Task 1
    t0_after_t1 = eval_numpy(model, X0_test, y0_test, task_id=0, multilabel=multilabel)

    return {
        "t0_after_t0": t0_after_t0,
        "t1_after_t1": t1_after_t1,
        "t0_after_t1": t0_after_t1,
        "forgetting_t0": {"acc_drop": t0_after_t0["acc"] - t0_after_t1["acc"]},
    }

def run_class_incremental_avalanche(
    spec: BackboneSpec,
    config: TrainConfig,
    # A (super)
    XA_train, yA_train, XA_val, yA_val, XA_test, yA_test, classesA,
    # B (sub, disjoint)
    XB_train, yB_train, XB_val, yB_val, XB_test, yB_test, classesB,
    method: str,
) -> Dict[str, Dict[str, float]]:
    set_seed(1337)
    in_ch = _infer_in_ch(XA_train)
    multilabel = _is_multilabel(yA_train)

    # Merge train+val
    XA_tr = np.concatenate([XA_train, XA_val], axis=0)
    yA_tr = np.concatenate([yA_train, yA_val], axis=0)
    XB_tr = np.concatenate([XB_train, XB_val], axis=0)
    yB_tr = np.concatenate([yB_train, yB_val], axis=0)

    bench = make_cil_benchmark(XA_tr, yA_tr, XA_test, yA_test, XB_tr, yB_tr, XB_test, yB_test)

    # Build single-head model with output dim = |A| (will be "expanded" implicitly by using shifted labels in CiL scenario).
    backbone = build_backbone(spec, in_ch=in_ch).to(DEVICE)
    feat_dim = spec.feat_dim
    if multilabel:
        out_dim = yA_train.shape[1] + yB_train.shape[1]  # full A+B space
    else:
        out_dim = len(classesA) + len(classesB)
    head = LinearHead(feat_dim, out_dim, multilabel)
    model = nn.Sequential(backbone, head).to(DEVICE)

    strat = make_strategy(method, model, multilabel, config.lr, config.ewc_lambda, config.si_c, plugins=None, eval_mb_size=config.batch_size)

    # ---- Train on A ----
    for _ in range(config.epochs):
        strat.train(bench.train_stream[0])
    # Evaluate A after A
    acc_A_after_A = eval_numpy(model, XA_test, yA_test if multilabel else yA_test, task_id=None, multilabel=multilabel)

    # ---- Train on B (labels are already shifted in the scenario for single-head setting) ----
    for _ in range(config.epochs):
        strat.train(bench.train_stream[1])

    # Evaluate A, B after learning
    if multilabel:
        # Build yA padded to A+B for fair macro-F1 calc
        nA, nB = yA_train.shape[1], yB_train.shape[1]
        def padA(y):
            z = np.zeros((y.shape[0], nA + nB), dtype=y.dtype)
            z[:, :nA] = y
            return z
        acc_A_after_AplusB = eval_numpy(model, XA_test, padA(yA_test), task_id=None, multilabel=True)
        acc_B_after_AplusB = eval_numpy(model, XB_test, np.pad(yB_test, ((0,0),(nA,0))), task_id=None, multilabel=True)
    else:
        acc_A_after_AplusB = eval_numpy(model, XA_test, yA_test, task_id=None, multilabel=False)
        acc_B_after_AplusB = eval_numpy(model, XB_test, yB_test + len(classesA), task_id=None, multilabel=False)

    return {
        "acc_A_after_A": acc_A_after_A,
        "acc_A_after_AplusB": acc_A_after_AplusB,
        "acc_B_after_AplusB": acc_B_after_AplusB,
        "forgetting_A": {"acc_drop": acc_A_after_A["acc"] - acc_A_after_AplusB["acc"]},
    }

# --------------- Model grid (parametric sizes) ----------------
def model_grid() -> List[BackboneSpec]:
    grid: List[BackboneSpec] = []
    # CNN: feat_dim equals last channel size
    grid += [
        BackboneSpec("cnn",   "Small",  dict(width=32, depth=2, dropout=0.1),  feat_dim=32*(2**(2-1))),
        BackboneSpec("cnn",   "Medium", dict(width=64, depth=3, dropout=0.1),  feat_dim=64*(2**(3-1))),
        BackboneSpec("cnn",   "Large",  dict(width=96, depth=4, dropout=0.1),  feat_dim=96*(2**(4-1))),
        BackboneSpec("cnn",   "Huge",   dict(width=128, depth=5, dropout=0.1), feat_dim=128*(2**(5-1))),
    ]
    # ResNet: rough upper channel as feat_dim
    grid += [
        BackboneSpec("resnet","Small",  dict(width=32, n_blocks=4, dropout=0.1),  feat_dim=32*(2**1)),
        BackboneSpec("resnet","Medium", dict(width=64, n_blocks=6, dropout=0.1),  feat_dim=64*(2**1)),
        BackboneSpec("resnet","Large",  dict(width=96, n_blocks=9, dropout=0.1),  feat_dim=96*(2**2)),
        BackboneSpec("resnet","Huge",   dict(width=128,n_blocks=12,dropout=0.1),  feat_dim=128*(2**2)),
    ]
    # ViT: feat_dim equals transformer dim
    grid += [
        BackboneSpec("vit",   "Small",  dict(patch_size=16, dim=96,  depth=4, n_heads=4,  mlp_dim=192, dropout=0.1), feat_dim=96),
        BackboneSpec("vit",   "Medium", dict(patch_size=16, dim=128, depth=6, n_heads=4,  mlp_dim=256, dropout=0.1), feat_dim=128),
        BackboneSpec("vit",   "Large",  dict(patch_size=16, dim=192, depth=8, n_heads=6,  mlp_dim=384, dropout=0.1), feat_dim=192),
        BackboneSpec("vit",   "Huge",   dict(patch_size=16, dim=256, depth=10,n_heads=8,  mlp_dim=512, dropout=0.1), feat_dim=256),
    ]
    return grid

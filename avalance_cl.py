# avalance_cl.py
# Avalanche harness for ECG continual learning (TiL & CiL)
# Data is loaded EXACTLY via your CL_PTBXL.py (no synthetic fallbacks).

import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Avalanche
from avalanche.training.strategies import Naive, EWC, SynapticIntelligence
from avalanche.training.plugins import EvaluationPlugin, MultiHeadPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, timing_metrics
from avalanche.benchmarks.utils import AvalancheDataset
from avalanche.benchmarks.scenarios.generic_benchmark_creation import create_generic_benchmark

from sklearn.metrics import f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# --------------------- Reproducibility ---------------------
def set_seed(seed: int = 1337):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------- Data ---------------------------
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
        X = _ensure_3d(X).astype(np.float32)
        self.X = torch.from_numpy(X).contiguous()      # (N, T, C)
        if _is_multilabel(y):
            self.y = torch.from_numpy(y.astype(np.float32)).contiguous()
        else:
            self.y = torch.from_numpy(y.astype(np.int64)).contiguous()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]              # (T, C) from CL_PTBXL normalization
        x = x.permute(1, 0)          # -> (C, T) for Conv1d
        y = self.y[idx]
        return x, y

def _to_avalanche_dataset(X: np.ndarray, y: np.ndarray, task_label: int) -> AvalancheDataset:
    ds = NumpyECGDataset(X, y)
    return AvalancheDataset(ds, task_labels=task_label)

def load_ecg_arrays() -> dict:
    """
    Import your CL_PTBXL.py and return its already-processed arrays.
    This matches PRML_project.py processing exactly (your code).
    """
    try:
        mod = importlib.import_module("CL_PTBXL")
    except ModuleNotFoundError:
        raise RuntimeError("CL_PTBXL.py not found in the working directory. Place it next to this file.")
    keys = [
        "X_train","X_val","X_test",
        "y_train_super","y_val_super","y_test_super",
        "y_train_sub","y_val_sub","y_test_sub",
        "classes_super","classes_sub"
    ]
    missing = [k for k in keys if not hasattr(mod, k)]
    if missing:
        raise RuntimeError(f"CL_PTBXL.py is missing variables: {missing}")
    data = {k: getattr(mod, k) for k in keys}
    # Hard assert shape consistency (mirrors your pipeline): (N, T, C) and multilabel Y
    for name in ("X_train","X_val","X_test"):
        arr = data[name]
        if arr.ndim not in (2,3):
            raise RuntimeError(f"{name} expected 2D/3D array, got shape {arr.shape}")
    for name in ("y_train_super","y_val_super","y_test_super","y_train_sub","y_val_sub","y_test_sub"):
        arr = data[name]
        if arr.ndim != 2 or set(np.unique(arr)) - {0,1}:
            raise RuntimeError(f"{name} must be multi-hot (0/1) 2D; got shape {arr.shape}, values {np.unique(arr)[:5]}")
    return data

# ----------------------- Models ---------------------------
class CNNBackbone(nn.Module):
    def __init__(self, in_ch=12, width=64, depth=3, dropout=0.1):
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
        x = self.feat(x)
        x = self.gap(x).squeeze(-1)
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
    def __init__(self, in_ch=12, width=64, n_blocks=6, dropout=0.1):
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
        self.out_ch = ch

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x).squeeze(-1)
        return x

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
        x = x.view(B, C, n, self.ps).permute(0, 2, 1, 3).contiguous()
        x = x.view(B, n, C * self.ps)  # (B, n, C*ps)
        return x

class MiniViT1D(nn.Module):
    def __init__(self, in_ch=12, patch_size=16, dim=128, depth=4, n_heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch = PatchExtract1D(patch_size)
        self.proj = nn.Linear(in_ch * patch_size, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = None
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=mlp_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.patch(x)                   # (B, n, C*ps)
        x = self.proj(x)                    # (B, n, dim)
        B, n, d = x.shape
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)      # (B, n+1, dim)
        if (self.pos is None) or (self.pos.shape[1] != n + 1):
            self.pos = nn.Parameter(torch.randn(1, n + 1, d, device=x.device))
        x = x + self.pos
        x = self.encoder(x)
        x = self.norm(x[:, 0, :])           # CLS
        return x

class LinearHead(nn.Module):
    def __init__(self, in_dim, out_dim, multilabel: bool):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.multilabel = multilabel
    def forward(self, z):
        return self.fc(z)  # logits

class MultiHeadModel(nn.Module):
    """
    Shared backbone + per-task heads (TiL).
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

# --------------------- Specs & grid ---------------------
@dataclass
class BackboneSpec:
    family: str   # 'cnn' | 'resnet' | 'vit'
    size: str     # 'Small' | 'Medium' | 'Large' | 'Huge'
    params: dict
    feat_dim: int

@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 128
    lr: float = 1e-3
    ewc_lambda: float = 2.0
    si_c: float = 2.0

def _infer_in_ch(X: np.ndarray) -> int:
    x = _ensure_3d(X)
    return x.shape[-1]  # (N,T,C) -> C (PTB-XL is typically 12)

def model_grid() -> List[BackboneSpec]:
    """
    Complete, consistent grid: 4 sizes per family.
    Feature dims are set to the backbone's last channel (CNN/ResNet) or transformer dim (ViT).
    """
    grid: List[BackboneSpec] = []
    # CNN
    grid += [
        BackboneSpec("cnn",   "Small",  dict(width=32,  depth=2, dropout=0.1),  feat_dim=32*(2**(2-1))),
        BackboneSpec("cnn",   "Medium", dict(width=64,  depth=3, dropout=0.1),  feat_dim=64*(2**(3-1))),
        BackboneSpec("cnn",   "Large",  dict(width=96,  depth=4, dropout=0.1),  feat_dim=96*(2**(4-1))),
        BackboneSpec("cnn",   "Huge",   dict(width=128, depth=5, dropout=0.1),  feat_dim=128*(2**(5-1))),
    ]
    # ResNet (approximate last channel progression; robust across blocks)
    grid += [
        BackboneSpec("resnet","Small",  dict(width=32,  n_blocks=4,  dropout=0.1), feat_dim=64),
        BackboneSpec("resnet","Medium", dict(width=64,  n_blocks=6,  dropout=0.1), feat_dim=128),
        BackboneSpec("resnet","Large",  dict(width=96,  n_blocks=9,  dropout=0.1), feat_dim=192),
        BackboneSpec("resnet","Huge",   dict(width=128, n_blocks=12, dropout=0.1), feat_dim=256),
    ]
    # ViT (feat_dim equals model dim)
    grid += [
        BackboneSpec("vit",   "Small",  dict(patch_size=16, dim=96,  depth=4,  n_heads=4, mlp_dim=192, dropout=0.1), feat_dim=96),
        BackboneSpec("vit",   "Medium", dict(patch_size=16, dim=128, depth=6,  n_heads=4, mlp_dim=256, dropout=0.1), feat_dim=128),
        BackboneSpec("vit",   "Large",  dict(patch_size=16, dim=192, depth=8,  n_heads=6, mlp_dim=384, dropout=0.1), feat_dim=192),
        BackboneSpec("vit",   "Huge",   dict(patch_size=16, dim=256, depth=10, n_heads=8, mlp_dim=512, dropout=0.1), feat_dim=256),
    ]
    return grid

# --------------------- Strategy builder ---------------------
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

# ------------------------ Evaluation ------------------------
def _pred_from_logits(logits: torch.Tensor, multilabel: bool) -> np.ndarray:
    if multilabel:
        return (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
    return torch.argmax(logits, dim=1).cpu().numpy()

def eval_numpy(model: nn.Module, X: np.ndarray, y: np.ndarray, task_id: Optional[int], multilabel: bool) -> Dict[str, float]:
    model.eval()
    X = _ensure_3d(X).astype(np.float32)
    X_t = torch.from_numpy(X).permute(0, 2, 1).to(DEVICE)  # (N, C, T)
    y_np = y.copy()
    logits_list = []
    bs = 512
    with torch.no_grad():
        if isinstance(model, MultiHeadModel) and task_id is not None:
            model.set_task(task_id)
        for i in range(0, len(X_t), bs):
            out = model(X_t[i:i+bs])
            logits_list.append(out.cpu())
    logits = torch.cat(logits_list, dim=0)
    y_pred = _pred_from_logits(logits, multilabel)
    if multilabel:
        macro_f1 = f1_score(y_np, y_pred, average='macro', zero_division=0)
        acc = (y_pred == y_np).mean()
    else:
        macro_f1 = f1_score(y_np.astype(np.int64), y_pred, average='macro', zero_division=0)
        acc = (y_pred == y_np.astype(np.int64)).mean()
    return {"macro_f1": float(macro_f1), "acc": float(acc)}

# ----------------------- Benchmarks ------------------------
def make_til_benchmark(
    X0_tr, y0_tr, X0_te, y0_te,
    X1_tr, y1_tr, X1_te, y1_te
):
    ex0_train = _to_avalanche_dataset(X0_tr, y0_tr, task_label=0)
    ex1_train = _to_avalanche_dataset(X1_tr, y1_tr, task_label=1)
    ex0_test  = _to_avalanche_dataset(X0_te, y0_te, task_label=0)
    ex1_test  = _to_avalanche_dataset(X1_te, y1_te, task_label=1)
    return create_generic_benchmark(
        train_datasets=[ex0_train, ex1_train],
        test_datasets=[ex0_test, ex1_test],
        task_labels=[0, 1],
        complete_test_set_only=False
    )

def make_cil_benchmark(
    XA_tr, yA_tr, XA_te, yA_te,
    XB_tr, yB_tr, XB_te, yB_te
):
    multilabel = _is_multilabel(yA_tr)
    if multilabel:
        nA = yA_tr.shape[1]; nB = yB_tr.shape[1]
        def shift_B(y):
            z = np.zeros((y.shape[0], nA + nB), dtype=y.dtype)
            z[:, nA:] = y
            return z
        yA_tr2, yA_te2 = yA_tr, yA_te
        yB_tr2, yB_te2 = shift_B(yB_tr), shift_B(yB_te)
    else:
        nA = int(np.max(yA_tr)) + 1
        yA_tr2, yA_te2 = yA_tr, yA_te
        yB_tr2, yB_te2 = yB_tr + nA, yB_te + nA

    ex0_train = _to_avalanche_dataset(XA_tr, yA_tr2, task_label=0)
    ex1_train = _to_avalanhe_dataset(XB_tr, yB_tr2, task_label=0)  # single head
    ex0_test  = _to_avalanche_dataset(XA_te, yA_te2, task_label=0)
    ex1_test  = _to_avalanche_dataset(XB_te, yB_te2, task_label=0)

    return create_generic_benchmark(
        train_datasets=[ex0_train, ex1_train],
        test_datasets=[ex0_test, ex1_test],
        task_labels=[0, 0],
        complete_test_set_only=False
    )

# ---------------------- Public runners ----------------------
def _build_backbone(spec: BackboneSpec, in_ch: int) -> nn.Module:
    if spec.family == "cnn":
        return CNNBackbone(in_ch=in_ch, **spec.params)
    if spec.family == "resnet":
        return ResNet1D(in_ch=in_ch, **spec.params)
    if spec.family == "vit":
        return MiniViT1D(in_ch=in_ch, **spec.params)
    raise ValueError(spec.family)

def run_task_incremental_avalanche(
    spec: BackboneSpec,
    config: TrainConfig,
    X0_train, y0_train, X0_val, y0_val, X0_test, y0_test, classes0,
    X1_train, y1_train, X1_val, y1_val, X1_test, y1_test, classes1,
    method: str,
) -> Dict[str, Dict[str, float]]:
    set_seed(1337)
    in_ch = _infer_in_ch(X0_train)
    multilabel = _is_multilabel(y0_train)

    # Merge train + val per task (matches your compile-time split behavior but uses more data to train)
    X0_tr = np.concatenate([X0_train, X0_val], axis=0)
    y0_tr = np.concatenate([y0_train, y0_val], axis=0)
    X1_tr = np.concatenate([X1_train, X1_val], axis=0)
    y1_tr = np.concatenate([y1_train, y1_val], axis=0)

    bench = make_til_benchmark(X0_tr, y0_tr, X0_test, y0_test, X1_tr, y1_tr, X1_test, y1_test)

    backbone = _build_backbone(spec, in_ch=in_ch).to(DEVICE)
    head_dims = {
        0: y0_train.shape[1] if multilabel else len(classes0),
        1: y1_train.shape[1] if multilabel else len(classes1),
    }
    model = MultiHeadModel(backbone, head_dims=head_dims, feat_dim=spec.feat_dim, multilabel=multilabel).to(DEVICE)

    mhp = MultiHeadPlugin()
    strat = make_strategy(method, model, multilabel, config.lr, config.ewc_lambda, config.si_c, plugins=[mhp], eval_mb_size=config.batch_size)

    for _ in range(config.epochs):
        model.set_task(0); strat.train(bench.train_stream[0])
    t0_after_t0 = eval_numpy(model, X0_test, y0_test, task_id=0, multilabel=multilabel)

    for _ in range(config.epochs):
        model.set_task(1); strat.train(bench.train_stream[1])
    t1_after_t1 = eval_numpy(model, X1_test, y1_test, task_id=1, multilabel=multilabel)
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
    XA_train, yA_train, XA_val, yA_val, XA_test, yA_test, classesA,
    XB_train, yB_train, XB_val, yB_val, XB_test, yB_test, classesB,
    method: str,
) -> Dict[str, Dict[str, float]]:
    set_seed(1337)
    in_ch = _infer_in_ch(XA_train)
    multilabel = _is_multilabel(yA_train)

    XA_tr = np.concatenate([XA_train, XA_val], axis=0)
    yA_tr = np.concatenate([yA_train, yA_val], axis=0)
    XB_tr = np.concatenate([XB_train, XB_val], axis=0)
    yB_tr = np.concatenate([yB_train, yB_val], axis=0)

    bench = make_cil_benchmark(XA_tr, yA_tr, XA_test, yA_test, XB_tr, yB_tr, XB_test, yB_test)

    backbone = _build_backbone(spec, in_ch=in_ch).to(DEVICE)
    if multilabel:
        out_dim = yA_train.shape[1] + yB_train.shape[1]
    else:
        out_dim = len(classesA) + len(classesB)
    head = LinearHead(spec.feat_dim, out_dim, multilabel)
    model = nn.Sequential(backbone, head).to(DEVICE)

    strat = make_strategy(method, model, multilabel, config.lr, config.ewc_lambda, config.si_c, plugins=None, eval_mb_size=config.batch_size)

    for _ in range(config.epochs):
        strat.train(bench.train_stream[0])
    acc_A_after_A = eval_numpy(model, XA_test, yA_test, task_id=None, multilabel=multilabel)

    for _ in range(config.epochs):
        strat.train(bench.train_stream[1])

    if multilabel:
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

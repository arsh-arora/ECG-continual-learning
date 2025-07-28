#!/usr/bin/env python
# coding: utf-8
"""
Ablation Study over CNN, ResNet, and ViT model families on PTB-XL (super-diagnostic task)

- Imports data splits, class weights, and utilities from PRML_project.py (no dummies).
- Trains variants:
    CNN: x2, x4, x8, x16 (width multipliers)
    ResNet: 18, 34, 50, 101 (basic-block variants; 50/101 use deeper blocks count)
    ViT: Small, Base, Large, Huge (1D ViT with patchifying on time axis)
- Logs: parameters, training time, Macro-F1 to 'ablation_results.csv'

Author: ablation script for Arsh
"""

import os
import time
import gc
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# --- Hard dependencies imported from your project (REAL pipeline, not dummies) ---
from CL_PTBXL import (
    # data & labels prepared at module import
    X_train, X_val, X_test,
    y_train_super, y_val_super, y_test_super,
    class_weight_super, classes_super,

    # training & evaluation utilities (losses/metrics are inside these)
    train_model, evaluate_model,
)

# Optional helper if present in PRML_project; else we compute macro F1 from report dict
try:
    from CL_PTBXL import get_macro_f1  # returns report['macro avg']['f1-score']
except Exception:
    def get_macro_f1(report_dict):
        return report_dict['macro avg']['f1-score']

# -----------------------
# Reproducibility & setup
# -----------------------
SEED = int(os.getenv("ABLATION_SEED", "42"))
tf.keras.utils.set_random_seed(SEED)
tf.config.experimental.enable_op_determinism = False  # set True if you want determinism over speed

# Mixed precision (optional — speed on modern GPUs); comment if you don't want it
try:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

# Strategy (multi-GPU if available)
try:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()
except Exception:
    strategy = tf.distribute.get_strategy()

# -----------------------
# Data dimensions & task
# -----------------------
INPUT_SHAPE = X_train.shape[1:]                 # (timesteps, channels) ; for PTB-XL: (1000, 12)
NUM_CLASSES = y_train_super.shape[1]            # multi-label count
CLASSES = list(classes_super)                   # class names for evaluate_model
BATCH_SIZE = int(os.getenv("ABLATION_BATCH", "64"))
EPOCHS = int(os.getenv("ABLATION_EPOCHS", "25"))

# -------------
# CNN Variants
# -------------
def make_cnn_model(width_mult: int) -> tf.keras.Model:
    """
    A scalable 1D CNN backbone with 4 stages.
    Width is scaled by width_mult: base filters [32, 64, 128, 256] * width_mult
    """
    base = [32, 64, 128, 256]
    filters = [f * width_mult for f in base]

    inputs = layers.Input(shape=INPUT_SHAPE)
    x = inputs
    for f in filters:
        x = layers.Conv1D(f, kernel_size=7, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2, strides=2, padding='same')(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256 * max(1, width_mult // 2))(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs, outputs, name=f"CNN_x{width_mult}")

# ----------------
# ResNet Variants
# ----------------
def residual_block_1d(x, filters, stride=1):
    """Basic 1D residual block with optional downsampling."""
    shortcut = x
    x = layers.Conv1D(filters, kernel_size=3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv1D(filters, kernel_size=3, strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # Downsample if needed
    if shortcut.shape[-1] != filters or stride != 1:
        shortcut = layers.Conv1D(filters, kernel_size=1, strides=stride, padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def make_resnet_model(depth: int) -> tf.keras.Model:
    """
    Basic-block ResNet for 1D signals.
    Note: Uses basic blocks even for 50/101 (parameter counts won't match canonical bottleneck models,
    but depth clearly increases capacity).
    """
    # Blocks per stage mapping (ResNet-18/34 canonical; 50/101 adapted to basic blocks for 1D)
    if depth == 18:
        layers_cfg = [2, 2, 2, 2]
    elif depth == 34:
        layers_cfg = [3, 4, 6, 3]
    elif depth == 50:
        layers_cfg = [3, 4, 6, 3]
    elif depth == 101:
        layers_cfg = [3, 4, 23, 3]
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}")

    filters = [64, 128, 256, 512]

    inputs = layers.Input(shape=INPUT_SHAPE)
    x = layers.Conv1D(64, kernel_size=7, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

    for stage, (f, n_blocks) in enumerate(zip(filters, layers_cfg)):
        for block in range(n_blocks):
            stride = 2 if (stage > 0 and block == 0) else 1
            x = residual_block_1d(x, f, stride=stride)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512)(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs, outputs, name=f"ResNet{depth}1D")

# -----------
# ViT 1D
# -----------
class AddClassToken(layers.Layer):
    def build(self, input_shape):
        self.cls = self.add_weight("cls", shape=(1, 1, input_shape[-1]), initializer="zeros", trainable=True)
        super().build(input_shape)

    def call(self, x):
        batch = tf.shape(x)[0]
        cls = tf.tile(self.cls, [batch, 1, 1])
        return tf.concat([cls, x], axis=1)

def transformer_block(x, dim, num_heads, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
    # LayerNorm
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    # Multi-Head Self-Attention
    h = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads, dropout=attn_drop)(h, h)
    x = layers.Add()([x, h])
    # MLP
    h = layers.LayerNormalization(epsilon=1e-6)(x)
    h = layers.Dense(int(dim * mlp_ratio))(h)
    h = layers.Activation('gelu')(h)
    h = layers.Dropout(drop)(h)
    h = layers.Dense(dim)(h)
    x = layers.Add()([x, h])
    return x

def make_vit_model(size: str) -> tf.keras.Model:
    """
    Vision-Transformer-like encoder for 1D ECG:
      - Patchify time axis by Conv1D (kernel=stride=patch)
      - CLS token + learnable positional embeddings
      - Transformer encoder stack
    """
    size = size.lower()
    cfg = {
        "small": dict(dim=384, depth=12, heads=6,  mlp_ratio=4.0),
        "base":  dict(dim=768, depth=12, heads=12, mlp_ratio=4.0),
        "large": dict(dim=1024, depth=24, heads=16, mlp_ratio=4.0),
        "huge":  dict(dim=1280, depth=32, heads=16, mlp_ratio=4.0),
    }
    if size not in cfg:
        raise ValueError(f"Unsupported ViT size: {size}")
    dim = cfg[size]["dim"]; depth = cfg[size]["depth"]; heads = cfg[size]["heads"]; mlp_ratio = cfg[size]["mlp_ratio"]

    patch_size = 20  # 1000/20=50 patches over time axis
    inputs = layers.Input(shape=INPUT_SHAPE)  # (T, C)
    # Patch embedding over time axis: Conv1D with kernel=stride=patch
    x = layers.Conv1D(filters=dim, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)  # (T/patch, dim)
    # Positional embeddings
    seq_len = x.shape[1]
    pos_embed = tf.Variable(tf.zeros((1, seq_len + 1, dim)), trainable=True, name="pos_embed")
    # Add class (CLS) token
    x = AddClassToken()(x)
    x = x + pos_embed  # broadcasting

    for _ in range(depth):
        x = transformer_block(x, dim=dim, num_heads=heads, mlp_ratio=mlp_ratio)

    # CLS pooling
    x = layers.Lambda(lambda t: t[:, 0])(x)
    x = layers.Dense(dim)(x)
    x = layers.Activation('gelu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='sigmoid', dtype='float32')(x)
    return models.Model(inputs, outputs, name=f"ViT1D_{size.capitalize()}")

# -----------------------------
# Ablation experiment definition
# -----------------------------
CNN_WIDTHS = [2, 4, 8, 16]
RESNET_DEPTHS = [18, 34, 50, 101]
VIT_SIZES = ["small", "base", "large", "huge"]

RESULTS = []  # each: dict(Model, Variant, Params, TrainTimeSec, MacroF1)

def _train_and_eval(model: tf.keras.Model):
    """Train using project's train_model, then evaluate using evaluate_model on test split."""
    start = time.time()
    # train_model signature from PRML_project:
    # train_model(model, X_train, y_train, X_val, y_val, class_weight, batch_size=64, epochs=25)
    _ = train_model(
        model,
        X_train, y_train_super,
        X_val,   y_val_super,
        class_weight_super,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    train_time = time.time() - start
    report = evaluate_model(model, X_test, y_test_super, CLASSES)
    macro_f1 = float(get_macro_f1(report))
    return train_time, macro_f1

def _record(model: tf.keras.Model, family: str, variant: str, train_time: float, macro_f1: float):
    RESULTS.append({
        "Model": family,
        "Variant": variant,
        "Parameters": int(model.count_params()),
        "Train Time (s)": float(train_time),
        "Macro-F1": float(macro_f1),
    })

def _cleanup():
    tf.keras.backend.clear_session()
    gc.collect()

# -------------
# Main routine
# -------------
def run_ablation():
    # CNN family
    for w in CNN_WIDTHS:
        with strategy.scope():
            model = make_cnn_model(w)
        print(f"\n[ABLATION] Training {model.name}")
        t, f1 = _train_and_eval(model)
        _record(model, "CNN", f"x{w}", t, f1)
        _cleanup()

    # ResNet family
    for depth in RESNET_DEPTHS:
        with strategy.scope():
            model = make_resnet_model(depth)
        print(f"\n[ABLATION] Training {model.name}")
        t, f1 = _train_and_eval(model)
        _record(model, "ResNet", f"{depth}", t, f1)
        _cleanup()

    # ViT family
    for size in VIT_SIZES:
        with strategy.scope():
            model = make_vit_model(size)
        print(f"\n[ABLATION] Training {model.name}")
        t, f1 = _train_and_eval(model)
        _record(model, "ViT", size.capitalize(), t, f1)
        _cleanup()

    # Save results
    df = pd.DataFrame(RESULTS, columns=["Model", "Variant", "Parameters", "Train Time (s)", "Macro-F1"])
    df.to_csv("ablation_results.csv", index=False)
    print("\n[DONE] Ablation results saved to ablation_results.csv")
    print(df.to_string(index=False))

if __name__ == "__main__":
    run_ablation()

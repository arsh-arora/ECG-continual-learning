# avalance_ablation.py
# Ablation driver using avalance_cl harness. No extra inputs; relies on CL_PTBXL.py.

import csv
from pathlib import Path

from avalance_cl import (
    set_seed, TrainConfig, model_grid, load_ecg_arrays,
    run_task_incremental_avalanche, run_class_incremental_avalanche
)

def main():
    set_seed(1337)
    data = load_ecg_arrays()

    out_csv = Path("./results_ecg_cl_avalanche.csv")
    if out_csv.exists():
        out_csv.unlink()

    fieldnames = [
        "scenario", "family", "size", "cl_method",
        "t0_acc_after_t0","t0_macroF1_after_t0",
        "t0_acc_after_t1","t0_macroF1_after_t1",
        "t1_acc_after_t1","t1_macroF1_after_t1",
        "t0_forgetting_acc",
        "cil_A_acc_after_A","cil_A_macroF1_after_A",
        "cil_A_acc_after_AplusB","cil_A_macroF1_after_AplusB",
        "cil_B_acc_after_AplusB","cil_B_macroF1_after_AplusB",
        "cil_A_forgetting_acc",
    ]
    with out_csv.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    cfg = TrainConfig(
        epochs=10,         # increase (e.g., 30) for full runs
        batch_size=128,
        lr=1e-3,
        ewc_lambda=2.0,
        si_c=2.0
    )
    cl_methods = ["naive", "ewc", "si"]

    for spec in model_grid():
        for method in cl_methods:
            print(f"\n=== TiL | {spec.family}-{spec.size} | {method} ===")
            til = run_task_incremental_avalanche(
                spec, cfg,
                data["X_train"], data["y_train_super"], data["X_val"], data["y_val_super"], data["X_test"], data["y_test_super"], data["classes_super"],
                data["X_train"], data["y_train_sub"],   data["X_val"], data["y_val_sub"],   data["X_test"], data["y_test_sub"],   data["classes_sub"],
                method=method,
            )
            row_til = {
                "scenario": "TiL",
                "family": spec.family,
                "size": spec.size,
                "cl_method": method,
                "t0_acc_after_t0": round(til["t0_after_t0"]["acc"], 4),
                "t0_macroF1_after_t0": round(til["t0_after_t0"]["macro_f1"], 4),
                "t0_acc_after_t1": round(til["t0_after_t1"]["acc"], 4),
                "t0_macroF1_after_t1": round(til["t0_after_t1"]["macro_f1"], 4),
                "t1_acc_after_t1": round(til["t1_after_t1"]["acc"], 4),
                "t1_macroF1_after_t1": round(til["t1_after_t1"]["macro_f1"], 4),
                "t0_forgetting_acc": round(til["forgetting_t0"]["acc_drop"], 4),
                "cil_A_acc_after_A": "",
                "cil_A_macroF1_after_A": "",
                "cil_A_acc_after_AplusB": "",
                "cil_A_macroF1_after_AplusB": "",
                "cil_B_acc_after_AplusB": "",
                "cil_B_macroF1_after_AplusB": "",
                "cil_A_forgetting_acc": "",
            }
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row_til)

            print(f"\n=== CiL | {spec.family}-{spec.size} | {method} ===")
            cil = run_class_incremental_avalanche(
                spec, cfg,
                data["X_train"], data["y_train_super"], data["X_val"], data["y_val_super"], data["X_test"], data["y_test_super"], data["classes_super"],
                data["X_train"], data["y_train_sub"],   data["X_val"], data["y_val_sub"],   data["X_test"], data["y_test_sub"],   data["classes_sub"],
                method=method,
            )
            row_cil = {
                "scenario": "CiL",
                "family": spec.family,
                "size": spec.size,
                "cl_method": method,
                "t0_acc_after_t0": "",
                "t0_macroF1_after_t0": "",
                "t0_acc_after_t1": "",
                "t0_macroF1_after_t1": "",
                "t1_acc_after_t1": "",
                "t1_macroF1_after_t1": "",
                "t0_forgetting_acc": "",
                "cil_A_acc_after_A": round(cil["acc_A_after_A"]["acc"], 4),
                "cil_A_macroF1_after_A": round(cil["acc_A_after_A"]["macro_f1"], 4),
                "cil_A_acc_after_AplusB": round(cil["acc_A_after_AplusB"]["acc"], 4),
                "cil_A_macroF1_after_AplusB": round(cil["acc_A_after_AplusB"]["macro_f1"], 4),
                "cil_B_acc_after_AplusB": round(cil["acc_B_after_AplusB"]["acc"], 4),
                "cil_B_macroF1_after_AplusB": round(cil["acc_B_after_AplusB"]["macro_f1"], 4),
                "cil_A_forgetting_acc": round(cil["forgetting_A"]["acc_drop"], 4),
            }
            with out_csv.open("a", newline="") as f:
                csv.DictWriter(f, fieldnames=fieldnames).writerow(row_cil)

    print(f"\n[Done] Wrote Avalanche results to {out_csv.resolve()}")

if __name__ == "__main__":
    main()

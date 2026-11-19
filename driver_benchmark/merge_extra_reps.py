"""Merge the results on the extra BF/TF replicates 5-12 (5 Nov 2026) into the main result files.
Originals are copied to results/archive_pre_extra_reps/ first (only once). Idempotent: rows of replicates 5-12
already in a main file are replaced, not duplicated.
Usage: python merge_extra_reps.py <stage>   stage = methods | stress"""
import sys, shutil
from pathlib import Path
import pandas as pd
R = Path(__file__).resolve().parent / "results"
A = R / "archive_pre_extra_reps"; A.mkdir(exist_ok=True)
NEW = {f"{m}_rep{r}" for m in ("BF", "TF") for r in range(5, 13)}
PAIRS = {"methods": [("v2_heldout_new.csv", "v2_heldout.csv"), ("ablation_v2_new.csv", "ablation_v2.csv"),
                     ("ablation_v2_extra_new.csv", "ablation_v2_extra.csv"), ("readout_heldout_new.csv", "readout_heldout.csv"),
                     ("baseline_metrics_new.csv", "baseline_metrics.csv"), ("cellrank_synthetic_new.csv", "cellrank_synthetic.csv")],
         "stress": [("fast_vs_exact_v2_new.csv", "fast_vs_exact_v2.csv")]}
for new, main in PAIRS[sys.argv[1]]:
    if not (A / main).exists():
        shutil.copy(R / main, A / main)
    old = pd.read_csv(R / main); add = pd.read_csv(R / new)
    assert set(add.dataset) <= NEW, (new, set(add.dataset) - NEW)
    out = pd.concat([old[~old.dataset.isin(NEW)], add], ignore_index=True)
    out.to_csv(R / main, index=False)
    print(f"{main}: {len(old)} -> {len(out)} rows (+{len(add)} from {new})")

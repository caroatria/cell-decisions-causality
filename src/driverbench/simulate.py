"""Run BoolODE on the benchmark networks and turn trajectories into snapshot scRNA-seq AnnData.

Each BoolODE 'cell' is a full stochastic trajectory. We keep the trajectory's FINAL state as its
ground-truth future fate (like clonal lineage tracing) and record when it committed, then sample
one (or a few) timepoints per trajectory as the observed snapshot cell.

Usage:
    python -m driverbench.simulate --models BF TF CBF --reps 5 --n-traj 600
    python -m driverbench.simulate --models BF TF --first-rep 5 --reps 8 --n-traj 500   # extra held-out replicates 5-12 (5 Nov 2026)
"""
from __future__ import annotations
import argparse, glob, os, subprocess, sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from driverbench.networks import MODELS

from driverbench.paths import BOOLODE, SIMS as SIM_DIR, SYNTH as DATA_DIR
MODEL_DIR = SIM_DIR / "models"   # BoolODE rule files written from networks.py
STEPS_PER_UNIT = 100  # BoolODE output resolution (columns per simulation time unit)


def run_boolode(model, rep: int, n_traj: int, force: bool = False) -> Path:
    out = SIM_DIR / model.name / f"rep{rep}"
    simdir = out / model.name / "simulations"
    if simdir.exists() and len(list(simdir.glob("E*.csv"))) >= n_traj and not force:
        return simdir
    files = model.write(MODEL_DIR)
    job = {"name": model.name, "simulation_time": model.simulation_time, "num_cells": n_traj,
           "do_parallel": True, "sample_cells": False, "nClusters": 1,
           "sample_pars": True, "sample_std": 0.05, "identical_pars": False, **files}
    cfg = {"global_settings": {"model_dir": str(MODEL_DIR), "output_dir": str(out),
                               "do_simulations": True, "do_post_processing": False, "modeltype": "hill"},
           "jobs": [job],
           "post_processing": {"GenSamples": [{"sample_size": 10, "nDatasets": 1}]}}
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    env = dict(os.environ, BOOLODE_SEED_OFFSET=str(rep * 100_000 + 1))
    r = subprocess.run([sys.executable, "boolode.py", "--config", str(cfg_path)], cwd=BOOLODE,
                       env=env, capture_output=True, text=True)
    if r.returncode != 0 and not simdir.exists():
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    return simdir


def commitment_time(traj: pd.DataFrame, switch: list[str]) -> tuple[str | None, float]:
    """Fate = switch gene highest at the end; commit time = first time after which the winner
    stays ahead of every competitor by at least half its final margin."""
    end = traj.loc[switch].iloc[:, -10:].mean(axis=1)
    win = end.idxmax()
    margin_final = end[win] - end.drop(win).max()
    if end[win] < 0.5 or margin_final < 0.3:   # never decided
        return None, np.nan
    diff = traj.loc[win].values - traj.loc[[s for s in switch if s != win]].values.max(axis=0)
    ok = diff >= 0.5 * margin_final
    # last index where condition fails; commitment is right after it
    fails = np.where(~ok)[0]
    idx = 0 if len(fails) == 0 else fails[-1] + 1
    return win, idx / STEPS_PER_UNIT


def build_snapshot(model, simdir: Path, rep: int, cells_per_traj: int = 1, n_null: int = 60,
                   scale: float = 15.0, dropout: float = 0.0) -> ad.AnnData:
    rng = np.random.default_rng(1000 + rep)
    files = sorted(simdir.glob("E*.csv"), key=lambda p: int(p.stem[1:]))
    rows, meta = [], []
    for f in files:
        traj = pd.read_csv(f, index_col=0)
        traj = traj.loc[[g for g in model.rules]]          # mRNA rows only, fixed gene order
        info = {"traj": int(f.stem[1:])}
        for d, sw in model.fate_genes.items():
            fate, tc = commitment_time(traj, sw)
            info[f"fate_{d}"], info[f"tcommit_{d}"] = fate, tc
        for d, (pd_, pf) in model.parents.items():   # nested decisions exist only in their parent lineage
            if info.get(f"fate_{pd_}") != pf:
                info[f"fate_{d}"], info[f"tcommit_{d}"] = None, np.nan
        T = traj.shape[1]
        for col in rng.choice(T, size=cells_per_traj, replace=False):
            rows.append(traj.iloc[:, col].values)
            meta.append({**info, "sim_time": col / STEPS_PER_UNIT})
    Xtrue = np.clip(np.vstack(rows), 0, None)
    obs = pd.DataFrame(meta)
    # technical noise: library size variation + Poisson sampling (+ optional dropout)
    lib = rng.lognormal(0, 0.3, size=(Xtrue.shape[0], 1))
    counts = rng.poisson(Xtrue * scale * lib)
    nulls = rng.poisson(rng.lognormal(np.log(scale * 0.8), 0.6, size=(1, n_null)) * lib
                        * rng.lognormal(0, 0.4, size=(Xtrue.shape[0], n_null)))
    counts = np.hstack([counts, nulls]).astype(np.float32)
    if dropout > 0:
        counts[rng.random(counts.shape) < dropout] = 0
    genes = list(model.rules) + [f"null{i + 1}" for i in range(n_null)]
    var = pd.DataFrame(index=genes)
    var["role"] = [model.roles.get(g, "null") for g in genes]
    var["decision"] = [model.decision.get(g, "") for g in genes]
    var["kind"] = [model.kind.get(g, var.loc[g, "role"]) for g in genes]
    a = ad.AnnData(X=counts.copy(), obs=obs.set_index(pd.Index([f"c{i}" for i in range(len(obs))])), var=var)
    a.layers["counts"] = counts
    a.layers["true_expression"] = np.hstack([Xtrue, np.zeros((Xtrue.shape[0], n_null))]).astype(np.float32)
    tot = counts.sum(1, keepdims=True); tot[tot == 0] = 1
    a.X = np.log1p(counts / tot * np.median(tot)).astype(np.float32)
    for d in model.fate_genes:
        a.obs[f"fate_{d}"] = a.obs[f"fate_{d}"].astype("category")
        a.obs[f"committed_{d}"] = a.obs["sim_time"] >= a.obs[f"tcommit_{d}"]
    a.uns["model"] = model.name
    a.uns["fate_genes"] = {d: list(v) for d, v in model.fate_genes.items()}
    a.uns["rep"] = rep
    a.uns["parents"] = {d: list(v) for d, v in model.parents.items()}
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--first-rep", type=int, default=0, help="simulate replicates first-rep .. first-rep+reps-1")
    ap.add_argument("--n-traj", type=int, default=600)
    ap.add_argument("--dropout", type=float, default=0.0)
    args = ap.parse_args()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in args.models:
        model = MODELS[name]()
        for rep in range(args.first_rep, args.first_rep + args.reps):
            simdir = run_boolode(model, rep, args.n_traj)
            a = build_snapshot(model, simdir, rep, dropout=args.dropout)
            tag = f"{name}_rep{rep}" + (f"_drop{args.dropout}" if args.dropout else "")
            a.write_h5ad(DATA_DIR / f"{tag}.h5ad")
            fates = {d: a.obs[f"fate_{d}"].value_counts(dropna=False).to_dict() for d in model.fate_genes}
            print(f"{tag}: {a.n_obs} cells x {a.n_vars} genes; fates {fates}; "
                  f"committed {[round(a.obs[f'committed_{d}'].mean(), 2) for d in model.fate_genes]}", flush=True)


if __name__ == "__main__":
    main()

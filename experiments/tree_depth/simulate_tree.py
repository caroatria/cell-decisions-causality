"""Simulate random nested lineage trees (networks_tree.py) for the depth experiment (17 Nov 2026).

For every split probability p and depth D = 1..4, N networks are drawn (each its own random topology and seed),
simulated with BoolODE and turned into a snapshot exactly like the other synthetic data (simulate.build_snapshot).
Trajectories are stored at every 20th time step (0.2 time units), rounded to 3 decimals, and deleted after the snapshot is built,
to save disk. Number of trajectories grows with the number of terminal fates (360 per fate, at least 1500; the first run used 120 / 500).
Output: data_tree/T<p>_d<D>_n<i>.h5ad  (e.g. T08_d3_n2 = p 0.8, depth 3, network 2)

Usage: python experiments/tree_depth/simulate_tree.py --p 0.8 0.5 --depths 1 2 3 4 --n 5 [--first 0]
"""
from __future__ import annotations
import argparse, os, shutil, time
from pathlib import Path
from driverbench import simulate
from networks_tree import draw_tree, tree_model

from driverbench.paths import ROOT, TREE
OUT = TREE
THIN = 20
TRAJ_PER_FATE = 360   # 17 Nov 2026: 3x the first run (120), so deep decisions get ~3x more cells (first run kept in data_tree_1x/)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, nargs="+", default=[0.8, 0.5])
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--keep-sims", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    os.environ["BOOLODE_THIN"] = str(THIN)
    os.environ["BOOLODE_FLOAT_FMT"] = "%.3f"
    simulate.STEPS_PER_UNIT = 100 // THIN
    for p in args.p:
        for d in args.depths:
            for i in range(args.first, args.first + args.n):
                tag = f"T{round(p * 10):02d}_d{d}_n{i}"
                if (OUT / f"{tag}.h5ad").exists():
                    continue
                seed = int(p * 1000) * 100 + d * 10 + i        # topology seed
                internal = draw_tree(p, d, seed)
                model = tree_model(internal, tag)
                n_traj = max(1500, TRAJ_PER_FATE * (len(internal) + 1))
                t0 = time.time()
                rep = seed % 997                                    # BoolODE seed offset + snapshot noise seed
                simdir = simulate.run_boolode(model, rep, n_traj)
                a = simulate.build_snapshot(model, simdir, rep)
                a.uns["model"], a.uns["p"], a.uns["depth"] = tag, p, d
                a.uns["topology"] = " ".join(sorted(internal, key=lambda f: (len(f), f)))
                a.write_h5ad(OUT / f"{tag}.h5ad")
                if not args.keep_sims:
                    shutil.rmtree(simulate.SIM_DIR / tag, ignore_errors=True)
                fates = {k: a.obs[f"fate_{k}"].value_counts().to_dict() for k in model.fate_genes}
                print(f"{tag}: {len(internal)} splits, {a.n_obs} cells x {a.n_vars} genes, "
                      f"{time.time() - t0:.0f}s; fates {fates}", flush=True)


if __name__ == "__main__":
    main()

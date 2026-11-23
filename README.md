# cell-decisions-causality: finding driver genes of cell-fate decisions

Which genes **drive** a cell-fate decision, as opposed to merely marking its outcome? This repository holds
everything for the IJCAI-27 paper (abstract 4 Jan, paper 11 Jan 2027) on **COUNTERFATE**: timing-constrained
interventional counterfactuals that rank genes by how much *setting* them (an intervention) shifts an undecided
cell's fate, instead of how well they *predict* the fate. It also holds the benchmark we built to test it,
real-data validations, and a case study on the sponge *Suberites domuncula*.

The central idea, and the central test: **drivers** (the regulators that cause a fate decision) and **markers**
(downstream genes that are equally fate-specific but cause nothing) look alike in expression data. Most methods
rank markers as high as drivers. Every benchmark here asks one question: *does a method rank drivers above markers?*

New here? Read this file top to bottom (about 10 minutes), then open `notebooks/results_and_figures.ipynb`.

---

## 1. Setup (once)

```bash
cd cell-decisions-causality
conda env create -f environment.yml        # Python 3.12 + everything in requirements.txt + this package (editable)
conda activate driver-benchmark
python -m ipykernel install --user --name driver-benchmark --display-name "Python (driver-benchmark)"
```
Without conda: `pip install -r requirements.txt && pip install -e .`

The `pip install -e .` step makes the `driverbench` package (in `src/driverbench/`) importable from anywhere,
so every script can be run from the repository root as `python experiments/<topic>/<script>.py`.

**macOS:** set these before running anything heavy (avoids an OpenMP crash); the run scripts set them for you:
```bash
export OMP_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue KMP_DUPLICATE_LIB_OK=TRUE
```

Optional, only when you need them:

| What | Command | Needed for |
|---|---|---|
| BoolODE simulator (pinned + patched) | `bash scripts/setup_boolode.sh` | simulating new networks |
| CellRank 2, CellOracle, CEFCON (own environments, ~6 GB, needs `uv`) | `bash scripts/setup_competitors.sh` | rerunning the published competitors |
| Real data | see `data/README.md` | Nestorowa, LARRY, sponge |

---

## 2. Folder layout

```
cell-decisions-causality/
├── README.md               you are here
├── environment.yml, requirements.txt, pyproject.toml
├── src/driverbench/        the library: method, baselines, benchmark harness (import from here, don't copy)
├── experiments/            one folder per experiment: its run, evaluation and figure scripts
│   ├── synthetic/          main benchmark on BoolODE networks (headline table, method figure, runtime)
│   ├── ablations/          ablations of the frozen method
│   ├── tree_depth/         random nested lineage trees of depth 1-4
│   ├── tf_prior/           all methods given the same TF list (incl. CellOracle / CEFCON on synthetic data)
│   ├── real_data/          Nestorowa 2016, LARRY 2020 (lineage tracing), Paul15 (diagnostic only)
│   └── sponge/             Suberites domuncula case study
├── competitors/            run_cellrank.py, run_celloracle.py, run_cefcon.py: run inside each tool's own env
├── scripts/                setup + "rerun everything" scripts
├── notebooks/              results_and_figures.ipynb: every table and figure, built from results/*.csv only
├── results/                one CSV per experiment (tracked in git; the paper's numbers come from here)
├── figures/                PNG (slides) + PDF (paper)
├── external/               BoolODE pin + patch, mouse TF list
├── docs/                   results_log.md (dated lab notebook of every experiment), file_map.md (old -> new paths)
├── archive/                superseded work, reference only (see archive/README.md):
│                           2026-02 group exploration, 2026-11 superseded benchmark code and runs
└── data/, logs/, cache/, sims/, tools/     not in git (large or regenerated; see data/README.md)
```

### The library (`src/driverbench/`)

| Module | What it is |
|---|---|
| `paths.py` | every folder the code uses. **Never hard-code a path; import it from here.** |
| `networks.py` | the BoolODE gene networks (BF, TF, CBF, HSC) with each gene's role: driver, marker, mediator, ... |
| `simulate.py` | BoolODE simulation -> snapshot scRNA-seq `.h5ad` with ground truth. `python -m driverbench.simulate` |
| `evaluate.py` | `contexts()` (one per fate decision) and `metrics()` (driver-vs-marker AUPRC etc.); CLI runs any scorer on `data/synthetic/` |
| `baselines.py` | all baselines (DE, fate correlation, RF+SHAP, observational counterfactual, ...), the fate classifier and the decision region |
| `probe_interventional.py` | the method's core: pseudotime, forward model, intervention + rollout |
| `method_timing.py` | onset estimate + timing rule; **`score_timing_cf_v2` is the frozen method (COUNTERFATE)** |
| `score_modes.py` | turning probability shifts into gene scores (`perfate` is the one we use) |
| `competitors.py` | adapters that export data, call `competitors/run_<tool>.py` in the tool's environment, read scores back |
| `realdata.py` | shared preprocessing for real data and the literature-based evaluation (core / extended regulator lists) |

---

## 3. The benchmark in one page

**Synthetic networks** (`data/synthetic/`, made by `driverbench.simulate`). BoolODE simulates Boolean gene
networks as stochastic ODEs; each cell is one time point of one simulated trajectory.

| Model | Structure | Decisions |
|---|---|---|
| BF | progenitor -> toggle switch A ⊣⊢ B | D1 (drivers A, B) |
| TF | three-way switch A, B, C | D1 |
| CBF | A ⊣⊢ B, then inside the A lineage a timer q1 -> q2 starts a second switch C ⊣⊢ D | D1, D2 (nested) |
| HSC | curated blood-stem-cell network (BEELINE) | D1-D3 |

Every network also has many markers per fate, including *hard negatives* (`var.kind`: `fast` targets that
respond almost as soon as the driver, and `repressed` genes that are ON in progenitors), a fate-independent
oscillator (nuisance) and null genes.

**Ground truth in each `.h5ad`:** `var.role`, `var.decision` (who is a driver of which decision);
`obs.fate_D*` (true future fate), `obs.tcommit_D*`, `obs.committed_D*`, `obs.sim_time`.

**The honesty rule (important).** A method may use the expression matrix `X` and the annotated terminal cells
(`ctx.terminal`: committed cells late in time, standing in for clusters a biologist would annotate).
It must **not** use `obs.sim_time`, `obs.fate_*`, `obs.tcommit_*` or `obs.committed_*`. Methods that do are
called `ORACLE_*` and are reference lines only.

**Dev / test split (important).** Every setting of the method was chosen on BF and TF replicates 0-2 only.
The **44 held-out decisions** are BF/TF replicates 3-12, all CBF and all HSC. Never tune anything on them.
The method is frozen (`score_timing_cf_v2`, K = 3, `perfate` scoring).

**Metric:** `auprc_vs_markers` = AUPRC of drivers vs the markers and mediators of the same decision (headline);
also `auprc_vs_fast_targets` (hardest), `auprc_all`, `prec_at_k`, `best_driver_rank`, each with its chance level.
Ties are broken at random and averaged over 50 draws.

**Score one dataset yourself:**
```python
import anndata as ad
from driverbench.evaluate import contexts, metrics
from driverbench.method_timing import score_timing_cf_v2
a = ad.read_h5ad("data/synthetic/HSC_rep0.h5ad")
for ctx in contexts(a, seed=0):                      # one context per fate decision
    scores = score_timing_cf_v2(a, ctx)              # pd.Series: gene -> driver score
    print(ctx.decision, scores.nlargest(5).index.tolist(),
          round(metrics(scores, a.var, ctx.decision)["auprc_vs_markers"], 2))
```

**Add a new method:** write `def score_mymethod(adata, ctx) -> pd.Series` (one score per gene, higher = more
driver-like) and evaluate it exactly like the snippet above, on the same 44 held-out decisions. Follow the
honesty rule, tune only on the dev replicates, and write results to `results/<experiment>.csv`.

---

## 4. Experiments and where their numbers live

Run every command from the repository root. Runtimes are for 2 CPU cores.

| Experiment | Run | Results | Figure |
|---|---|---|---|
| Main synthetic benchmark (44 decisions) | `bash scripts/run_all.sh` (all; ~3 h with simulation, `--no-sim` ~45 min) | `baseline_metrics.csv`, `hsc_metrics.csv`, `v2_heldout.csv`, `cellrank_synthetic.csv` | notebook, `fig_synthetic_competitors.png` |
| Extra replicates 5-12 | `bash experiments/synthetic/run_extra_reps.sh` | merged into the files above | |
| Method overview figure | `python experiments/synthetic/fig_method_overview.py` | | `fig_method_overview.*` |
| Runtime scaling | `python experiments/synthetic/scaling_v2.py` | `scaling_v2_*.csv` | notebook |
| Ablations | `python experiments/ablations/ablation_v2.py`, `ablation_v2_extra.py`, `readout_check.py heldout` | `ablation_v2*.csv`, `readout_heldout.csv` | `fig_ablation_v2.py` |
| Tree depth (nested splits, p = 0.8 / 0.5) | `bash experiments/tree_depth/run_tree_all.sh` (several hours) | `tree_depth.csv` (`tree_depth_1x.csv` = first run with 1/3 of the cells) | `fig_tree_depth.*`, `fig_tree_level.*` |
| TF prior for every method (incl. CellOracle, CEFCON) | `python experiments/tf_prior/eval_tfprior.py` (~2 h) | `tfprior_synthetic.csv` | |
| Nestorowa 2016 | `python experiments/real_data/nestorowa_prepare.py`, then `nestorowa_eval.py` | `nestorowa_eval*.csv` | notebook, `fig_nestorowa.*` |
| LARRY 2020 (clonal fate) | `python experiments/real_data/larry_prepare.py`, then `SEED=0 TAG=s0 python experiments/real_data/larry_eval.py` (and s1, s2), `larry_bootstrap.py` | `larry_*.csv` | notebook, `fig_larry.*` |
| Competitors on real data | `bash scripts/run_cellrank_all.sh`, `bash experiments/real_data/run_larry_competitors.sh <seed>` | `*_cellrank.csv`, `*_competitors.csv` | |
| Sponge case study | `bash experiments/sponge/run_sponge_experiments.sh` | `exp_*` / `sponge_*` CSVs, `sponge_compare.csv` | `experiments/sponge/fig_sponge.py` |
| All tables + figures | open the notebook, or `bash scripts/run_all.sh --figures` | reads `results/` only | `figures/` |

Paul15 (`python -m driverbench.realdata <paul15.h5ad>`, `experiments/real_data/paul15_*.py`) is a diagnostic
dataset only and is not in the paper.

**Verified paper numbers** are in the project doc *Verified result numbers for the IJCAI-27 draft*. Two result
files are stale and must not be quoted: `results/ablation_v2_summary.csv` and `results/overview_all_methods.csv`
(they predate the extra replicates).

### Headline (44 held-out synthetic decisions, driver-vs-marker AUPRC)

| COUNTERFATE (ours) | fate correlation | CellRank 2 | obs. counterfactual | DE terminal | RF + SHAP | random |
|---|---|---|---|---|---|---|
| **0.82** | 0.67 | 0.61 | 0.59 | 0.50 | 0.49 | 0.19 |

---

## 5. Working conventions

- **Code goes in the right place.** Reusable logic -> `src/driverbench/`. A script that produces one result
  file or figure -> `experiments/<topic>/`. Scripts in the same experiment folder may import each other.
- **Paths:** `from driverbench.paths import SYNTH, REAL, RESULTS, FIGURES, ...`; never `"../data/..."`.
- **Results are append-only history.** Write a new file (or a new column) instead of overwriting a result
  someone quotes. Record what you ran and why in `docs/results_log.md` (dated entry, newest at the bottom).
- **Long runs are resumable** (`eval_tree.py`, `eval_tfprior.py`, `simulate_tree.py` skip finished work), so an
  interrupted run can simply be restarted.
- **Don't commit data.** `.h5ad` files, `data/`, `sims/`, `cache/`, `logs/`, `tools/` are git-ignored.
- **Branches:** make a branch for anything that changes `src/driverbench/` and open a pull request; the
  method is frozen for the paper, so changes to `method_timing.py` / `probe_interventional.py` need a discussion first.

## 6. Known open issues (23 Nov 2026)

- **Tie handling on real data.** `realdata.evaluate` breaks score ties at hits@k by gene order and prints
  floored average ranks. COUNTERFATE's per-fate score creates ties by design (one gene per fate per rank),
  which inflates Nestorowa core hits@10 from 4.5 (expected) to 5. Fix pending: random tiebreak for hits@k,
  ranks printed with .5.
- **Tree-depth rerun with 3x cells** (23 Nov 2026): reduced to p = 0.8, depths 1-3, networks 0-4; `tree_depth.csv`
  will hold it, `tree_depth_1x.csv` keeps the first run. See `experiments/tree_depth/README.md`.
- **`larry_prepare.py` is new** (the original h5ad was built interactively); check that its output matches
  130,887 cells x 25,289 genes before rerunning LARRY.
- CellOracle on synthetic data uses knockouts only; an overexpression run is a planned sensitivity check.

---

## 7. Sponge cluster annotation (old atlas, `suberites_domuncula_cytotrace_SUB4.h5ad`)

| cluster | cell type |
|---|---|
| 0 | terminal skin (pinacocytes) |
| 1 | large transition group |
| 2, 3 | terminal skeletal cells (sclerocytes) |
| 5 | stem cells |
| 6 | terminal digestive cells (metabolocytes) |
| 8 | possibly terminal digestive cells, possibly something else |
| 12 | terminal current-generating cells (choanocytes) |

The case study uses its own cluster mapping (`experiments/sponge/run_sponge_experiments.sh`) and reads the atlas
from `data/raw/suberites_domuncula/`.

*History:* until 23 Nov 2026 all current work lived in a `driver_benchmark/` subfolder; it was moved one level up
unchanged. `docs/file_map.md` maps older file names to the current ones.

> **Lab notebook (moved from README.md on 19 Nov 2026).** Dated record of every experiment, newest at the bottom.
> File names refer to the old flat layout (e.g. `eval_v2.py`, `data/`); see `file_map.md` for the new paths.
> The current overview and how-to-run is `../README.md`.

# Driver-vs-marker benchmark

Synthetic benchmark for methods that claim to find **driver genes of cell-fate decisions**.
Every simulated network has known drivers (a fate switch) and many known **markers**
(downstream readouts that are as fate-specific as the drivers but don't cause the fate).
The headline question for any method: *does it rank drivers above markers?*

## Folder layout
```
run_all.sh                     reproduces everything (see "How to run")
requirements.txt
notebooks/
  results_and_figures.ipynb    all result tables + figures 1-5, built from results/*.csv
-- benchmark ------------------------------------------------------------------
networks.py                    Boolean gene networks with known roles (driver / marker / ...)
simulate.py                    BoolODE simulation -> snapshot scRNA-seq AnnData with ground truth
boolode_commit.txt, boolode_patches.diff   pinned BoolODE version + our 3 small patches
evaluate.py                    runs scorers on datasets, computes the metrics
-- methods ---------------------------------------------------------------------
probe_interventional.py        OUR METHOD, steps 1-7 (pseudotime, fate classifier, undecided
                               cells, forward model, intervene + fast-forward, score). Without a
                               timing mask this is "ours without timing rule" (the first version).
method_timing.py               OUR METHOD, version 1: adds the onset estimate + timing rule
                               (scorer OURS_timing_cf = exact; OURS_timing_cf_fast = with speed-ups,
                               use this for real data with thousands of genes)
baselines.py                   all baselines (CellRank-style, DE, SHAP/BUNGEE fixed, counterfactuals, ...)
sweep_timing.py                dev-only tuning of the onset threshold
scaling.py                     runtime measurements
paul15_eval.py                 real-data test on Paul15 vs literature regulators (prepared, not yet run)
-- outputs ---------------------------------------------------------------------
data/       20 synthetic + 5 HSC datasets (.h5ad)
results/    metrics per dataset x decision x method (CSV)
figures/    fig1-fig5 as PNG (slides) and PDF (paper)
logs/       simulation logs
archive/    superseded runs: first "easy" benchmark (data_easy/), early test/dev runs,
            the old standalone figure script
```

## How to run
```bash
pip install -r requirements.txt
bash run_all.sh            # everything: BoolODE setup, simulation (~2-3 h on 2 cores), methods, figures
bash run_all.sh --no-sim   # reuse data/, rerun all methods + figures (~30-45 min)
bash run_all.sh --figures  # only rebuild tables and figures from results/ (seconds)
```
Or open `notebooks/results_and_figures.ipynb` and run all cells. It reads only `results/*.csv`.

Run a single method on one dataset:
```python
import anndata as ad
from evaluate import contexts, metrics
from method_timing import score_timing_cf
a = ad.read_h5ad("data/HSC_rep0.h5ad")
for ctx in contexts(a, seed=0):                  # one context per fate decision
    scores = score_timing_cf(a, ctx)             # pd.Series: gene -> driver score
    print(ctx.decision, scores.sort_values(ascending=False).head(5).index.tolist(),
          metrics(scores, a.var, ctx.decision)["auprc_vs_markers"])
```

## Networks
| Model | Structure | Drivers | Markers |
|---|---|---|---|
| BF  | progenitor p1→p2→p3 → toggle switch A ⊣⊢ B | A, B (decision D1) | 12 per fate: 3 fast + 3 slow direct targets, 4 second-layer, 2 fast repressed |
| TF  | three-way switch A, B, C | A, B, C (D1) | 10 per fate |
| CBF | A ⊣⊢ B; then in the A lineage A→q1→q2→ switch C ⊣⊢ D | D1: A, B. D2: C, D | 10 per fate; q1/q2 = mediators |
| all | + repressilator c1–c3 with readouts c4, c5 (fate-independent, cell-cycle-like nuisance), + 60 null genes | | |

**Hard negatives (`var.kind`):**
- `fast`: direct targets with a low activation threshold. They respond almost as soon as the driver does.
- `repressed`: genes switched off by the fate TF. They are ON in progenitors, so they also diverge early.

A method that only checks "diverges early" will struggle with these. A method that recovers direction (driver → target) should not.

`networks.HARD = False` reproduces the first, easy version (slow targets only, no nuisance program). In the easy version, a CellRank-style correlation restricted to uncommitted cells already scores perfectly, which is why the hard version exists. The easy datasets are kept in `archive/data_easy/`.

The switches are self-activating and mutually inhibiting, which is the standard bistable fate motif (cf. BoolODE `dyn-bifurcating`, Gata1/PU.1).

## Ground truth stored in each `.h5ad`
- `var.role`, `var.decision`: driver / marker / progenitor / mediator / null.
- `obs.fate_D*`: the **true future fate** of the trajectory each cell was sampled from. This is the analogue of lineage tracing.
- `obs.tcommit_D*`, `obs.committed_D*`: when the trajectory committed, and whether the cell is past that point.
- `obs.sim_time`: true time. Methods should not use it except to pick a root or annotated terminal cells.
- `layers.counts` (Poisson counts with library-size noise), `X` (log-normalised), `layers.true_expression`.

## What methods may use (to stay honest)
Each method gets the expression matrix plus **annotated terminal cell types** (committed cells in the last 30% of time). These stand in for the clusters a biologist would annotate. Methods that use true future fate are labelled `ORACLE` and serve as references only.

## Metrics (per dataset × decision)
- `auprc_vs_markers`: drivers vs the downstream genes of the same decision. **This is the headline metric.**
- `auprc_vs_fast_targets`: drivers vs fast and repressed direct targets only. This is the hardest comparison.
- `auprc_all`: drivers vs all other genes.
- `prec_at_k`: precision in the top-k genes, with k = number of drivers.
- `best_driver_rank`: rank of the highest-scoring driver.
- Chance levels are reported next to each AUPRC.

## Patches to BoolODE (Murali-group/BoolODE, master)
1. `BoolODE/run_experiment.py`: removed `n_jobs=8` from `KMeans` (scikit-learn ≥ 1.0).
2. `BoolODE/run_experiment.py`: per-cell seed = `cellid + $BOOLODE_SEED_OFFSET`, so replicates differ.
3. `boolode.py`: global numpy/random seed from `$BOOLODE_SEED_OFFSET`, for parameter sampling.

## First results (22 Oct 2026; hard networks, 5 replicates × 500 cells)
Mean ± sd of `auprc_vs_markers`. Chance is about 0.08–0.14.

| method | BF-D1 | CBF-D1 | CBF-D2 | TF-D1 | mean |
|---|---|---|---|---|---|
| ORACLE (true future fate, uncommitted cells) | 1.00 ± 0.00 | 0.93 ± 0.09 | 0.72 ± 0.32 | 0.79 ± 0.13 | 0.86 |
| fate_corr (CellRank lineage-drivers style) | 0.94 ± 0.13 | 0.32 ± 0.15 | 0.68 ± 0.18 | 0.85 ± 0.05 | **0.70** |
| fate_corr_decision_region | 0.85 ± 0.15 | 0.25 ± 0.07 | 0.64 ± 0.23 | 0.75 ± 0.07 | 0.62 |
| obs_counterfactual (Wachter) | 0.65 ± 0.27 | 0.61 ± 0.16 | 0.48 ± 0.17 | 0.67 ± 0.12 | 0.60 |
| de_terminal | 0.47 ± 0.06 | 0.64 ± 0.25 | 0.39 ± 0.06 | 0.64 ± 0.14 | 0.54 |
| rf_shap_fixed (BUNGEE corrected) | 0.40 ± 0.35 | 0.65 ± 0.26 | 0.34 ± 0.18 | 0.64 ± 0.25 | 0.51 |
| absorption_corr_decision_region | 0.47 ± 0.38 | 0.27 ± 0.11 | 0.48 ± 0.34 | 0.65 ± 0.28 | 0.47 |
| PROBE_interventional_linear | 0.20 ± 0.05 | 0.26 ± 0.12 | 0.47 ± 0.25 | 0.48 ± 0.23 | 0.35 |
| random | 0.11 | 0.24 | 0.13 | 0.22 | 0.17 |

**Takeaways**
- **The bar to beat is plain fate-probability correlation** (0.70 mean). It is the best realistic method in 9/20 cases, but it breaks down on the nested decision (CBF-D1: 0.32 vs oracle 0.93).
- **Variance across replicates is large.** Use more replicates and cells before drawing paper-level conclusions.
- **The naive interventional probe** (sparse linear time-lag forward model + clamping) is the weakest realistic method. It learns marker → driver edges from collinearity. Neither restricting the fit to uncommitted cells nor a TF prior fixed it (tested on BF-easy). The forward model is where framing B needs real novelty.
- **"Fast" direct targets were meant as hard negatives but are not the hardest.** Slow, strongly saturated markers and repressed genes outrank drivers more often (`var.kind` lets you check this). Baseline rankings also flip between the easy and hard designs, so conclusions should be checked on a second simulator family (SERGIO / scMultiSim).

## Our method, version 1: timing-constrained interventional counterfactuals (22 Oct 2026)
`method_timing.py` → scorer `OURS_timing_cf`. Same pipeline as the probe, but the forward model may only use gene x to predict gene y if x's label-free onset along pseudotime (windowed-variance rise reaching 20% of its maximum) is not later than y's. This blocks marker → driver edges.

**Protocol:**
- The onset threshold was chosen on BF + TF replicates 0–2 (`sweep_timing.py`, `results/dev_sweep_timing.csv`).
- BF + TF replicates 3–4 were not used for tuning.
- CBF was held out and evaluated exactly once.

| | BF-D1 | CBF-D1 | CBF-D2 | TF-D1 | mean | dev (tuned) | dev (unseen reps) | held-out CBF |
|---|---|---|---|---|---|---|---|---|
| ORACLE | 1.00 | 0.93 | 0.72 | 0.79 | 0.86 | 0.87 | 0.92 | 0.83 |
| **OURS_timing_cf** | 0.65 | **0.69** | 0.65 | 0.81 | **0.70** | 0.77 | 0.67 | **0.67** |
| fate_corr | **0.94** | 0.32 | **0.68** | **0.85** | **0.70** | 0.91 | 0.87 | 0.50 |
| obs_counterfactual | 0.65 | 0.61 | 0.48 | 0.67 | 0.60 | | | 0.54 |
| PROBE (no timing constraint) | 0.20 | 0.26 | 0.47 | 0.48 | 0.35 | | | 0.37 |

Paired against fate_corr over 20 dataset × decision cases: 8 wins, 3 ties, 9 losses, mean difference 0.00 (Wilcoxon p = 0.85).
- Nested decisions (held-out CBF): ours wins 7/10, mean +0.17.
- Simple bifurcation (BF): ours loses 5/5, mean −0.29.

**What we learned**
- **The timing constraint is what makes the interventional idea work** (0.35 → 0.70). Onset ordering alone, without the forward model, is poor (0.40 BF, 0.29 TF), so the gain is not timing by itself.
- **The remaining errors are almost all fast direct targets.** They reach their own relative onset before the driver, so the constraint lets them "regulate" it.
- **The method is sensitive to the onset threshold** (BF: 0.15–0.67 across thresholds 0.05–0.3). Performance on dev replicates the threshold was not tuned on dropped (0.77 → 0.67). Expect some optimism in the tuned numbers.
- **Not yet a win overall.** It is complementary to fate_corr: better on nested decisions, worse on a single clean bifurcation.

## Validation on a curated biological network: BEELINE HSC model (22 Oct 2026)
`networks.hsc_curated()`: the hematopoietic stem cell Boolean model (Krumsiek et al. 2011, BoolODE `data/HSC.txt`), with the wiring unchanged.
- **Nested decisions and drivers** (literature pairs):
  - D1 MEP vs GMP: Gata1 vs Pu1
  - D2 Ery vs Mk, within the Gata1 lineage: Eklf vs Fli1
  - D3 Mono vs Gran, within the Pu1 lineage: EgrNab vs Gfi1
- **Other genes:** the remaining network genes (Gata2, Fog1, Scl, Cebpa, cJun) are labelled `network` and excluded from the driver-vs-marker metric. Synthetic readout markers and the nuisance oscillator are added as before.
- **Untuned:** this network was never used for tuning. The method setting is the one chosen on BF/TF dev replicates.
- **Simulation caveats:**
  - About 38% of trajectories end in an all-off state of this model and get no fate label.
  - Fate proportions vary a lot between replicates.
  - In HSC_rep2, D2 has only 2 Ery cells and no annotated Ery terminals, so it is undefined and excluded for all methods. That leaves 14 dataset × decision cases.

| method | D1 MEP/GMP | D2 Ery/Mk | D3 Mono/Gran | mean |
|---|---|---|---|---|
| **OURS_timing_cf** | **0.70** | **0.74** | 0.52 | **0.65** |
| ORACLE (true future fate) | 0.78 | 0.40 | 0.69 | 0.63 |
| rf_shap_fixed | 0.66 | 0.38 | 0.54 | 0.52 |
| fate_corr | 0.42 | 0.40 | 0.63 | 0.49 |
| de_terminal | 0.56 | 0.43 | 0.47 | 0.48 |
| obs_counterfactual | 0.36 | 0.39 | **0.71** | 0.48 |
| PROBE (no timing constraint) | 0.39 | 0.54 | 0.32 | 0.42 |
| random | 0.15 | 0.18 | 0.27 | 0.20 |

**Paired comparisons (14 cases).** Wins/losses for OURS against each baseline, with mean AUPRC difference and Wilcoxon p:

| baseline | wins/losses | mean difference | p |
|---|---|---|---|
| fate_corr | 9/5 | +0.15 | 0.20 |
| rf_shap_fixed | 8/6 | +0.11 | 0.30 |
| de_terminal | 11/3 | +0.16 | 0.03 |
| obs_counterfactual | 9/5 | +0.16 | 0.19 |

**Reading:**
- The method transfers to real biological wiring. It is best on two of three decisions and in mean.
- It is again strongest on nested decisions (D2), consistent with the synthetic results.
- It is weakest on D3, where Gfi1 ⊣ EgrNab plus the cJun feed-forward loop is less switch-like.
- With 5 replicates and large spread, only the comparison against DE is significant. More replicates are needed.
- The oracle is low on D2 because few uncommitted cells with a known Ery/Mk fate exist in the snapshots.

## Speed-ups (29 Oct 2026)
- **Options** in `probe_interventional.score_interventional_probe`:
  - `candidates`: intervene only on the top-N genes by fate correlation, or on a given list such as TFs. All genes stay in the forward model.
  - `max_cells`: subsample the undecided cells.
  - `batch`: genes intervened on together.
  - `n_jobs`: parallel Lasso fits.

  Defaults reproduce the exact version: same scores to about 1e-16, checked on BF_rep0 and HSC_rep1.
- **Fast scorer:** `method_timing.score_timing_cf_fast`, with defaults of 300 candidate genes and 300 cells.
- **Sparse arithmetic is not the gain:** it is used only if the learned forward model is under 5% filled. Measured density is about 14% (≈58 inputs per gene at 400 genes), where dense arithmetic is faster.
- **Runtime, one decision, 500 cells, 2 cores:**

| genes | exact (before) | fast (100 genes tested, 150 cells) |
|---|---|---|
| 200 | 5.7 s | 1.9 s |
| 400 | 32 s | 6.2 s |
| 800 | 179 s | 25 s |
| 2,000 | (est. 40+ min) | 3.1 min |
| 4,000 | | 14.5 min |

  Runtime still grows about with genes squared: the forward-model fits plus interventions over all genes.
- **Accuracy:** even with deliberately strict shortcuts (30 genes tested, 100 cells), the AUPRC is unchanged on all 34 benchmark decisions. The mean is 0.682 vs 0.678 exact, within 0.05 in 33/34, never worse by more than 0.1 (`results/fast_vs_exact.csv`, notebook section 8).
- The previous dense implementation is kept as `archive/probe_interventional_v1_dense.py`.

## Real-data validation: Paul15 (first run, 29 Oct 2026)
`paul15_eval.py`: MEP → Ery vs Mk, and GMP → Mo vs Neu. About 1,500 genes (1,500 HVGs + literature genes). Ground truth = literature regulators; **the lists in the script still need expert review.** Output: `results/paul15_literature_eval.csv`, `logs/paul15.log`.

Run with:
`python paul15_eval.py ../archive/2026-02_group_exploration/trajectory_analysis_output/paul15_processed.h5ad`

| method | Ery/Mk core hits in top 20 (of 3) | Mo/Neu core hits in top 20 (of 4) | median rank of core regulators (of 1,501) |
|---|---|---|---|
| fate_corr (CellRank-style) | 1 (Klf1 #11) | 2 (Irf8 #9, Cebpe #15) | 78 / 22 |
| de_terminal | 1 (Klf1 #10) | 1 (Irf8 #9) | 151 / 53 |
| rf_shap_fixed | 1 | 1 | 131 / 112 |
| **OURS_timing_cf_fast** (300 candidates, 300 cells) | **0** | **0** | 187 / 193 |
| random | 0 | 0 | 550 / 716 |

Random expectation is about 0.05 hits in the top 20.

**Our method does not transfer to Paul15 yet.**
- **Ery/Mk top genes are mostly megakaryocyte markers** (F2r, Treml1, Lat). Meis1 (a known Mk regulator) is #5, Pbx1 #42, Tal1 #65, but Klf1 is only #187.
- **The candidate filter drops regulators.** Myb, Zfpm1, Klf4 and Egr1 were not among the 300 most fate-correlated genes, so they were never tested.
- **Few undecided cells.** Most Paul15 cells are already committed; the Mk branch has only 68 cells.

**Next diagnostics:**
1. Exact version without the candidate filter.
2. TFs as candidates and as the only allowed regulators.
3. Pseudotime sanity check.
4. A second real dataset (mouse gastrulation, Tal1 knockout).

### Paul15 diagnostics (29 Oct 2026)
`paul15_diagnostics.py` (variants A–C), `paul15_tf_baselines.py` (baselines ranked among TFs only).
TF list: pySCENIC `mm_mgi_tfs.txt` in `external/tfs/`. 146 of the 1,501 genes are TFs, including all 19 literature regulators.

| variant | Ery/Mk top 10 | Ery/Mk: literature regulators in top 20 (of 12) | Mo/Neu: notable ranks |
|---|---|---|---|
| A exact, all genes (no candidate filter) | Mk markers (Rab37, F2r, Lat, …) | 2 | Irf8 #101, Sfpi1 #60 |
| B TF candidates | Meis1, Mef2c, Pbx1, Tal1, …, Fli1 | 6 | Sfpi1 #8, Irf8 #13, Cebpa #23 |
| C TF candidates + TF regulators | Meis1, Mef2c, Lrrfip1, Pbx1, Elf1, Gata2, Gfi1b, Tal1, Fli1, … | 5 | **Irf8 #1**, Sfpi1 #9 |
| fate_corr, ranked among TFs | Mthfd1, Klf1, Gata2, …, Pbx1, Mef2c, Fli1 | 7 | Irf8 #2, Cebpe #3, Gfi1 #5 |
| de_terminal, ranked among TFs | Klf1, Mthfd1, Gata2, Gata1, … | 7 | Irf8, Cebpe, Gfi1 in the top 10 |

**Findings**
1. **The candidate filter was not the problem.** The exact version on all genes is as bad or worse (42 min for Ery/Mk).
2. **Restricting to TFs is what makes the output biologically sensible.** Ery/Mk then ranks the known Mk regulators on top (Meis1, Pbx1, Mef2c, Tal1, Fli1, Gfi1b), and Mo/Neu ranks Irf8 #1. Competing methods (SCENIC, CellOracle, CEFCON) also restrict to TFs.
3. **Ranked among the same TFs, fate_corr and DE are still as good or better**, especially on Mo/Neu (Irf8, Cebpe, Gfi1 all in their top 5).
4. **Our method is one-sided.** It finds the regulators of one fate (Mk side, Mo side) and misses the other (Klf1 for Ery; Cebpe and Gfi1 for Neu). Likely cause: the score is the raw increase in P(fate). If undecided cells already lean towards a fate, there is little headroom to increase it, so that fate's drivers are under-scored. Candidate fix: score the shift in log-odds, or rank per fate and interleave.

## Version 2: shorter fast-forward + per-fate ranking (29 Oct 2026)
Motivation: on Paul15 the fast-forward (K=15 steps) drives almost every undecided cell to one fate, so
the method only finds regulators of that fate ("one-sidedness").

Choice made on dev data only (BF + TF replicates 0-2; `horizon_check.py benchmark`, `score_modes.py`):
K=3 steps and the `perfate` rule (rank genes separately per fate by the probability shift towards that
fate, take each gene's best per-fate rank). Both K=3 and K=5 reach 1.00 on all 6 dev decisions; K=3 chosen,
K=5 reported as sensitivity check. Paul15 (`horizon_check.py paul15`) was used only as a diagnostic.

Held-out test (`eval_v2.py` -> `results/v2_heldout.csv`): BF/TF replicates 3-4, CBF (all), HSC (all);
28 decisions, driver-vs-marker AUPRC:

| method | mean | vs fate_corr (wins/losses, Wilcoxon p) |
|---|---|---|
| fate_corr (best baseline) | 0.55 | |
| OURS v1 (K=15, prob) | 0.66 | |
| **OURS v2 (K=3, perfate)** | **0.75** | 20/7, p=0.017 |
| OURS v2, K=5 (sensitivity) | 0.77 | 20/6, p=0.007 |
| ORACLE (true future fate) | 0.75 | |

Per model (v2, K=3): BF 0.83 (fate_corr 0.85), TF 1.00 (0.89), CBF D1/D2 0.75/1.00 (0.32/0.68),
HSC D1/D2/D3 0.74/0.60/0.47 (0.42/0.40/0.63). Weak spot: HSC-D3 (Gfi1/EgrNab, neutrophil vs monocyte).

Paul15 with v2 (TF-restricted, variant C): ErMk Klf1 #21, Fli1 #9, 5/12 known regulators in top 20 —
not better than K=1 or fate_corr. Remaining cause: the fate classifier is overconfident on progenitors
(MoNeu undecided GMP cells get P(Mo)=0.99), so there is no real decision region to intervene on.
Next: calibrated / graph-based fate probabilities; and an independent real dataset with true transition
cells (Paul15 is now a diagnostic set, not a test set).

## Fate-probability source ("readout") check (29 Oct 2026)
`readout_check.py`, option `readout=` in `score_interventional_probe`: logreg (original), pca_logreg (20 PCs),
soft_logreg (soft labels from graph absorption), knn_absorb (graph absorption of nearest cells).
- Dev (BF/TF reps 0-2): logreg 1.00, pca_logreg 1.00, soft_logreg / knn_absorb worse on BF (0.51-0.67). Dev cannot
  separate logreg and pca_logreg, so the default stays **logreg** (no choice made on held-out data).
- Held-out (28 decisions): logreg 0.75, pca_logreg 0.77 (4/3, p=0.24), soft_logreg 0.74, knn_absorb 0.69.
- Paul15: no option fixes it. `paul15_drift_check.py` shows why — mean P(fate) of undecided cells:

| | before fast-forward | 1 step | 3 steps |
|---|---|---|---|
| ErMk, logreg | Ery 0.74 | 0.89 | 0.98 |
| ErMk, pca_logreg | Ery 0.61 | 0.78 | 0.96 |
| MoNeu, logreg | Mo 0.98 | 1.00 | 1.00 |
| MoNeu, soft_logreg | Mo 0.47 | 0.63 | 0.79 |

  Two causes: (1) the classifier is over-confident on GMP cells (soft labels fix this); (2) the forward model itself
  drifts every cell towards the majority fate (Ery: 372 terminal cells vs 68 Mk), even from balanced starts.
  Next: a fate-balanced forward model (training pairs weighted so each fate branch counts equally).

## Independent real-data test: Nestorowa 2016 (2 Nov 2026)
`nestorowa_prepare.py` (raw GEO counts + MGI Ensembl->symbol) -> `data_real/nestorowa16.h5ad`;
`nestorowa_eval.py` -> `results/nestorowa_eval.csv`. Setup fixed before running; method frozen (v2); run once.
Decision: stem cells -> MegE-primed (turquoise MEP-sorted, 287) vs lympho-myeloid-primed (gold GMP/LMPP-sorted, 254).
Literature (TO REVIEW): core Gata1 Klf1 Tal1 | Spi1 Cebpa Ikzf1; extended Gata2 Zfpm1 Nfe2 Lmo2 Myb Gfi1b Fli1 | Irf8 Gfi1 Cebpb Satb1.
1,522 genes (1,500 HVGs + literature), 165 TFs.

| method | ranked among | core in top 20 (of 6) | all in top 20 (of 17) | median core rank |
|---|---|---|---|---|
| **OURS v2, TF-restricted** | TFs (165) | **6** | 12 | **4** |
| OURS v2 fast (all genes) | TFs | 5 | 11 | 5 |
| fate_corr | TFs | 5 | 12 | 6 |
| de_terminal | TFs | 5 | 10 | 6 |
| rf_shap_fixed | TFs | 5 | 13 | 8 |
| obs_counterfactual | TFs | 4 | 8 | 9 |
| random | TFs | 1 | 1 | 77 |
| OURS v2 fast | all genes (1,522) | 2 | 2 | 46 |
| fate_corr / de_terminal / rf_shap | all genes | 3 | 4-5 | 24-36 |

OURS v2 TF top 10: Klf1 Spi1 Cebpa Gata1 Gata2 Ddx4 Esr1 Hlf Irf8 Tal1 (both sides of the decision, unlike Paul15).
Among all genes our fast version is worse than the baselines (Ikzf1, Nfe2, Lmo2, Myb are not among its 300
pre-selected candidates). One decision, 6 core genes: no significance claim possible.

## Independent real-data test: LARRY lineage tracing (Weinreb 2020, in vitro)

Script `larry_eval.py` (resume finished methods with `RESUME=1`). Neutrophil vs monocyte decision; 8,000 lineage cells sampled (seed 0). Methods see only cell-type annotation and one root cell, no time points or clones.
Reference: for 1,053 day-2 cells with clonal fate, |Spearman| of expression with the Neu fraction among the clone's day-4/6 descendants. "Early" = top 50 genes (top 10 TFs); "late" = top terminal |LFC| genes not among the top 200 (30 TFs). Headline = AUPRC early vs late.

| method | AUPRC all genes (chance 0.20) | AUPRC TFs (chance 0.25) |
|---|---|---|
| OURS_v2_TF (TF candidates and regulators) | – | **0.71** |
| absorption_corr_decision_region | **0.66** | **0.71** |
| OURS_v2_fast | 0.48 | 0.62 |
| obs_counterfactual | 0.40 | 0.60 |
| fate_corr_decision_region | 0.43 | 0.57 |
| rf_shap_fixed | 0.37 | 0.55 |
| de_terminal | 0.27 | 0.52 |
| fate_corr | 0.30 | 0.51 |
| OURS_v1_fast | 0.39 | 0.51 |
| random | 0.23 | 0.34 |

Our method beats fate_corr, DE, SHAP and observational counterfactuals but ties the strongest baseline (absorption correlation in the decision region) among TFs and trails it among all genes. Single seed; the TF reference sets are small (10 vs 30), so differences of ~0.1 are within noise. The OURS_v2_TF run uses max_cells=300 (the uncapped run ran out of memory).

## Ablations of version 2 (2 Nov 2026)
`ablation_v2.py` → `results/ablation_v2.csv`. Same 28 held-out decisions as `eval_v2.py`. Mean driver-vs-marker AUPRC; paired Wilcoxon against v2.
New options in `score_interventional_probe`: `pseudotime` ("dpt" default, "true", "pca_dist", "dpt_root") and `region_q` (0.5 default; 0 = all non-terminal cells). The defaults reproduce v2 exactly. The pre-change file is `archive/probe_interventional_v2_pre_ablation.py`.

| variant | mean | BF | CBF | HSC | TF | vs v2 (better/worse, p) |
|---|---|---|---|---|---|---|
| **v2** (K=3, decision region, DPT) | **0.75** | 0.83 | 0.87 | 0.60 | 1.00 | |
| K=0: no rollout (direct effect only) | 0.32 | 0.18 | 0.34 | 0.32 | 0.32 | 3/24, p<0.001 |
| all non-terminal cells (no decision region) | 0.75 | 0.83 | 0.87 | 0.61 | 1.00 | 3/6, p=0.37 |
| DPT from a perturbed root (mean of 3 seeds: 0.65/0.67/0.61) | 0.64 | 0.79 | 0.79 | 0.46 | 1.00 | 6/16, p=0.010 |
| simulator time instead of DPT | 0.60 | 0.67 | 0.70 | 0.47 | 1.00 | 8/15, p=0.026 |
| distance from the root in PCA space | 0.56 | 0.46 | 0.59 | 0.49 | 1.00 | 5/17, p=0.003 |

- **The propagation is what makes it work.** Without the rollout, only the direct effect of the clamped gene on the classifier is left. That scores 0.32, below the Wachter observational-counterfactual baseline (0.52).
- **The decision region has no effect** on synthetic data. It can be dropped or kept for speed; report it.
- **Pseudotime matters.** A naive pseudotime costs 0.19, and moving the root to a neighbouring cell costs 0.10, mostly on HSC.
- **Simulator time is worse than DPT.** Likely reasons: clock time is not developmental progress (cells commit at different times), and the lag and onset threshold were tuned on the DPT scale. So it is not a fair upper bound.

## LARRY robustness: three cell subsamples, gene bootstrap, wider candidate filters (2 Nov 2026)
`SEED=s TAG=s<s> METHODS=... python larry_eval.py` for seeds 0–2 (8,000 cells each; seed 0 = the original run, reproduced exactly), then `larry_bootstrap.py` → `results/larry_bootstrap.csv`, `results/larry_bootstrap_paired.csv`. Per-gene scores: `results/larry_scores_s*.csv`. Version 1 was not rerun.
New variants (regulators = all genes): `OURS_v2_c1000` (1,000 candidates) and `OURS_v2_c300TF` (top 300 plus every TF).

Early-vs-late AUPRC, mean over 3 cell subsamples (SD):

| method | TFs (chance 0.25) | all genes (chance 0.20) |
|---|---|---|
| absorption_corr_decision_region | **0.80** (0.10; 0.71/0.90/0.80) | **0.60** (0.13; 0.66/0.69/0.46) |
| OURS_v2_c300TF | 0.72 (0.04) | 0.49 (0.01) |
| OURS_v2_TF | 0.70 (0.01) | – |
| OURS_v2_c1000 | 0.70 (0.04) | 0.50 (0.01) |
| OURS_v2_fast | 0.67 (0.06) | 0.49 (0.01) |
| rf_shap_fixed | 0.57 | 0.36 |
| fate_corr_decision_region | 0.57 | 0.40 |
| obs_counterfactual | 0.54 | 0.38 |
| fate_corr | 0.51 | 0.29 |
| de_terminal | 0.49 | 0.28 |
| random | 0.38 | 0.22 |

Paired gene bootstrap (resampling the early and late gene sets, 2,000×; averaged over the seeds):
- **All genes:** ours beats fate_corr (+0.20), DE (+0.21), SHAP (+0.13) and observational counterfactuals (+0.10–0.11). The 95% CIs exclude 0 for all four.
- **TFs:** the same ordering, but with only 10 early vs 30 late TFs the CIs are wide. Only fate_corr and DE are borderline significant.
- **The absorption baseline is better than ours on average** (TFs −0.09 to −0.13, all genes −0.10 to −0.11). It is unstable across cell subsamples, though: SD 0.10–0.13 vs 0.01–0.04 for ours. The seed-0 tie in the first run was the low end of its range.

Wider candidate filters barely change anything. LARRY all genes: 0.49 → 0.50; Nestorowa all genes: core@20 2/6 for every filter. *Spi1* moves from rank 912 to about 200–350, and *Ikzf1* and *Myb* (Nestorowa) from 911 to 13–111. So the all-gene gap comes from markers ranking high, not from regulators being filtered out.

## Removing the classifier's direct effect (3 Nov 2026) — no improvement, not adopted
The classifier reads the clamped gene itself, so every score contains a direct, observational part; K=0 alone scores 0.32. Two removals are now options in `score_interventional_probe`; the default is unchanged, and the pre-change file is `archive/probe_interventional_v2_pre_direct.py`:
- `score_mode="perfate_sub"` subtracts the K=0 shift;
- `"perfate_mask"` resets the clamped gene to its unclamped value before readout.

Scripts: `direct_check.py` → `results/direct_check.csv`; `OURS_v2{sub,mask}_*` in `larry_eval.py` (TAG=d0–d2, same cells as s0–s2) and in `nestorowa_eval.py` (`results/nestorowa_eval_direct.csv`).

| | v2 | minus direct effect (sub) | masked at readout (mask) |
|---|---|---|---|
| synthetic held-out, 28 decisions | 0.746 | 0.742 (7 better / 5 worse, p=0.69) | 0.749 (7/4, p=0.66) |
| — CBF / HSC | 0.87 / 0.60 | 0.80 / 0.65 | 0.80 / 0.66 |
| LARRY all genes, top 300 + TFs (mean of 3 subsamples) | 0.49 | 0.52 | 0.53 |
| LARRY TFs, TF-only variant | 0.70 | 0.66 | 0.66 |
| LARRY TFs, top 300 + TFs | 0.72 (SD 0.04) | 0.72 (SD 0.09) | 0.71 (SD 0.11) |
| Nestorowa all genes, core@20 | 2/6 | 2/6 | 2/6 |

The all-gene top 10 on LARRY is almost unchanged: Gstm1, Lpl, Mmp8, Mpo, Lyz2, Elane, Ctsc, Lipg… So marker leakage does not come from the direct effect. It comes from the *propagated* effect along marker → marker edges: early-onset markers are allowed to "regulate" later markers of the same program. Restricting regulators to TFs (the TF variant) is the fix that addresses this.

## Published competitors: CEFCON and CellOracle (3 Nov 2026)
Installed with `setup_competitors.sh` (own Python 3.10 environments); adapters in `competitors.py` and
`competitors/run_{cefcon,celloracle}.py`. Both get the same cells, genes, terminal labels, root, DPT, fate
probabilities and decision region as every other method (exported to `cache/competitors/<tag>/input.h5ad`).
- **CEFCON**: lineages = terminal cells of a fate + progenitors with P(fate) ≥ 0.3 (≤ 2,000 cells); mouse prior network
  shipped with CEFCON; 350 epochs; driver set with the free SCIP solver. Score = influence (+1000 if in the driver set),
  max rank percentile over lineages. Genes constant within a lineage are dropped.
- **CellOracle**: mouse scATAC-atlas base GRN, tutorial settings, in-silico knockout of every base-GRN regulator among the
  genes; readout = change in P(fate) of the same classifier design on the undecided cells. Genes CellOracle refuses to
  simulate (too few GRN links) score 0. On LARRY, input capped at 3,000 cells (stratified by fate; 8,000 cells exceed
  8 GB). Only 75 (Nestorowa), 99 (Paul15) and 75 (LARRY) regulators can be knocked out; other genes tie at the bottom.
- Both need a prior network on mouse symbols, so they are not run on the synthetic benchmark.

| Dataset (metric) | Ours, TF | Ours, c300TF | CellOracle | CEFCON | best baseline |
|---|---|---|---|---|---|
| Nestorowa, among TFs (core@20 / all@20 / median core rank) | **6/6 / 12/17 / 3.5** | — | 6/6 / 9/17 / 6 | 5/6 / 12/17 / 14 | fate corr. 5/6 / 12/17 / 6 |
| Paul15 ErMk, among TFs (core@20 of 3 / median) | 1 / 24 | 2 / 12 | 2 / 8 | **3 / 7** | fate corr. 2 / 10 |
| Paul15 MoNeu, among TFs (core@20 of 4 / median) | 1 / 69 | 1 / 36 | 1 / 78 | 2 / 23 | **fate corr. 3 / 4** |
| LARRY, among TFs (AUPRC, mean ± sd over 3 samples) | 0.70 ± 0.01 | 0.72 ± 0.04 | 0.63 ± 0.08 | 0.37 ± 0.04 | **absorption 0.80 ± 0.09** |
| LARRY, all genes (AUPRC) | — | 0.49 ± 0.01 | 0.31 ± 0.02 | 0.36 ± 0.01 | **absorption 0.60 ± 0.13** |

LARRY paired gene bootstrap (B = 2,000, per sample): ours beats CEFCON among TFs in all three samples (significant at
0.05 in 2/3 for TF-only, 2/3 for c300TF; s0 p ≈ 0.06) and both competitors among all genes (CellOracle significant in 2/3,
CEFCON in 2/3). Against CellOracle among TFs the difference is +0.10 to +0.18 in samples 1–2 and −0.03 in sample 0,
never significant. Files: `results/{nestorowa_eval_competitors,paul15_competitors,paul15_tf_competitors}.csv`,
`results/larry_eval_c{0,1,2}.csv`, `results/larry_bootstrap{,_paired}.csv`.
**Reading**: ours is the best or tied-best gene-level method on Nestorowa and LARRY and beats CEFCON on LARRY; on Paul15
the published methods (CEFCON on ErMk) and simple baselines (MoNeu) are better. CellOracle is close to ours among TFs
but covers far fewer genes. Both competitors use curated prior networks, which may favour literature references.

## CellRank 2 (4 Nov 2026)
The real CellRank package (2.0.6, Weiler et al. 2024; `competitors/run_cellrank.py`, own environment), replacing the
"CellRank-style" fate-correlation baseline as the published reference. No velocity: PseudotimeKernel on the shared DPT
(soft threshold), 15-NN graph on 20 PCs, GPCCA with terminal states = the annotated terminal cells, fate probabilities,
`compute_lineage_drivers` (correlation with each fate probability over all cells). Score = max over fates. No prior
network, so it also runs on the synthetic benchmark. ~10 s per decision. Run with `run_cellrank_all.sh`.

| | synthetic held-out (28) | Nestorowa TFs core@20 | Paul15 TFs core@20 ErMk / MoNeu | LARRY TFs | LARRY all genes |
|---|---|---|---|---|---|
| CellRank 2 | 0.56 (BF 0.53, TF 0.87, CBF 0.56, HSC 0.52) | 5/6 (all genes 3/6) | 2/3 / **4/4** (all genes 1/3 / 1/4) | 0.56 ± 0.05 | 0.33 ± 0.02 |
| Ours v2 | **0.75**; 20 wins / 7 losses, Wilcoxon p = 0.010 | **6/6** (TF-only) | 1/3 / 1/4 (TF-only) | **0.70–0.72** | **0.49** |

LARRY paired bootstrap: ours beats CellRank among all genes in every sample (p ≤ 0.003) and among TFs in every sample
(significant in 2/3 for TF-only and c300TF). CellRank is the best method on Paul15 MoNeu (4/4), where ours is weak.
Full table of every method on every dataset: `results/overview_all_methods.csv`.

## Ablation figure (4 Nov 2026)
`ablation_v2_extra.py` adds, on the same 28 held-out decisions: no timing rule, K = 1 and 10, onset threshold 0.1 and 0.3
(`results/ablation_v2_extra.csv`). `fig_ablation_v2.py` combines them with `ablation_v2.csv`, `v2_heldout.csv` (K = 5,
scoring rule) and `readout_heldout.csv` (fate predictor) into `figures/fig_ablation_v2_panels.{pdf,png}` and
`results/ablation_v2_summary.csv` (mean AUPRC, wins/losses and paired Wilcoxon p against the chosen setting).
Significant drops: no timing rule 0.75 → 0.42 (23/4, p < 0.001); K = 0 0.32 (24/3, p < 0.001); pseudotime from a
shifted root 0.64 (p = 0.010), PCA distance 0.56 (p = 0.003), simulator time 0.60 (p = 0.026). Not significant:
K = 1/5/10, onset threshold 0.1/0.3, all non-terminal cells, max-shift scoring, the alternative fate predictors.

## Notebook on version 2 (5 Nov 2026)
`notebooks/results_and_figures.ipynb` now uses version 2 and the 28 held-out decisions throughout, with real CellRank 2.
Figures (same style; one fixed colour + marker per method): `fig_synthetic_competitors` (competitors selectable via
`COMPETITORS`), `fig3_method_comparison` (all methods), `fig4_per_decision`, `fig5_runtime_scaling`, ablation panels.
New measurements for version 2:
- `stress_v2.py` → `results/fast_vs_exact_v2.csv`: 30 candidate genes / 100 cells vs exact on the 28 decisions:
  0.753 vs 0.746, within 0.05 in 27/28, never worse by > 0.1 (Wilcoxon p = 0.84).
- `scaling_v2.py` → `results/scaling_v2_{genes,cells}.csv` (2 cores, one BF decision): exact 0.4 s (100 genes) to
  308 s (1,600 genes), roughly cubic in genes (one Lasso per gene); real-data settings (300 genes, 300 cells)
  26 s at 800, 103 s at 1,600, 459 s at 3,200 genes; cells 0.4 / 1.1 / 2.9 s for 500 / 2,000 / 4,000 cells.

## Extra held-out replicates and no oracle in figures (5 Nov 2026)
`simulate.py --models BF TF --first-rep 5 --reps 8 --n-traj 500` added replicates 5-12 of the bifurcation and
trifurcation networks (never used for tuning). `run_extra_reps.sh` evaluates every method and ablation on them
(scripts take `ONLY_NEW=1`) and `merge_extra_reps.py` merges them into the main result files (originals in
`results/archive_pre_extra_reps/`). Held-out set: 44 decisions (BF 10, TF 10, CBF 10, HSC 14).
Ours 0.82 vs CellRank 2 0.61, fate correlation 0.67, DE 0.50, RF+SHAP 0.49 (all paired Wilcoxon p < 0.01).
Ablations on 44 decisions: no timing rule 0.37, K=0 0.29, K=1 0.75 (p=0.04), onset threshold 0.1 0.72 (p=0.02),
shifted root 0.75, PCA distance 0.68, true time 0.72 (all p<0.05), graph fate predictor 0.77 (p=0.04);
not significant: K=5/10, threshold 0.3, all non-terminal cells, max-shift scoring, PCA+LR, soft labels.
The true-fate correlation reference (ORACLE_early_true_fate) stays in the results files but is no longer shown.

## Cell weighting of the intervention effect (7 Nov 2026) — tried, not adopted
`probe_interventional.score_interventional_probe(cell_weight=...)`; default "uniform" is unchanged (checked: identical
scores). `cell_weight_check.py` → `results/cell_weight_check.csv`, 44 held-out decisions, mean AUPRC:
uniform 0.821, per-fate weight P_f(1-P_f) 0.822, per-fate region 0.822, pseudotime weight (middle of the decision
counts most) 0.824. 34–40 of 44 decisions unchanged; no significant difference (Wilcoxon p = 1.0 for pseudotime;
too few changed decisions to test the others). Trifurcation stays at 1.0 with every variant.

## Sponge experiments (8 Nov 2026)
The frozen method finds one strong, stable choanocyte driver (POU1F1, ΔP 0.12–0.19) but only weak, unstable
effects for sclerocytes, metabolocytes and pinacocytes (max ΔP 0.007–0.03; top-5 Jaccard 0.23–0.63 across
subsamples). Likely causes: undecided cells concentrated near the choanocyte branch, four fates treated as one
decision, small fate probabilities. `sponge_run.py` gained `--k`, `--cell-weight`, `--score-mode perfate_logodds`
(new in probe_interventional) and `--tree` (stem → choano vs branch; transition → the other fates); per-fate
tables now include log-odds shifts (dL_*), the measure used for ranking and the logFC reference.
`run_sponge_experiments.sh` runs all variants on the same 4 subsamples; `sponge_compare.py` summarises effect
size, stand-out TFs, top-5 stability and direction per fate. Defaults unchanged.

## Sponge experiments: results (10 Nov 2026)
All 6 runs × 4 subsamples (80% of 2,616 cells) finished; `logs/sponge_compare.txt`, `results/sponge_consensus.csv`.
Pre-set rule: a variant is better if the top-5 TFs of most weak fates (sclero, digest, skin) are more stable across
subsamples, without losing POU1F1 for choanocytes and without fewer top TFs going up in their fate.
- No variant passes. Top-5 Jaccard (base → fate_region / logodds / K5 / K10 / tree):
  sclero 0.51 → 0.72 / 0.43 / 0.72 / 0.67 / 0.07; digest 0.28 → 0.31 / 0.31 / 0.26 / 0.26 / 0.19;
  skin 0.42 → 0.26 / 0.25 / 0.16 / 0.12 / 0.17. Longer rollouts lower the share of top TFs that go up in their fate.
- Tree mode is technically limited: 30 choanocytes vs ~990 branch cells (P(choano) ≈ 0.01 in undecided cells) and
  P(sclero) ≈ 0.02 in the second decision → the classifier is dominated by class imbalance.
- POU1F1 (SUB4.g65) ranks 1st for choanocytes in 23/24 runs (2nd once).
- Consensus over all 24 runs (top-10 in ≥ 75% of runs): choano g65, g14569, g14567, g1395, g5080, g17564 (first four
  up ≥ 1.7 logFC); sclero g17016, g18083, g6624, g17667 (g17016, g17667 go down in sclerocytes: candidate
  knock-down/repressor drivers); digest g17172; skin none.
- `n_standout` (> 5 MAD) is not informative here (dozens of TFs pass); ignore it.
Decision: keep the frozen method for the sponge; report the consensus TFs as robust candidates.

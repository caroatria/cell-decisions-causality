# Archive

Superseded work, kept for reference. Nothing here is used by the current pipeline in `driver_benchmark/`.

## 2026-02_group_exploration/  (group exploration, Feb 2026)
- `BUNGEE.py`: RF + prediction entropy + SHAP at branch points.
  **Known bug:** with SHAP >= 0.45, `shap_values[i]` indexes cells, not classes, so the reported
  "top genes" are just the first K genes of the HVG list (identical for every fate). The gene lists in
  `results/branchpoint_genes.csv` and `results/ld/` are therefore artifacts and should not be interpreted.
  A corrected version is `rf_shap_fixed` in `driver_benchmark/baselines.py`.
- `michal/`: centroid-direction and terminal-classifier exploration (marimo notebooks). The
  terminal-classifier idea lives on as the fate classifier of the current method.
- `miki/`: silhouette "swing state" exploration.
- `manuel_first_data_analysis.ipynb`: silhouette variants, Granger causality between clusters, CellRank attempts.
- `benchmarks.ipynb`: Paul15 / Moignard15 preprocessing (outputs in `trajectory_analysis_output/`).
- `my_method.ipynb`: ortholog mapping of the BUNGEE gene lists (affected by the bug above).
- `results/`: BUNGEE outputs (two ~2.6 GB `adata_with_uncertainty.h5ad` files, figures, gene lists).
- `trajectory_analysis_output/`: processed Paul15 / Moignard15 files and figures.
  `paul15_processed.h5ad` is the file `driver_benchmark/paul15_eval.py` will use.

Archived on 29 Oct 2026. Paths inside these notebooks still point to the old locations
(e.g. `data/suberites_domuncula_cytotrace_SUB4.h5ad`), so they will not run unchanged.

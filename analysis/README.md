# Analysis

Reusable analysis code published with the repository. Nothing here is derived from patient data.

Run these scripts from the repository root so local imports resolve consistently.

## Tracked modules

- `metrics/`: Dice, HD95, and surface-Dice scoring (`metrics_rectal.py`, `lits_metric_utils.py`)
- `cka/`: CKA representation-analysis driver and its configs
- `attention/`: architecture-agnostic attention-flow utilities
- `preprocessing/`: folder and single-case inference helpers (`run_folder_inference.py`, `run_single_case_pipeline.py`)
- `tumor/`: tumor intensity analysis
- `stats/`: parameter counting
- `uncertainty/`: calibration, ECE, temperature scaling, and low-confidence case selection
- `features/`: deep-feature and embedding extraction
- `runtime/`: inference runtime, peak memory, parameter, and FLOP benchmarking

## Local-only content

These stay out of Git:

- `analysis/figures/`: figure scripts for the accompanying manuscripts
- `analysis/radiomics/`: CERR-based radiomics extraction; depends on a local `project_paths` module and on generated feature tables
- `analysis/results/`, `analysis/outputs/`: generated figures, tables, embeddings, and reports
- `analysis/csvs/`: tabular outputs that may contain pseudonymized case IDs
- `analysis/uncertainty/*.csv`: generated calibration and uncertainty summaries
- large binary or tabular outputs matched by the repository-wide ignore rules

# Reproducibility Protocol

## Environment

- Python 3.12
- PyTorch 2.14.1 + CUDA 13.0
- NVIDIA RTX 4060 Laptop GPU, 8 GB
- CPU-only execution is supported but not benchmarked

Create or reuse the project virtual environment:

```powershell
code\.venv\Scripts\python.exe -m pip install -r code\requirements.txt
```

## Data

All headline data come from the official PDEBench datasets:

- `datasets/pdebench_burgers_official`
- `datasets/pdebench_advection_beta1_official`
- `datasets/pdebench_advection_beta{01,02,04,07,20,40,70}_official`
- `datasets/pdebench_diff_sorp_official`
- `datasets/pdebench_diff_react_2d_t10`
- `datasets/pdebench_ns_incom_2d`

The external Burgers `Nu=0.1` test set is fetched without downloading the
full 8.2 GB HDF5:

```powershell
code\.venv\Scripts\python.exe datasets\fetch_burgers_external.py `
  --url https://hf-mirror.com/datasets/pdebench/Burgers/resolve/main/1D_Burgers_Sols_Nu0.1.hdf5 `
  --output datasets\external\burgers_nu0.1\test.npz `
  --samples 1000
```

## Training and Evaluation

Primary 1D dual-metric runs:

```powershell
code\.venv\Scripts\python.exe code\run_pg_derivative_refiner_queue.py `
  --seeds 7 11 23 --steps 4000

code\.venv\Scripts\python.exe code\run_pg_advection_output_refiner_queue.py `
  --seeds 7 11 23 --steps 4000
```

Residual and statistical audits:

```powershell
code\.venv\Scripts\python.exe code\analyze_residual_credibility.py
code\.venv\Scripts\python.exe code\analyze_paired_statistics.py
code\.venv\Scripts\python.exe code\analyze_2d_physics.py
```

External generalization:

```powershell
code\.venv\Scripts\python.exe code\evaluate_burgers_external_viscosity.py
code\.venv\Scripts\python.exe code\evaluate_advection_refiner_generalization.py
```

## Selection Rules

- Checkpoints are selected on `dev`, never on `test`.
- Burgers derivative refiner: at most 2% dev L2 increase; lowest dev residual.
- Advection output refiner: at most 30% dev L2 increase; lowest dev residual.
- External evaluations use frozen weights and perform no test-time tuning.
- The historical 3-seed Burgers and Advection results remain exploratory because
  the final hyperparameter boundary was informed by test behavior. The external
  viscosity and cross-beta checks are the primary unseen-data evidence.

## Statistical Protocol

- Common 1000-sample test split for paired comparisons.
- 20,000 paired bootstrap resamples.
- Paired t-test and Wilcoxon signed-rank test.
- Holm correction over four primary claims:
  Burgers L2, Burgers residual, Advection L2, Advection residual.

## Expected Artifacts

- `analysis_plots_clean/pino_comparison.json`
- `analysis_plots_clean/residual_credibility.json`
- `analysis_plots_clean/paired_statistics.json`
- `analysis_plots_clean/burgers_external_nu0.1.json`
- `analysis_plots_clean/advection_refiner_generalization.json`
- `analysis_plots_clean/physics_2d_audit.json`
- `analysis_plots_clean/diffusion_sorption_audit.json`
- `docs/EXPERIMENT_AUDIT_20261127.md`

Checkpoints and datasets are intentionally excluded from Git. Preserve them
locally or publish them separately with SHA256 checksums.

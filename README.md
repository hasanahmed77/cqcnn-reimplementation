# CQ-CNN Reimplementation

Re-implementing the CQ-CNN research paper for learning and research purposes.

Original paper implementation:  
[alz-cq-cnn repository](https://github.com/mominul-ssv/alz-cq-cnn)

## Project Structure

```text
cqcnn-reimplementation-clean/
|-- experiments/
|   |-- medmnist/       # BreastMNIST and PneumoniaMNIST training scripts
|   `-- oasis2/         # OASIS-2 coronal classical vs 2-qubit hybrid scripts
|-- notebooks/
|   `-- classification/ # Analysis notebooks and experiment dashboards
|-- results/
|   `-- classification/ # Generated CSV metrics
|-- tests/
|   `-- smoke/          # Small sanity checks for model/QNN behavior
|-- README.md
`-- .venv/              # Local Python environment, ignored by git
```

## Run OASIS-2 Coronal Training

```bash
source .venv/bin/activate
python experiments/oasis2/oasis2_coronal_experiment.py
```

This writes one CSV per model:

```text
results/classification/oasis2_coronal_128_64/oasis2_coronal_2qubit_classical.csv
results/classification/oasis2_coronal_128_64/oasis2_coronal_2qubit_hybrid.csv
```

By default, the OASIS-2 coronal run trains both models for 10 trials using
seeds 42 through 51. Final rows include a convergence status, and summary rows
include failure counts/rates for stuck trials.

When the hybrid model runs, it also writes gradient diagnostics here:

```text
results/classification/oasis2_coronal_128_64/oasis2_coronal_2qubit_hybrid_diagnostics.csv
```

To test whether the hybrid model is sensitive to optimizer step size, run the
hybrid-only learning-rate sweep:

```bash
source .venv/bin/activate
python experiments/oasis2/oasis2_coronal_lr_sweep.py
```

Sweep outputs are written under:

```text
results/classification/oasis2_coronal_128_64/lr_sweep/
```

The sweep summary is:

```text
results/classification/oasis2_coronal_128_64/lr_sweep/hybrid_lr_sweep_summary.csv
```

## Analyze Results In Notebook

```bash
source .venv/bin/activate
jupyter notebook notebooks/classification/oasis2_coronal_2qubit_analysis.ipynb
```

Use the notebook to rerun the experiment if needed, load the CSVs, plot learning
curves, compare final metrics, and record interpretation notes.

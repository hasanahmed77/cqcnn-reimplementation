Re-implementing CQ-CNN research paper for learning purpose.

## Project Structure

```text
cqcnn-reimplementation-clean/
|-- experiments/
|   |-- medmnist/       # BreastMNIST and PneumoniaMNIST classical/hybrid runs
|   `-- oasis2/         # OASIS-2 coronal classical vs 2-qubit hybrid runs
|-- results/
|   `-- classification/ # Generated experiment CSV outputs
|-- tests/
|   `-- smoke/          # Small model/QNN sanity-check scripts
|-- README.md
`-- .venv/              # Local Python environment
```

## Running OASIS-2 Coronal 2-Qubit Comparison

```bash
source .venv/bin/activate
python experiments/oasis2/oasis2_coronal_experiment.py
```

By default this runs both the classical CNN and 2-qubit hybrid CQ-CNN with:

- 128 stratified training samples
- 64 stratified test samples
- 10 epochs
- 3 trials
- batch size 8

It writes one CSV per model:

```text
results/classification/oasis2_coronal_128_64/oasis2_coronal_2qubit_classical.csv
results/classification/oasis2_coronal_128_64/oasis2_coronal_2qubit_hybrid.csv
```

## Preliminary Cross-Dataset Results

All experiments used:

- 2-qubit hybrid CQ-CNN
- matched classical CNN baseline
- 128 training samples
- 64 test samples
- 10 epochs
- 3 trials
- mean ± standard deviation reported across trials

| Dataset        | Classical Test Acc | Hybrid Test Acc | Classical Test Loss | Hybrid Test Loss | Better Model |
| -------------- | -----------------: | --------------: | ------------------: | ---------------: | ------------ |
| BreastMNIST    |    0.7031 ± 0.0221 | 0.6875 ± 0.0000 |     0.5805 ± 0.0114 |  0.6941 ± 0.0910 | Classical    |
| PneumoniaMNIST |    0.5938 ± 0.0000 | 0.6094 ± 0.0221 |     0.8140 ± 0.1175 |  0.7814 ± 0.1343 | Hybrid       |

Re-implementing CQ-CNN research paper for learning purpose.

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

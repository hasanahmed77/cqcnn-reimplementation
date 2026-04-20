from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

import torch

n_qubits = 2

feature_map = ZZFeatureMap(n_qubits)
ansatz = RealAmplitudes(n_qubits, reps=1)

qc = QuantumCircuit(n_qubits)
qc.compose(feature_map, inplace=True)
qc.compose(ansatz, inplace=True)

qnn = EstimatorQNN(
    circuit=qc,
    input_params=feature_map.parameters,
    weight_params=ansatz.parameters,
    input_gradients=True,
)

model = TorchConnector(qnn)

x = torch.tensor([[0.5, 0.8]], dtype=torch.float32)
output = model(x)

print("input:", x)
print("output:", output)
print("shape:", output.shape)
print(qc.draw())

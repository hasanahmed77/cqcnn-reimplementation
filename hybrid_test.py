import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


def create_qnn(n_qubits):
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
    return TorchConnector(qnn)


class HybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4 * 29 * 29, 4)
        self.fc2 = nn.Linear(4, 2)  # 2-qubit bridge
        self.qnn = create_qnn(2)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        print("input       :", x.shape)

        x = F.relu(self.conv1(x))
        print("after conv1 :", x.shape)

        x = F.max_pool2d(x, 2)
        print("after pool1 :", x.shape)

        x = F.relu(self.conv2(x))
        print("after conv2 :", x.shape)

        x = F.max_pool2d(x, 2)
        print("after pool2 :", x.shape)

        x = self.dropout(x)
        print("after drop  :", x.shape)

        x = x.view(x.shape[0], -1)
        print("after flat  :", x.shape)

        x = F.relu(self.fc1(x))
        print("after fc1   :", x.shape)

        x = self.fc2(x)
        print("after fc2   :", x.shape)

        x = self.qnn(x)
        print("after qnn   :", x.shape)

        x = self.fc3(x)
        print("after fc3   :", x.shape)

        x = cat((x, 1 - x), -1)
        print("after cat   :", x.shape)

        return x


if __name__ == "__main__":
    model = HybridNet()
    x = torch.randn(1, 1, 128, 128)
    y = model(x)
    print("final output:", y)
    print("final shape :", y.shape)

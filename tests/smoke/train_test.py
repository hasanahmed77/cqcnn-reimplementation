import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import cat
import torch.optim as optim

from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


# ===== QNN =====
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


# ===== Hybrid Model =====
class HybridNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4 * 29 * 29, 4)
        self.fc2 = nn.Linear(4, 2)
        self.qnn = create_qnn(2)
        self.fc3 = nn.Linear(1, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.qnn(x)
        x = self.fc3(x)
        x = cat((x, 1 - x), -1)
        return x


# ===== Dummy Data =====
batch_size = 4

# fake images
data = torch.randn(batch_size, 1, 128, 128)

# fake labels (0 or 1)
targets = torch.randint(0, 2, (batch_size,))


# ===== Setup =====
model = HybridNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# ===== Training Loop (3 steps only) =====
for step in range(3):
    optimizer.zero_grad()

    outputs = model(data)
    loss = criterion(outputs, targets)

    loss.backward()
    optimizer.step()

    print(f"Step {step}: Loss = {loss.item():.4f}")

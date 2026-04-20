import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from torchvision import transforms

import medmnist
from medmnist import INFO


# =========================
# Config
# =========================
TRIALS = 3
SEEDS = [42, 43, 44]
DATA_FLAG = "breastmnist"
BATCH_SIZE = 8
LR = 0.001
EPOCHS = 10
TRAIN_LIMIT = 128
TEST_LIMIT = 64
DEVICE = torch.device("cpu")


# =========================
# Classical Baseline Model
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ClassicalNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(4 * 29 * 29, 4)
        self.fc2 = nn.Linear(4, 2)  # direct 2-class output

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =========================
# Data
# =========================
info = INFO[DATA_FLAG]
DataClass = getattr(medmnist, info["python_class"])

transform = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ]
)

train_dataset = DataClass(split="train", transform=transform, download=True)
test_dataset = DataClass(split="test", transform=transform, download=True)

train_dataset = Subset(train_dataset, range(min(TRAIN_LIMIT, len(train_dataset))))
test_dataset = Subset(test_dataset, range(min(TEST_LIMIT, len(test_dataset))))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# Train / Eval helpers
# =========================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.squeeze().long().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.squeeze().long().to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# =========================
# Main
# =========================
def main():
    trial_results = []

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples : {len(test_dataset)}")

    for trial_idx, seed in enumerate(SEEDS[:TRIALS], start=1):
        print(f"\n=== Trial {trial_idx} | Seed {seed} ===")
        set_seed(seed)

        model = ClassicalNet().to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()

        last_train_loss, last_train_acc = None, None
        last_test_loss, last_test_acc = None, None

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion
            )
            test_loss, test_acc = evaluate(model, test_loader, criterion)

            last_train_loss, last_train_acc = train_loss, train_acc
            last_test_loss, last_test_acc = test_loss, test_acc

            print(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}"
            )

        trial_results.append(
            {
                "train_loss": last_train_loss,
                "train_acc": last_train_acc,
                "test_loss": last_test_loss,
                "test_acc": last_test_acc,
            }
        )

    train_losses = np.array([r["train_loss"] for r in trial_results])
    train_accs = np.array([r["train_acc"] for r in trial_results])
    test_losses = np.array([r["test_loss"] for r in trial_results])
    test_accs = np.array([r["test_acc"] for r in trial_results])

    print("\n=== Final Summary Across Trials ===")
    print(f"Train Loss: {train_losses.mean():.4f} ± {train_losses.std():.4f}")
    print(f"Train Acc : {train_accs.mean():.4f} ± {train_accs.std():.4f}")
    print(f"Test Loss : {test_losses.mean():.4f} ± {test_losses.std():.4f}")
    print(f"Test Acc  : {test_accs.mean():.4f} ± {test_accs.std():.4f}")


if __name__ == "__main__":
    main()

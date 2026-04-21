import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import cat
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


DATA_ROOT = Path(
    "/Users/mustakimahmedhasan/Workspace/Research/Datasets/"
    "cq-cnn-dataset-oasis-2/preprocessed_balanced"
)
RESULTS_DIR = Path("results/classification/oasis2_coronal_128_64")
LABEL_MAPPING = {"nondemented": 0, "moderate_dementia": 1}
CLASS_NAMES = ["nondemented", "moderate_dementia"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
FAILURE_MACRO_F1_THRESHOLD = 0.40
FAILURE_BALANCED_ACC_THRESHOLD = 0.55
DIAGNOSTIC_MODULES = ("fc2", "qnn", "fc3")


@dataclass
class ExperimentConfig:
    model_name: str
    data_root: Path
    output_dir: Path
    image_size: int
    batch_size: int
    lr: float
    epochs: int
    trials: int
    seeds: list[int]
    train_limit: int | None
    test_limit: int | None
    device: torch.device
    n_qubits: int = 2
    result_stem: str | None = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Oasis2CoronalDataset(Dataset):
    def __init__(self, split, data_root, image_size):
        self.split = split
        self.base_path = Path(data_root) / split / "coronal"
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        for class_name, label in LABEL_MAPPING.items():
            class_path = self.base_path / class_name
            if not class_path.is_dir():
                raise FileNotFoundError(f"Missing class directory: {class_path}")

            image_paths = sorted(
                path
                for path in class_path.iterdir()
                if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            samples.extend((path, label) for path in image_paths)

        if not samples:
            raise RuntimeError(f"No images found under {self.base_path}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("L")
        return self.transform(image), torch.tensor(label, dtype=torch.long)


class ClassicalNet(nn.Module):
    def __init__(self, image_size=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(flattened_size(image_size), 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class HybridNet(nn.Module):
    def __init__(self, image_size=128, n_qubits=2):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 5)
        self.conv2 = nn.Conv2d(2, 4, 5)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(flattened_size(image_size), 4)
        self.fc2 = nn.Linear(4, n_qubits)
        self.qnn = create_qnn(n_qubits)
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
        return cat((x, 1 - x), dim=-1)


def flattened_size(image_size):
    size_after_conv1 = (image_size - 5 + 1) // 2
    size_after_conv2 = (size_after_conv1 - 5 + 1) // 2
    return 4 * size_after_conv2 * size_after_conv2


def create_qnn(n_qubits):
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
    from qiskit_machine_learning.connectors import TorchConnector
    from qiskit_machine_learning.neural_networks import EstimatorQNN

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


def make_model(model_name, image_size, n_qubits):
    if model_name == "classical":
        return ClassicalNet(image_size=image_size)
    if model_name == "hybrid":
        return HybridNet(image_size=image_size, n_qubits=n_qubits)
    raise ValueError(f"Unknown model: {model_name}")


def limit_dataset(dataset, limit, seed):
    if limit is None:
        return dataset

    labels = labels_for_dataset(dataset)
    by_class = {0: [], 1: []}
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)

    rng = random.Random(seed)
    per_class = limit // len(by_class)
    remainder = limit % len(by_class)
    selected = []

    for class_idx, label in enumerate(sorted(by_class)):
        indices = by_class[label][:]
        rng.shuffle(indices)
        take = per_class + (1 if class_idx < remainder else 0)
        selected.extend(indices[: min(take, len(indices))])

    selected.sort()
    return Subset(dataset, selected)


def labels_for_dataset(dataset):
    if isinstance(dataset, Subset):
        base_labels = labels_for_dataset(dataset.dataset)
        return [base_labels[i] for i in dataset.indices]
    return [label for _, label in dataset.samples]


def make_balanced_subset(dataset, seed):
    labels = labels_for_dataset(dataset)
    by_class = {0: [], 1: []}
    for idx, label in enumerate(labels):
        by_class[int(label)].append(idx)

    min_count = min(len(indices) for indices in by_class.values())
    rng = random.Random(seed)
    selected = []
    for label in sorted(by_class):
        indices = by_class[label][:]
        rng.shuffle(indices)
        selected.extend(indices[:min_count])
    selected.sort()
    return Subset(dataset, selected)


def make_loaders(config, seed):
    train_dataset = Oasis2CoronalDataset("train", config.data_root, config.image_size)
    test_dataset = Oasis2CoronalDataset("test", config.data_root, config.image_size)

    train_dataset = limit_dataset(train_dataset, config.train_limit, seed)
    test_dataset = limit_dataset(test_dataset, config.test_limit, seed)
    balanced_test_dataset = make_balanced_subset(test_dataset, seed)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    balanced_test_loader = DataLoader(
        balanced_test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )
    return train_loader, test_loader, balanced_test_loader


def clone_module_params(module):
    return [param.detach().clone() for param in module.parameters()]


def module_grad_norm(module):
    squared_norm = 0.0
    for param in module.parameters():
        if param.grad is None:
            continue
        squared_norm += param.grad.detach().norm(2).item() ** 2
    return squared_norm**0.5


def module_update_norm(module, params_before):
    squared_norm = 0.0
    for before, after in zip(params_before, module.parameters()):
        squared_norm += (after.detach() - before).norm(2).item() ** 2
    return squared_norm**0.5


def capture_params_before_step(model):
    return {
        module_name: clone_module_params(getattr(model, module_name))
        for module_name in DIAGNOSTIC_MODULES
    }


def capture_grad_norms(model):
    return {
        f"{module_name}_grad_norm": module_grad_norm(getattr(model, module_name))
        for module_name in DIAGNOSTIC_MODULES
    }


def capture_update_norms(model, params_before):
    return {
        f"{module_name}_weight_update_norm": module_update_norm(
            getattr(model, module_name), params_before[module_name]
        )
        for module_name in DIAGNOSTIC_MODULES
    }


def mean_diagnostics(diagnostic_rows):
    if not diagnostic_rows:
        return {}

    keys = diagnostic_rows[0].keys()
    return {
        key: float(np.mean([row[key] for row in diagnostic_rows]))
        for key in keys
    }


def train_one_epoch(model, loader, optimizer, criterion, device, collect_diagnostics=False):
    model.train()
    total_loss = 0.0
    y_true = []
    y_pred = []
    diagnostic_rows = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        params_before = (
            capture_params_before_step(model) if collect_diagnostics else None
        )
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        if collect_diagnostics:
            batch_diagnostics = capture_grad_norms(model)
        optimizer.step()
        if collect_diagnostics:
            batch_diagnostics.update(capture_update_norms(model, params_before))
            diagnostic_rows.append(batch_diagnostics)

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(outputs.argmax(dim=1).detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / len(loader.dataset),
        "acc": accuracy_score(y_true, y_pred),
        "diagnostics": mean_diagnostics(diagnostic_rows),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())
        y_prob.extend(probs[:, 1].detach().cpu().numpy().tolist())

    metrics = classification_metrics(y_true, y_pred, y_prob)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def classification_metrics(y_true, y_pred, y_prob):
    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "auc": auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def convergence_status(metrics):
    if (
        metrics["macro_f1"] <= FAILURE_MACRO_F1_THRESHOLD
        or metrics["balanced_acc"] <= FAILURE_BALANCED_ACC_THRESHOLD
    ):
        return "stuck"
    return "converged"


def append_row(path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    should_write_header = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config):
    result_stem = (
        config.result_stem
        if config.result_stem is not None
        else f"oasis2_coronal_2qubit_{config.model_name}"
    )
    result_csv = config.output_dir / f"{result_stem}.csv"
    diagnostics_stem = (
        f"{result_stem}_diagnostics"
        if config.result_stem is not None
        else "oasis2_coronal_2qubit_hybrid_diagnostics"
    )
    diagnostics_csv = config.output_dir / f"{diagnostics_stem}.csv"
    result_csv.unlink(missing_ok=True)
    if config.model_name == "hybrid":
        diagnostics_csv.unlink(missing_ok=True)

    fields = [
        "row_type",
        "model",
        "dataset",
        "plane",
        "n_qubits",
        "eval_split",
        "trial",
        "seed",
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "test_macro_f1",
        "test_balanced_acc",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "balanced_acc",
        "sensitivity",
        "specificity",
        "auc",
        "tn",
        "fp",
        "fn",
        "tp",
        "lr",
        "convergence_status",
        "failure_count",
        "success_count",
        "failure_rate",
        "failure_macro_f1_threshold",
        "failure_balanced_acc_threshold",
    ]
    diagnostic_fields = [
        "model",
        "dataset",
        "plane",
        "n_qubits",
        "trial",
        "seed",
        "epoch",
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "test_macro_f1",
        "test_balanced_acc",
        "qnn_grad_norm",
        "qnn_weight_update_norm",
        "fc2_grad_norm",
        "fc2_weight_update_norm",
        "fc3_grad_norm",
        "fc3_weight_update_norm",
        "convergence_status",
        "failure_macro_f1_threshold",
        "failure_balanced_acc_threshold",
    ]

    rows_for_comparison = []

    for trial_idx, seed in enumerate(config.seeds[: config.trials], start=1):
        set_seed(seed)
        train_loader, test_loader, balanced_test_loader = make_loaders(config, seed)

        model = make_model(
            config.model_name,
            image_size=config.image_size,
            n_qubits=config.n_qubits,
        ).to(config.device)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
        criterion = nn.CrossEntropyLoss()

        print(
            f"\n=== {config.model_name} | trial {trial_idx}/{config.trials} "
            f"| seed {seed} ==="
        )
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Test samples : {len(test_loader.dataset)}")
        print(f"Balanced test: {len(balanced_test_loader.dataset)}")

        train_metrics = None
        test_metrics = None

        for epoch in range(1, config.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                config.device,
                collect_diagnostics=config.model_name == "hybrid",
            )
            test_metrics = evaluate(model, test_loader, criterion, config.device)
            lr = optimizer.param_groups[0]["lr"]
            epoch_convergence_status = convergence_status(test_metrics)

            append_row(
                result_csv,
                fields,
                {
                    "row_type": "epoch",
                    "model": config.model_name,
                    "dataset": "oasis-2",
                    "plane": "coronal",
                    "n_qubits": config.n_qubits,
                    "eval_split": "original",
                    "trial": trial_idx,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_acc": train_metrics["acc"],
                    "test_loss": test_metrics["loss"],
                    "test_acc": test_metrics["acc"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_balanced_acc": test_metrics["balanced_acc"],
                    "precision": "",
                    "recall": "",
                    "f1": "",
                    "macro_f1": "",
                    "balanced_acc": "",
                    "sensitivity": "",
                    "specificity": "",
                    "auc": "",
                    "tn": "",
                    "fp": "",
                    "fn": "",
                    "tp": "",
                    "lr": lr,
                    "convergence_status": "",
                    "failure_count": "",
                    "success_count": "",
                    "failure_rate": "",
                    "failure_macro_f1_threshold": FAILURE_MACRO_F1_THRESHOLD,
                    "failure_balanced_acc_threshold": FAILURE_BALANCED_ACC_THRESHOLD,
                },
            )
            if config.model_name == "hybrid":
                diagnostics = train_metrics["diagnostics"]
                append_row(
                    diagnostics_csv,
                    diagnostic_fields,
                    {
                        "model": config.model_name,
                        "dataset": "oasis-2",
                        "plane": "coronal",
                        "n_qubits": config.n_qubits,
                        "trial": trial_idx,
                        "seed": seed,
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "train_acc": train_metrics["acc"],
                        "test_loss": test_metrics["loss"],
                        "test_acc": test_metrics["acc"],
                        "test_macro_f1": test_metrics["macro_f1"],
                        "test_balanced_acc": test_metrics["balanced_acc"],
                        "qnn_grad_norm": diagnostics["qnn_grad_norm"],
                        "qnn_weight_update_norm": diagnostics[
                            "qnn_weight_update_norm"
                        ],
                        "fc2_grad_norm": diagnostics["fc2_grad_norm"],
                        "fc2_weight_update_norm": diagnostics[
                            "fc2_weight_update_norm"
                        ],
                        "fc3_grad_norm": diagnostics["fc3_grad_norm"],
                        "fc3_weight_update_norm": diagnostics[
                            "fc3_weight_update_norm"
                        ],
                        "convergence_status": epoch_convergence_status,
                        "failure_macro_f1_threshold": FAILURE_MACRO_F1_THRESHOLD,
                        "failure_balanced_acc_threshold": FAILURE_BALANCED_ACC_THRESHOLD,
                    },
                )

            print(
                f"Epoch {epoch}/{config.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['acc']:.4f} | "
                f"Test Loss: {test_metrics['loss']:.4f} | "
                f"Test Acc: {test_metrics['acc']:.4f} | "
                f"Macro F1: {test_metrics['macro_f1']:.4f} | "
                f"Balanced Acc: {test_metrics['balanced_acc']:.4f}"
            )

        original_metrics = test_metrics
        balanced_metrics = evaluate(model, balanced_test_loader, criterion, config.device)

        for eval_split, metrics in [
            ("original", original_metrics),
            ("balanced", balanced_metrics),
        ]:
            row = {
                "row_type": "final",
                "model": config.model_name,
                "dataset": "oasis-2",
                "plane": "coronal",
                "n_qubits": config.n_qubits,
                "eval_split": eval_split,
                "trial": trial_idx,
                "seed": seed,
                "epoch": config.epochs,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "test_loss": metrics["loss"],
                "test_acc": metrics["acc"],
                "test_macro_f1": metrics["macro_f1"],
                "test_balanced_acc": metrics["balanced_acc"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "macro_f1": metrics["macro_f1"],
                "balanced_acc": metrics["balanced_acc"],
                "sensitivity": metrics["sensitivity"],
                "specificity": metrics["specificity"],
                "auc": metrics["auc"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "lr": optimizer.param_groups[0]["lr"],
                "convergence_status": convergence_status(metrics),
                "failure_count": "",
                "success_count": "",
                "failure_rate": "",
                "failure_macro_f1_threshold": FAILURE_MACRO_F1_THRESHOLD,
                "failure_balanced_acc_threshold": FAILURE_BALANCED_ACC_THRESHOLD,
            }
            append_row(result_csv, fields, row)
            rows_for_comparison.append(row)

    summary_rows = summarize_rows(rows_for_comparison)
    for row in summary_rows:
        append_row(result_csv, fields, row)

    return rows_for_comparison, summary_rows


def summarize_rows(rows):
    metric_names = [
        "train_loss",
        "train_acc",
        "test_loss",
        "test_acc",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "balanced_acc",
        "sensitivity",
        "specificity",
        "auc",
    ]
    count_names = ["tn", "fp", "fn", "tp"]
    summary_rows = []

    grouped = {}
    for row in rows:
        key = (row["model"], row["eval_split"])
        grouped.setdefault(key, []).append(row)

    for (model_name, eval_split), group in grouped.items():
        for stat_name, reducer in [("mean", np.nanmean), ("std", np.nanstd)]:
            summary = {
                "row_type": "summary",
                "model": f"{model_name}_{stat_name}",
                "dataset": "oasis-2",
                "plane": "coronal",
                "n_qubits": 2,
                "eval_split": eval_split,
                "trial": "",
                "seed": "",
                "epoch": "",
                "test_macro_f1": "",
                "test_balanced_acc": "",
                "lr": "",
                "convergence_status": "summary",
                "failure_macro_f1_threshold": FAILURE_MACRO_F1_THRESHOLD,
                "failure_balanced_acc_threshold": FAILURE_BALANCED_ACC_THRESHOLD,
            }
            for metric in metric_names:
                summary[metric] = reducer([float(row[metric]) for row in group])
            for metric in count_names:
                summary[metric] = reducer([float(row[metric]) for row in group])
            if stat_name == "mean":
                failure_count = sum(
                    1 for row in group if row["convergence_status"] == "stuck"
                )
                summary["failure_count"] = failure_count
                summary["success_count"] = len(group) - failure_count
                summary["failure_rate"] = failure_count / len(group)
            else:
                summary["failure_count"] = ""
                summary["success_count"] = ""
                summary["failure_rate"] = ""
            summary_rows.append(summary)

    return summary_rows


def compare_models(output_dir):
    result_files = [
        output_dir / "oasis2_coronal_2qubit_classical.csv",
        output_dir / "oasis2_coronal_2qubit_hybrid.csv",
    ]
    rows = []
    for path in result_files:
        if not path.exists():
            continue
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)

    if not rows:
        return None

    original_rows = [
        row
        for row in rows
        if row["row_type"] == "final" and row["eval_split"] == "original"
    ]
    grouped = {}
    for row in original_rows:
        grouped.setdefault(row["model"], []).append(row)

    score_rows = []
    for model_name, group in grouped.items():
        score_rows.append(
            {
                "model": model_name,
                "macro_f1": np.nanmean([float(row["macro_f1"]) for row in group]),
                "balanced_acc": np.nanmean(
                    [float(row["balanced_acc"]) for row in group]
                ),
                "auc": np.nanmean([float(row["auc"]) for row in group]),
                "test_loss": np.nanmean([float(row["test_loss"]) for row in group]),
            }
        )

    if not score_rows:
        return None

    winner = sorted(
        score_rows,
        key=lambda row: (
            row["macro_f1"],
            row["balanced_acc"],
            row["auc"],
            -row["test_loss"],
        ),
        reverse=True,
    )[0]
    return winner


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train classical and hybrid CQ-CNN models on OASIS-2 coronal."
    )
    parser.add_argument(
        "--model",
        choices=["classical", "hybrid", "both"],
        default="both",
        help="Which model to train.",
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--train-limit", type=int, default=128)
    parser.add_argument("--test-limit", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--result-stem",
        default=None,
        help="Optional custom CSV filename stem for a single-model run.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.seeds) < args.trials:
        raise ValueError(
            f"Need at least {args.trials} seeds, but only received {len(args.seeds)}."
        )

    device = torch.device(args.device)
    model_names = ["classical", "hybrid"] if args.model == "both" else [args.model]
    if args.result_stem is not None and len(model_names) != 1:
        raise ValueError("--result-stem can only be used with --model classical or hybrid.")

    all_rows = []
    for model_name in model_names:
        config = ExperimentConfig(
            model_name=model_name,
            data_root=args.data_root,
            output_dir=args.output_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            trials=args.trials,
            seeds=args.seeds,
            train_limit=args.train_limit,
            test_limit=args.test_limit,
            device=device,
            result_stem=args.result_stem,
        )
        rows, _ = run_experiment(config)
        all_rows.extend(rows)

    winner = compare_models(args.output_dir)
    if winner is None:
        return

    print(
        "\nWinner by original-test macro_f1: "
        f"{winner['model']} | macro_f1={winner['macro_f1']:.4f} | "
        f"balanced_acc={winner['balanced_acc']:.4f} | "
        f"auc={winner['auc']:.4f}"
    )


if __name__ == "__main__":
    main()

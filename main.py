import argparse
from datetime import datetime
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Accumulator:
    def __init__(self, n: int):
        self.data = [0.0] * n

    def add(self, *args: float) -> None:
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, idx: int) -> float:
        return self.data[idx]


def load_array(
    data_arrays: Tuple[torch.Tensor, torch.Tensor],
    batch_size: int,
    is_train: bool = True,
) -> DataLoader:
    dataset = TensorDataset(*data_arrays)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)


def accuracy(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(torch.float32).sum())


def evaluate_loss(
    net: nn.Module,
    data_iter: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            loss = loss_fn(net(x), y)
            metric.add(loss.item() * y.shape[0], y.shape[0])
    return metric[0] / metric[1]


def evaluate_accuracy(
    net: nn.Module,
    data_iter: DataLoader,
    device: torch.device,
) -> float:
    net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for x, y in data_iter:
            x = x.to(device)
            y = y.to(device)
            metric.add(accuracy(net(x), y), y.numel())
    return metric[0] / metric[1]


def train_epoch_ch3(
    net: nn.Module,
    train_iter: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    net.train()
    metric = Accumulator(3)
    for x, y in train_iter:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_hat = net(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimizer.step()
        metric.add(loss.item() * y.shape[0], accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def select_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    base_columns = ["Height", "Weight", "Age", "Sex", "Position"]
    all_cols = df.columns.tolist()
    if "BallControl" in all_cols and "GKReflexes" in all_cols:
        start_idx = all_cols.index("BallControl")
        end_idx = all_cols.index("GKReflexes")
        skill_columns = all_cols[start_idx:end_idx + 1]
    else:
        excluded = {
            "ID",
            "Name",
            "Natinality",
            "Overal",
            "Potential",
            "PreferredFoot",
            "BirthDate",
            "PlayerWorkRate",
            "Value",
            "Wage",
            "Club",
            "Club_KitNumber",
            "Club_JoinedClub",
            "Club_ContractLength",
            "Sex",
            "Position",
        }
        skill_columns = [
            column
            for column in all_cols
            if column not in excluded
            and pd.api.types.is_numeric_dtype(df[column])
        ]
    return base_columns, skill_columns


@dataclass
class DatasetPack:
    train_iter: DataLoader
    test_iter: DataLoader
    input_dim: int
    num_classes: int
    idx_to_label: Dict[int, str]


Metrics = Dict[str, float]
ResultRow = Dict[str, Any]


@dataclass(frozen=True)
class ArchitectureConfig:
    name: str
    hidden: Sequence[int]
    activation: str
    dropout: float
    lr: float
    batchnorm: bool


def prepare_data(
    csv_path: str,
    batch_size: int,
    random_state: int,
) -> DatasetPack:
    df = cast(pd.DataFrame, pd.read_csv(filepath_or_buffer=csv_path))

    base_columns, skill_columns = select_columns(df)
    use_columns = base_columns + skill_columns
    data = df[use_columns].copy()

    data = pd.get_dummies(data, columns=["Sex"], dtype=np.float32)
    data = data.dropna()

    y_labels = sorted(
        str(label) for label in data["Position"].unique().tolist()
    )
    label_to_idx = {label: i for i, label in enumerate(y_labels)}
    idx_to_label: Dict[int, str] = {
        i: str(label) for label, i in label_to_idx.items()
    }

    y = data["Position"].map(label_to_idx).astype(np.int64)
    x = data.drop(columns=["Position"]).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.30,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.to_numpy(), dtype=torch.long)
    y_test_t = torch.tensor(y_test.to_numpy(), dtype=torch.long)

    train_iter = load_array(
        (x_train_t, y_train_t),
        batch_size=batch_size,
        is_train=True,
    )
    test_iter = load_array(
        (x_test_t, y_test_t),
        batch_size=batch_size,
        is_train=False,
    )

    return DatasetPack(
        train_iter=train_iter,
        test_iter=test_iter,
        input_dim=x_train_t.shape[1],
        num_classes=len(y_labels),
        idx_to_label=idx_to_label,
    )


def get_activation(name: str) -> nn.Module:
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Activacion no soportada: {name}")
    return activations[key]()


def build_mlp(
    input_dim: int,
    hidden_layers: Sequence[int],
    num_classes: int,
    activation: str = "relu",
    dropout: float = 0.0,
    use_batchnorm: bool = False,
) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = input_dim
    for units in hidden_layers:
        layers.append(nn.Linear(in_dim, units))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(units))
        layers.append(get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        in_dim = units
    layers.append(nn.Linear(in_dim, num_classes))
    return nn.Sequential(*layers)


def train_model(
    net: nn.Module,
    train_iter: DataLoader,
    test_iter: DataLoader,
    writer: SummaryWriter,
    model_name: str,
    device: torch.device,
    num_epochs: int,
    lr: float = 0.001,
    tb_log_every: int = 5,
) -> Metrics:
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.to(device)

    final_metrics: Metrics = {
        "train_loss": 0.0,
        "test_loss": 0.0,
        "train_accuracy": 0.0,
        "test_accuracy": 0.0,
    }
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_epoch_ch3(
            net,
            train_iter,
            loss_fn,
            optimizer,
            device,
        )
        test_loss = evaluate_loss(net, test_iter, loss_fn, device)
        test_acc = evaluate_accuracy(net, test_iter, device)

        should_log_tb = (
            epoch == 1
            or epoch == num_epochs
            or epoch % tb_log_every == 0
        )
        if should_log_tb:
            writer.add_scalar("metrics/loss/train", train_loss, epoch)
            writer.add_scalar("metrics/loss/test", test_loss, epoch)
            writer.add_scalar("metrics/accuracy/train", train_acc, epoch)
            writer.add_scalar("metrics/accuracy/test", test_acc, epoch)

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            print(
                f"[{model_name}] epoca {epoch:03d}/{num_epochs} | "
                f"loss_train={train_loss:.4f} | "
                f"loss_test={test_loss:.4f} | "
                f"acc_train={train_acc:.4f} | "
                f"acc_test={test_acc:.4f}"
            )

        final_metrics = {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
        }

    writer.flush()
    writer.close()
    return final_metrics


def save_model_checkpoint(net: nn.Module, save_dir: str, filename: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, filename)
    torch.save(net.state_dict(), checkpoint_path)
    return checkpoint_path


def create_experiment_runs_dir(base_runs_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_dir = os.path.join(base_runs_dir, timestamp)
    os.makedirs(experiment_dir, exist_ok=True)
    return experiment_dir


def print_tensorboard_hint(runs_dir: str) -> None:
    print("\nTensorBoard")
    print("-----------")
    print(f"Logs guardados en: {runs_dir}")
    print("Para una vista limpia de la ultima corrida ejecuta:")
    print(f"tensorboard --logdir \"{runs_dir}\"")
    print("Si quieres comparar con historico completo:")
    print("tensorboard --logdir runs")
    print("Luego abre http://localhost:6006 en tu navegador.")


def run_base_models(
    data: DatasetPack,
    device: torch.device,
    num_epochs: int,
    runs_dir: str,
    tb_log_every: int,
) -> List[ResultRow]:
    base_defs = {
        "chico": [4, 4],
        "medio": [16, 16],
        "grande": [256, 256],
    }

    results: List[ResultRow] = []
    for model_name, hidden in base_defs.items():
        net = build_mlp(
            data.input_dim,
            hidden,
            data.num_classes,
            activation="relu",
        )
        model_runs_dir = os.path.join(runs_dir, "base", model_name)
        print(f"Registrando TensorBoard en: {model_runs_dir}")
        writer = SummaryWriter(log_dir=model_runs_dir)
        metrics = train_model(
            net=net,
            train_iter=data.train_iter,
            test_iter=data.test_iter,
            writer=writer,
            model_name=model_name,
            device=device,
            num_epochs=num_epochs,
            lr=0.001,
            tb_log_every=tb_log_every,
        )
        checkpoint_path = save_model_checkpoint(net, model_runs_dir, "model_final.pt")
        print(f"Modelo guardado en: {checkpoint_path}")
        row: ResultRow = {
            "modelo": model_name,
            "hidden_layers": str(hidden),
            "activation": "relu",
            "dropout": 0.0,
            "lr": 0.001,
            **metrics,
        }
        results.append(row)
    return results


def run_competition_models(
    data: DatasetPack,
    device: torch.device,
    num_epochs: int,
    runs_dir: str,
    tb_log_every: int,
) -> List[ResultRow]:
    architectures = [
        ArchitectureConfig(
            name="comp_1",
            hidden=[64],
            activation="relu",
            dropout=0.0,
            lr=0.001,
            batchnorm=False,
        ),
        ArchitectureConfig(
            name="comp_2",
            hidden=[128, 64],
            activation="relu",
            dropout=0.2,
            lr=0.001,
            batchnorm=True,
        ),
        ArchitectureConfig(
            name="comp_3",
            hidden=[256, 128, 64],
            activation="leakyrelu",
            dropout=0.3,
            lr=0.0005,
            batchnorm=True,
        ),
        ArchitectureConfig(
            name="comp_4",
            hidden=[32, 32, 32],
            activation="tanh",
            dropout=0.1,
            lr=0.001,
            batchnorm=False,
        ),
        ArchitectureConfig(
            name="comp_5",
            hidden=[128, 128],
            activation="elu",
            dropout=0.25,
            lr=0.0001,
            batchnorm=True,
        ),
    ]

    results: List[ResultRow] = []
    for cfg in architectures:
        net = build_mlp(
            input_dim=data.input_dim,
            hidden_layers=cfg.hidden,
            num_classes=data.num_classes,
            activation=cfg.activation,
            dropout=cfg.dropout,
            use_batchnorm=cfg.batchnorm,
        )
        model_runs_dir = os.path.join(runs_dir, "competition", cfg.name)
        print(f"Registrando TensorBoard en: {model_runs_dir}")
        writer = SummaryWriter(log_dir=model_runs_dir)
        metrics = train_model(
            net=net,
            train_iter=data.train_iter,
            test_iter=data.test_iter,
            writer=writer,
            model_name=cfg.name,
            device=device,
            num_epochs=num_epochs,
            lr=cfg.lr,
            tb_log_every=tb_log_every,
        )
        checkpoint_path = save_model_checkpoint(net, model_runs_dir, "model_final.pt")
        print(f"Modelo guardado en: {checkpoint_path}")
        row: ResultRow = {
            "modelo": cfg.name,
            "hidden_layers": str(cfg.hidden),
            "activation": cfg.activation,
            "dropout": cfg.dropout,
            "lr": cfg.lr,
            **metrics,
        }
        results.append(row)

    results = sorted(
        results,
        key=lambda item: (-item["test_accuracy"], item["test_loss"]),
    )
    return results


def print_results_table(results: List[ResultRow], title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for row in results:
        print(
            f"{row['modelo']:>10} | "
            f"hidden={row['hidden_layers']:<18} | "
            f"act={row['activation']:<9} | "
            f"drop={row['dropout']:<4} | "
            f"lr={row['lr']:<7} | "
            f"acc_test={row['test_accuracy']:.4f} | "
            f"loss_test={row['test_loss']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Practica RNA - MLPs en PyTorch con FIFA 2021",
    )
    parser.add_argument(
        "--csv-path",
        default="fifa2021_training.csv",
        help="Ruta al archivo de entrenamiento CSV",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size para DataLoaders",
    )
    parser.add_argument(
        "--base-epochs",
        type=int,
        default=500,
        help="Epocas para modelos base",
    )
    parser.add_argument(
        "--competition-epochs",
        type=int,
        default=300,
        help="Epocas para competencia",
    )
    parser.add_argument(
        "--run-competition",
        action="store_true",
        help="Si se activa, entrena tambien 5 arquitecturas de competencia",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs",
        help="Directorio de salida para TensorBoard",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para reproducibilidad",
    )
    parser.add_argument(
        "--tb-log-every",
        type=int,
        default=5,
        help="Frecuencia de registro en TensorBoard (ademas de epoca 1 y final)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo para entrenar",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Se pidio CUDA pero no esta disponible en este entorno"
            )
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    if args.tb_log_every < 1:
        raise ValueError("--tb-log-every debe ser >= 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    os.makedirs(args.runs_dir, exist_ok=True)
    experiment_runs_dir = create_experiment_runs_dir(args.runs_dir)

    data = prepare_data(
        args.csv_path,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    print(f"Clases Position: {data.idx_to_label}")
    print(
        f"Input dim: {data.input_dim} | "
        f"Num classes: {data.num_classes} | "
        f"Device: {device}"
    )

    base_results = run_base_models(
        data,
        device=device,
        num_epochs=args.base_epochs,
        runs_dir=experiment_runs_dir,
        tb_log_every=args.tb_log_every,
    )
    print_results_table(base_results, title="Resultados modelos base")

    if args.run_competition:
        comp_results = run_competition_models(
            data,
            device=device,
            num_epochs=args.competition_epochs,
            runs_dir=experiment_runs_dir,
            tb_log_every=args.tb_log_every,
        )
        print_results_table(
            comp_results,
            title="Resultados competencia (top por accuracy test)",
        )

    print_tensorboard_hint(experiment_runs_dir)


if __name__ == "__main__":
    main()

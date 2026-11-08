import math
import random
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ExperimentConfig:
    series_length: int = 600
    window_size: int = 24
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    seed: int = 42
    use_mps: bool = False
    epoch_delay: float = 0.1


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        window_size: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.input_projection = nn.Linear(1, d_model)

        self.position_embedding = nn.Parameter(
            torch.randn(1, window_size, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 2),
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.position_embedding
        encoded = self.transformer(x)

        last_token = encoded[:, -1, :]

        return self.classifier(last_token)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def generate_series(length: int, seed: int):
    rng = np.random.default_rng(seed)

    returns = np.zeros(length, dtype=np.float32)

    for index in range(1, length):
        cycle = 0.002 * math.sin(index / 12)
        momentum = 0.65 * returns[index - 1]
        noise = rng.normal(0, 0.004)

        returns[index] = momentum + cycle + noise

    prices = 100 * np.exp(np.cumsum(returns))

    return returns, prices


def make_dataset(returns, prices, window_size: int):
    features = []
    labels = []
    metadata = []

    for end in range(window_size, len(returns) - 1):
        window = returns[end - window_size:end]

        current_price = float(prices[end])
        next_price = float(prices[end + 1])
        label = 1 if next_price > current_price else 0

        features.append(window)
        labels.append(label)

        metadata.append(
            {
                "time": int(end),
                "current_value": current_price,
                "next_value": next_price,
            }
        )

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    # [样本数, 时间窗口, 特征数]
    x = x[:, :, np.newaxis]

    return x, y, metadata


def choose_device(use_mps: bool):
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def run_experiment(
    config: ExperimentConfig,
    progress_callback: Callable | None = None,
):
    """
    运行一次完整实验。

    progress_callback 会在每个 epoch 完成后收到：

    {
        "epoch": 1,
        "epochs": 10,
        "loss": 0.69,
        "accuracy": 0.55,
        "progress": 10
    }
    """

    set_seed(config.seed)

    device = choose_device(config.use_mps)

    returns, prices = generate_series(
        length=config.series_length,
        seed=config.seed,
    )

    x, y, metadata = make_dataset(
        returns=returns,
        prices=prices,
        window_size=config.window_size,
    )

    split_index = int(len(x) * 0.8)

    x_train = x[:split_index]
    y_train = y[:split_index]

    x_test = x[split_index:]
    y_test = y[split_index:]

    test_metadata = metadata[split_index:]

    train_dataset = TensorDataset(
        torch.tensor(x_train),
        torch.tensor(y_train),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )

    model = TimeSeriesTransformer(
        window_size=config.window_size,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    loss_function = nn.CrossEntropyLoss()

    history = []

    for epoch in range(1, config.epochs + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        sample_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = loss_function(logits, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_x)

            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_y).sum().item()
            sample_total += len(batch_y)

        average_loss = total_loss / sample_total
        training_accuracy = correct / sample_total

        epoch_result = {
            "epoch": epoch,
            "epochs": config.epochs,
            "loss": float(average_loss),
            "accuracy": float(training_accuracy),
            "progress": int(epoch / config.epochs * 100),
        }

        history.append(epoch_result)

        if progress_callback is not None:
            progress_callback(epoch_result)

        if config.epoch_delay > 0:
            time.sleep(config.epoch_delay)

    model.eval()

    with torch.no_grad():
        test_tensor = torch.tensor(x_test).to(device)
        test_logits = model(test_tensor)

        probabilities = torch.softmax(test_logits, dim=1)
        predictions = test_logits.argmax(dim=1).cpu().numpy()
        confidence = (
            probabilities.max(dim=1)
            .values
            .cpu()
            .numpy()
        )

    result_rows = []

    for index, prediction in enumerate(predictions):
        actual = int(y_test[index])
        predicted = int(prediction)
        is_correct = predicted == actual

        row = test_metadata[index]

        result_rows.append(
            {
                "time": row["time"],
                "current_value": round(
                    row["current_value"],
                    3,
                ),
                "next_value": round(
                    row["next_value"],
                    3,
                ),
                "actual_direction": (
                    "上涨" if actual == 1 else "下跌"
                ),
                "predicted_direction": (
                    "上涨" if predicted == 1 else "下跌"
                ),
                "confidence": float(confidence[index]),
                "correct": bool(is_correct),
            }
        )

    test_accuracy = float(
        (predictions == y_test).mean()
    )

    series = [
        {
            "time": int(index),
            "value": float(value),
        }
        for index, value in enumerate(prices)
    ]

    return {
        "device": str(device),
        "series": series,
        "history": history,
        "results": result_rows,
        "metrics": {
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
            "test_accuracy": test_accuracy,
            "correct_count": int(
                (predictions == y_test).sum()
            ),
            "incorrect_count": int(
                (predictions != y_test).sum()
            ),
        },
    }


def print_progress(progress):
    print(
        f"Epoch {progress['epoch']:02d}/"
        f"{progress['epochs']:02d} | "
        f"Loss {progress['loss']:.4f} | "
        f"Accuracy {progress['accuracy']:.2%}"
    )


if __name__ == "__main__":
    test_config = ExperimentConfig(
        series_length=300,
        window_size=24,
        epochs=5,
        use_mps=False,
    )

    output = run_experiment(
        config=test_config,
        progress_callback=print_progress,
    )

    print()
    print("运行设备：", output["device"])
    print(
        "测试准确率：",
        f"{output['metrics']['test_accuracy']:.2%}",
    )
    print("预测记录：", len(output["results"]))
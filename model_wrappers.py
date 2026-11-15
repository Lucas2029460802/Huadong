from __future__ import annotations

import random

import numpy as np


class TorchMLPRegressor:
    """Small PyTorch regressor with a scikit-learn-like fit/predict interface."""

    def __init__(
        self,
        hidden_dim: int = 128,
        epochs: int = 30,
        batch_size: int = 8192,
        learning_rate: float = 1e-3,
        random_state: int = 2026,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.model_ = None
        self.x_mean_ = None
        self.x_std_ = None

    def _seed_torch(self) -> None:
        import torch

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _make_model(self, input_dim: int):
        try:
            import torch
            from torch import nn
        except Exception as exc:
            raise ImportError("使用 --framework pytorch 前，请先安装 PyTorch：pip install torch") from exc

        self._seed_torch()
        return nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

    def _transform(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        return (x - self.x_mean_) / self.x_std_

    def fit(self, x, y, sample_weight=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        self._seed_torch()

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        weights = (
            np.ones((len(y), 1), dtype=np.float32)
            if sample_weight is None
            else np.asarray(sample_weight, dtype=np.float32).reshape(-1, 1)
        )

        self.x_mean_ = np.nanmean(x, axis=0).astype(np.float32)
        self.x_std_ = np.nanstd(x, axis=0).astype(np.float32)
        self.x_std_[self.x_std_ < 1e-6] = 1.0
        x = self._transform(x)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = self._make_model(x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        dataset = TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(weights),
        )
        g = torch.Generator()
        g.manual_seed(self.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=g,
        )
        self.model_.train()
        for epoch in range(self.epochs):
            # 每轮固定 dropout 等与顺序相关的随机性；batch 顺序由 DataLoader 的 generator 决定
            torch.manual_seed(self.random_state + epoch * 1_000_003)
            for xb, yb, wb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                wb = wb.to(device)
                pred = self.model_(xb)
                loss = (wb * (pred - yb).pow(2)).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.device_ = str(device)
        self.model_.to("cpu")
        self.model_.eval()
        return self

    def predict(self, x) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("模型尚未训练")
        import torch

        x = self._transform(x)
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                xb = torch.from_numpy(x[start : start + self.batch_size])
                preds.append(self.model_(xb).cpu().numpy().reshape(-1))
        return np.concatenate(preds)

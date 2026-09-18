"""
Module 2: Local Engagement Prediction Models & Training Routines.

Implements:
1. PyTorch EngagementMLP (d -> 64 -> 32 -> 1) with weight getter/setter for Flower FL.
2. Training routine with Adam and optional FedProx proximal regularization.
3. Centralized baseline models: XGBoost Regressor and LightGBM Regressor.
4. Evaluation metrics calculation: MAE, RMSE, MSE, R^2.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class EngagementMLP(nn.Module):
    """
    Feedforward Neural Network for Engagement Score Prediction.
    Architecture: Input (d=13) -> Dense(64) -> ReLU -> Dense(32) -> ReLU -> Dense(1) -> Sigmoid
    """
    def __init__(self, input_dim: int = 13, hidden1: int = 64, hidden2: int = 32):
        super(EngagementMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid()  # Guarantees predictions stay in continuous [0, 1] range
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def get_model_parameters(model: nn.Module) -> List[np.ndarray]:
    """Extract model parameters as a list of NumPy arrays (for Flower FL)."""
    return [val.cpu().detach().numpy() for _, val in model.state_dict().items()]

def set_model_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays (for Flower FL)."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute regression evaluation metrics: MAE, RMSE, MSE, R^2."""
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()
    # Handle NaN or Inf if any
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1.0, neginf=0.0)
    
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mse": round(mse, 4),
        "r2": round(r2, 4)
    }

def train_mlp_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    global_params: Optional[List[torch.Tensor]] = None,
    proximal_mu: float = 0.0
) -> float:
    """
    Train MLP for one epoch.
    Supports FedProx proximal penalty if proximal_mu > 0 and global_params is provided.
    """
    model.train()
    running_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)

        # FedProx proximal term: (mu / 2) * ||w - w_t||^2
        if proximal_mu > 0.0 and global_params is not None:
            prox_loss = 0.0
            for param, global_p in zip(model.parameters(), global_params):
                prox_loss += torch.sum((param - global_p.to(device)) ** 2)
            loss = loss + (proximal_mu / 2.0) * prox_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(y_batch)
        total_samples += len(y_batch)

    return running_loss / max(1, total_samples)

def evaluate_mlp(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = torch.device("cpu")
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate MLP on test DataLoader, returning metrics, ground truths, and predictions."""
    model.eval()
    all_y = []
    all_preds = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch)
            all_y.extend(y_batch.numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())

    y_true = np.array(all_y)
    y_pred = np.array(all_preds)
    metrics = compute_metrics(y_true, y_pred)
    return metrics, y_true, y_pred

def train_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
    """Train an XGBoost regressor baseline."""
    model = xgb.XGBRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=2
    )
    model.fit(X_train, y_train)
    return model

def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor baseline."""
    model = lgb.LGBMRegressor(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        random_state=42,
        verbose=-1,
        n_jobs=2
    )
    model.fit(X_train, y_train)
    return model

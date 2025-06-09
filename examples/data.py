import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris, load_digits, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np

DATA_PARAMS = {
    "test_size": 0.25,        # 25% → temp
    "validation_split": 0.5,  # 50% of temp → val/test (12.5% each)
}


def preprocess_tabular(arr: np.ndarray | torch.Tensor) -> torch.Tensor:
    X = torch.tensor(arr, dtype=torch.float32) if not isinstance(arr, torch.Tensor) else arr.float()
    mins = X.min(dim=0).values
    maxs = X.max(dim=0).values
    diffs = (maxs - mins).clamp(min=1.0)
    X = (X - mins) / diffs         # [0,1]
    return X * 2.0 - 1.0           # [-1,1]


def load_dataset(
    dataset: str,
    batch_size: int,
    num_workers: int = 3,
    seed: int = None,
    subset_size: int = None,
    n_classes: int = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load and preprocess one of {"iris", "digits", "wine", "breast_cancer"}.
    Args:
        dataset: which dataset to load.
        batch_size, num_workers, seed: passed to DataLoader / splitting.
        n_classes: if not None, keep only examples with label < n_classes.
        subset_size: if not None, after filtering by n_classes, take a stratified subset of this size.
    Returns:
        train_loader, val_loader, test_loader (75%/12.5%/12.5% splits, stratified).
    """
    ds = dataset.lower()

    if ds == "iris":
        data = load_iris()
        X_raw, y_raw = data.data, data.target

    elif ds == "wine":
        data = load_wine()
        X_raw, y_raw = data.data, data.target

    elif ds == "breast_cancer":
        data = load_breast_cancer()
        X_raw, y_raw = data.data, data.target

    elif ds == "digits":
        data = load_digits()
        X_raw, y_raw = data.images, data.target  # X_raw: (N, 8, 8), y_raw: (N,)

    else:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    # Filter by n_classes if given
    if n_classes is not None:
        mask = y_raw < n_classes
        X_raw = X_raw[mask]
        y_raw = y_raw[mask]

    N_filtered = len(y_raw)
    if N_filtered == 0:
        raise ValueError(f"No examples remain after filtering for n_classes={n_classes}.")

    # Stratified subset if requested
    if subset_size is not None:
        if subset_size < (n_classes or 1):
            raise ValueError(f"subset_size must be ≥ n_classes (got {subset_size}).")
        if subset_size > N_filtered:
            raise ValueError(f"subset_size ({subset_size}) exceeds {N_filtered} available examples.")
        strat = y_raw if len(np.unique(y_raw)) > 1 else None
        X_raw, _, y_raw, _ = train_test_split(
            X_raw, y_raw, train_size=subset_size,
            random_state=seed, shuffle=True, stratify=strat
        )

    # Preprocess features
    if ds != "digits":
        X_tensor = preprocess_tabular(X_raw)            # (N_final, n_features)
        y_tensor = torch.tensor(y_raw, dtype=torch.long)
    else:
        imgs = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1) / 16.0  # (N,1,8,8)
        imgs = imgs[:, :, 1:7, 1:7]               # crop → (N,1,6,6)
        imgs = F.avg_pool2d(imgs, kernel_size=2)  # → (N,1,3,3)
        X_tensor = (imgs * 2.0 - 1.0).view(len(imgs), -1)  # (N,9)
        y_tensor = torch.tensor(y_raw, dtype=torch.long)

    # Convert to numpy for splitting
    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy()

    # Stratify only if more than one class remains
    strat0 = y_np if len(np.unique(y_np)) > 1 else None
    X_train_np, X_temp_np, y_train_np, y_temp_np = train_test_split(
        X_np, y_np,
        test_size=DATA_PARAMS["test_size"],
        random_state=seed,
        shuffle=True,
        stratify=strat0
    )
    strat1 = y_temp_np if len(np.unique(y_temp_np)) > 1 else None
    X_val_np, X_test_np, y_val_np, y_test_np = train_test_split(
        X_temp_np, y_temp_np,
        test_size=DATA_PARAMS["validation_split"],
        random_state=seed,
        shuffle=True,
        stratify=strat1
    )

    # Convert back to tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    X_val = torch.tensor(X_val_np, dtype=torch.float32)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.long)
    y_val = torch.tensor(y_val_np, dtype=torch.long)
    y_test = torch.tensor(y_test_np, dtype=torch.long)

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import cudaq

import os
from typing import Any, Dict
from tqdm import tqdm

from torchq.models import QNN

from examples.classic_models import ParamMatchedFCN
from examples.train_functions import train_step, val_step, test_step
from examples.utils import Accuracy, get_unique_filename, set_seed, count_parameters
from examples.data import load_dataset

# Global configuration dictionaries for data, training, and model parameters.
DATASET = "iris"
IN_FEATURES = 4
OUT_FEATURES = 3
SUBSET_SIZE = None

TRAINING_PARAMS: Dict[str, Any] = {
    "batch_size": 8,
    "num_workers": 4,
    "learning_rate": 0.1,
    "epochs": 20,
    "seed": 42,
}

QNN_PARAMS: Dict[str, Any] = {
    "in_features": IN_FEATURES,
    "out_features": OUT_FEATURES,
    "num_layers": 9,
    "shots": 2048,
    "feature_map": "z",
    "var_form": "efficientsu2",
    "reupload": False,
}

FCN_PARAMS: Dict[str, int] = {
    "in_features": IN_FEATURES,
    "out_features": OUT_FEATURES,
}


def main() -> None:
    r"""Main training and evaluation loop.

    Loads the wine dataset, preprocesses the data, splits it into train, validation,
    and test sets, and trains both a quantum neural network (QNN) and a classical
    fully connected network (FCN). Logs training and validation metrics to TensorBoard,
    and saves the QNN model state after training.
    """
    # Set random seeds for reproducibility.
    set_seed(TRAINING_PARAMS["seed"])

    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set target for cudaq based on device type and gate fusion.
    if device.type == "cuda":
        cudaq.set_target('nvidia', option='fp32')
    else:
        cudaq.set_target("qpp-cpu")

    train_loader, val_loader, test_loader = load_dataset(
        dataset=DATASET,
        batch_size=TRAINING_PARAMS["batch_size"],
        num_workers=TRAINING_PARAMS["num_workers"],
        seed=TRAINING_PARAMS["seed"],
        subset_size=SUBSET_SIZE,
        n_classes=OUT_FEATURES,
    )

    # Initialize models.
    model_qnn: nn.Module = QNN(
        in_features=QNN_PARAMS["in_features"],
        out_features=QNN_PARAMS["out_features"],
        num_layers=QNN_PARAMS["num_layers"],
        shots=QNN_PARAMS["shots"],
        feature_map=QNN_PARAMS["feature_map"],
        var_form=QNN_PARAMS["var_form"],
        reupload=QNN_PARAMS["reupload"],
    )
    d: int = count_parameters(model_qnn)
    print(f"Quantum model parameter count:", d)

    model_fcn: nn.Module = ParamMatchedFCN(
        in_features=FCN_PARAMS["in_features"],
        out_features=FCN_PARAMS["out_features"],
        d=d,
    )
    real_d: int = count_parameters(model_fcn)
    print(f"Classical model parameter count:", real_d)

    model_qnn.to(device)
    model_fcn.to(device)

    # Define loss function and optimizers.
    criterion = nn.CrossEntropyLoss()
    optimizer_qnn = optim.Adam(
        model_qnn.parameters(), lr=TRAINING_PARAMS["learning_rate"], weight_decay=0.01
    )
    optimizer_fcn = optim.Adam(
        model_fcn.parameters(), lr=TRAINING_PARAMS["learning_rate"], weight_decay=0.01
    )

    # Create TensorBoard writers with unique log directories.
    writer_qnn = SummaryWriter(get_unique_filename("runs/quantum_neural_network"))
    writer_fcn = SummaryWriter(get_unique_filename("runs/multilayer_perceptron"))
    print(
        "TensorBoard logs saved to: runs/quantum_neural_network, runs/multilayer_perceptron "
        "and runs/hybrid_quantum_neural_network"
    )
    print("Run 'tensorboard --logdir=runs' to view the logs.")

    # Initialize accuracy trackers.
    acc_train_qnn = Accuracy()
    acc_val_qnn = Accuracy()
    acc_test_qnn = Accuracy()
    acc_train_fcn = Accuracy()
    acc_val_fcn = Accuracy()
    acc_test_fcn = Accuracy()

    # Initial evaluation on the test set.
    test_step(model_qnn, test_loader, criterion, acc_test_qnn, device)
    test_step(model_fcn, test_loader, criterion, acc_test_fcn, device)
    print("Initial test evaluation logged to TensorBoard.")

    # # Training loop.
    for epoch in tqdm(range(1, TRAINING_PARAMS["epochs"] + 1)):
        acc_train_qnn.reset()
        train_step(
            model_qnn,
            train_loader,
            criterion,
            optimizer_qnn,
            acc_train_qnn,
            writer_qnn,
            epoch,
            device,
        )
        acc_val_qnn.reset()
        val_step(
            model_qnn, val_loader, criterion, acc_val_qnn, writer_qnn, epoch, device
        )

        acc_train_fcn.reset()
        train_step(
            model_fcn,
            train_loader,
            criterion,
            optimizer_fcn,
            acc_train_fcn,
            writer_fcn,
            epoch,
            device,
        )
        acc_val_fcn.reset()
        val_step(
            model_fcn, val_loader, criterion, acc_val_fcn, writer_fcn, epoch, device
        )

    # Final test evaluation.
    acc_test_qnn.reset()
    test_step(model_qnn, test_loader, criterion, acc_test_qnn, device)
    acc_test_fcn.reset()
    test_step(model_fcn, test_loader, criterion, acc_test_fcn, device)

    print(f"Final test accuracy (QNN): {acc_test_qnn.compute():.4f}")
    print(f"Final test accuracy (MLP): {acc_test_fcn.compute():.4f}")

    # Save the quantum model state.
    torch.save(model_qnn.state_dict(), "models/qnn_state_dict.pth")
    torch.save(model_fcn.state_dict(), "models/mlp_state_dict.pth")


if __name__ == "__main__":
    main()

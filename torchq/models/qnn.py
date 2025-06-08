import torch
import torch.nn as nn
import torchq.quantum.feature_maps as qfm
import torchq.quantum.variational_forms as qvf
from torchq.nn.layers import QuantumLayer
from typing import Union


class QNN(nn.Module):
    r"""Quantum neural network module.

    Applies a quantum layer. The quantum layer uses a parameterized quantum circuit defined
    by a feature map and a variational form, allowing integration of quantum computations
    within PyTorch models.

    Args:
        in_features (int): Number of input features (qubits).
        out_features (int): Number of output features.
        num_layers (int, optional): Number of variational layers in the quantum circuit.
            Defaults to 1.
        shots (int, optional): Number of shots for circuit execution. Defaults to 1024.
        feature_map (Union[str, qfm.FeatureMap], optional): Specification for the feature map.
            If a string is provided, supported values include "z" or "zz". Defaults to "zz".
        var_form (Union[str, qvf.VariationalForm], optional): Specification for the variational form.
            If a string is provided, supported values include "efficientsu2", "realamplitudes",
            or "paulitwodesign". Defaults to "efficientsu2".
        reupload (bool, optional): Whether to use reuploading of the feature map at each layer.
            Defaults to False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 1,
        shots: int = 248,
        feature_map: Union[str, qfm.FeatureMap] = "z",
        var_form: Union[str, qvf.VariationalForm] = "efficientsu2",
        reupload: bool = False,
    ) -> None:
        super().__init__()

        self.quantum_layer: nn.Module = QuantumLayer(
            in_features=in_features,
            out_features=out_features,
            num_layers=num_layers,
            shots=shots,
            feature_map=feature_map,
            var_form=var_form,
            reupload=reupload,
        )

        self.in_features: int = in_features
        self.out_features: int = out_features

    def forward(self, x) -> torch.Tensor:
        r"""Forward pass of the QNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor after quantum processing and linear transformation.
        """

        return self.quantum_layer(x)


class HybridQNN(nn.Module):
    r"""Hybrid quantum neural network module.

    Applies a quantum layer followed by a classical linear layer. The quantum layer
    uses a parameterized quantum circuit defined by a feature map and a variational form,
    allowing integration of quantum computations within PyTorch models.

    Args:
        in_features (int): Number of input features (qubits).
        out_features (int): Number of output features.
        num_layers (int, optional): Number of variational layers in the quantum circuit.
            Defaults to 1.
        shots (int, optional): Number of shots for circuit execution. Defaults to 1024.
        feature_map (Union[str, qfm.FeatureMap], optional): Specification for the feature map.
            If a string is provided, supported values include "z" or "zz". Defaults to "zz".
        var_form (Union[str, qvf.VariationalForm], optional): Specification for the variational form.
            If a string is provided, supported values include "efficientsu2", "realamplitudes",
            or "paulitwodesign". Defaults to "efficientsu2".
        reupload (bool, optional): Whether to use reuploading of the feature map at each layer.
            Defaults to False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 1,
        shots: int = 248,
        feature_map: Union[str, qfm.FeatureMap] = "z",
        var_form: Union[str, qvf.VariationalForm] = "efficientsu2",
        reupload: bool = False,
    ) -> None:
        super().__init__()

        self.quantum_layer: nn.Module = QuantumLayer(
            in_features=in_features,
            out_features=2**in_features,
            num_layers=num_layers,
            shots=shots,
            feature_map=feature_map,
            var_form=var_form,
            reupload=reupload,
        )

        self.linear: nn.Module = nn.Linear(2**in_features, out_features)
        self.in_features: int = in_features
        self.out_features: int = out_features

    def forward(self, x) -> torch.Tensor:
        r"""Forward pass of the QNN.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor after quantum processing and linear transformation.
        """

        quantum_output: torch.Tensor = self.quantum_layer(x)
        return self.linear(quantum_output)

import cudaq
import torch
import torch.nn as nn
from torchq.nn.functional import QuantumFunction, quantum_fn
import torchq.quantum.feature_maps as qfm
import torchq.quantum.variational_forms as qvf
from torchq.nn.init import beta_initialization
from typing import Union, List
from math import pi


class QuantumLayer(nn.Module):
    r"""Quantum layer for hybrid quantum-classical networks.

    Constructs a parameterized quantum circuit using a feature map and variational form,
    then evaluates it via a custom autograd function. The resulting quantum output is
    subsequently passed to a classical layer.

    Args:
        in_features (int): Number of input features (qubits).
        num_layers (int, optional): Number of variational layers in the circuit. Defaults to 1.
        shots (int, optional): Number of shots for circuit evaluation. Defaults to 1024.
        feature_map (Union[str, qfm.FeatureMap], optional): Feature map specification.
            If a string, supported values include "z" and "zz". Defaults to "zz".
        var_form (Union[str, qvf.VariationalForm], optional): Variational form specification.
            If a string, supported values include "efficientsu2", "realamplitudes", and "paulitwodesign".
            Defaults to "efficientsu2".
        reupload (bool, optional): Whether to use reuploading of the feature map for each layer.
            Defaults to False.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_layers: int = 1,
        shots: int = 1024,
        feature_map: Union[str, qfm.FeatureMap] = "z",
        var_form: Union[str, qvf.VariationalForm] = "efficientsu2",
        reupload: bool = False,
    ) -> None:
        super().__init__()
        if out_features > 2**in_features:
            raise ValueError(
                f"out_features ({out_features}) must not exceed 2**num_qubits "
                f"({2**in_features})."
            )

        if out_features < 2:
            raise ValueError(
                f"out_features ({out_features}) must be greater than 1."
            )

        self.in_features: int = in_features
        self.out_features: int = out_features
        self.num_layers: int = num_layers
        self._shots: int = shots
        self.reupload: bool = reupload

        # Select feature map based on string identifier if needed.
        if isinstance(feature_map, str):
            fm: str = feature_map.lower()
            if fm == "z":
                self.feature_map = qfm.ZFeatureMap(
                    in_features=in_features, num_layers=1
                )
            elif fm == "zz":
                self.feature_map = qfm.ZZFeatureMap(
                    in_features=in_features, num_layers=1
                )
            else:
                raise ValueError("Unknown feature_map string: " + feature_map)
        elif feature_map is None:
            self.feature_map = qfm.ZFeatureMap(in_features=in_features, num_layers=1)
        else:
            self.feature_map = feature_map

        # Select variational form based on string identifier if needed.
        vf_num_qubits = max(self.in_features, self.out_features)
        if isinstance(var_form, str):
            vf: str = var_form.lower()
            if vf == "paulitwodesign":
                self.var_form = qvf.PauliTwoDesign(
                    num_qubits=vf_num_qubits, num_layers=num_layers
                )
            elif vf == "realamplitudes":
                self.var_form = qvf.RealAmplitudes(
                    num_qubits=vf_num_qubits, num_layers=num_layers
                )
            elif vf == "efficientsu2":
                self.var_form = qvf.EfficientSU2(
                    num_qubits=vf_num_qubits, num_layers=num_layers
                )
            else:
                raise ValueError("Unknown var_form: " + var_form)
        elif var_form is None:
            self.var_form = qvf.EfficientSU2(
                num_qubits=vf_num_qubits, num_layers=num_layers
            )
        else:
            self.var_form = var_form

        # Create the quantum circuit function.
        self.quantum_circuit = QuantumFunction(
            in_features=self.in_features,
            out_features=self.out_features,
            feature_map=self.feature_map,
            var_form=self.var_form,
            shots=self._shots,
            reupload=self.reupload,
            shift=pi / 2,
        )

        # Initialize variational parameters.
        param_count: int = self.var_form.get_parameter_count()
        self._theta = nn.Parameter(torch.rand(param_count))
        beta_initialization(self._theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass for the quantum layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output of the quantum circuit.
        """
        return quantum_fn(self._theta, x, self.quantum_circuit)

    def __str__(self) -> str:
        feature_list: List[float] = torch.rand(self.in_features).tolist()
        theta_list: List[float] = self._theta.detach().tolist()
        kernel_str = cudaq.draw(self.quantum_circuit.kernel, feature_list, theta_list)
        return kernel_str

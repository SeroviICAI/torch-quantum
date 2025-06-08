import cudaq
import torch
import torch.nn as nn
import torchq.quantum as qu
import torchq.quantum.feature_maps as qfm
import torchq.quantum.variational_forms as qvf
from torch.autograd import Function
from typing import List, Tuple
from math import pi


class QuantumFunction(Function):
    r"""Custom autograd function for evaluating parameterized quantum circuits.

    This function wraps the evaluation of a quantum circuit defined by a feature map and
    a variational form. Gradients are computed via a finite-difference (parameter shift) rule.
    Fully compatible with functorch transforms via forward-mode setup_context.

    Attributes:
        in_features (int): Number of input features (qubits).
        out_features (int): Number of output features (qubits to measure).
        feature_map (qfm.FeatureMap): The feature map object.
        var_form (qvf.VariationalForm): The variational form object.
        backend (Any): Quantum backend for circuit evaluation.
        shots (int): Number of shots for circuit evaluation.
        reupload (bool): Whether to use reuploading strategy.
        _kernel (cudaq.Kernel): The pre-built combined circuit kernel.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        feature_map: qfm.FeatureMap,
        var_form: qvf.VariationalForm,
        shots: int,
        reupload: bool = False,
        shift: float = pi / 2,
    ) -> None:
        r"""Initialize QuantumFunction with circuit components.

        Args:
            in_features (int): Number of input qubits.
            out_features (int): Number of qubits to measure.
            feature_map (FeatureMap): Data encoding circuit.
            var_form (VariationalForm): Parameterized ansatz.
            shots (int): Number of measurement shots.
            reupload (bool): If True, interleave feature map each layer. Defaults to False.
            shift (float): Finite-difference shift for parameter shift rule. Defaults to pi/2.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.feature_map = feature_map
        self.var_form = var_form
        self.shots = shots
        self.reupload = reupload
        self._kernel: cudaq.Kernel = self.build_kernel()
        self.shift = shift

    @property
    def kernel(self) -> cudaq.Kernel:
        r"""Return the pre-built quantum circuit kernel."""
        return self._kernel

    def build_kernel(self) -> cudaq.Kernel:
        r"""Construct the quantum circuit kernel by combining feature map and variational form.

        Depending on whether reuploading is enabled, the circuit applies the feature map
        once or repeatedly for each layer.

        Returns:
            cudaq.Kernel: The constructed quantum circuit.
        """
        kernel, x, thetas = cudaq.make_kernel(List[float], List[float])
        qvec = kernel.qalloc(self.in_features)
        if not self.reupload:
            kernel.apply_call(self.feature_map.kernel, qvec, x)
            kernel.apply_call(self.var_form.kernel, qvec, thetas)
        else:
            ptr: int = 0
            # Apply the feature map and layer-specific variational form in a loop.
            for _, layer in self.var_form.layers.items():
                kernel.apply_call(self.feature_map.kernel, qvec, x)
                count: int = layer.get_parameter_count()
                kernel.apply_call(layer.kernel, qvec, thetas[ptr : ptr + count])
                ptr += count
        return kernel

    def run(self, theta_vals: List[nn.Parameter], x: torch.Tensor) -> torch.Tensor:
        r"""Execute the quantum circuit for a batch of inputs.

        For each input sample, the circuit is executed and every qubit is later measured in
        the Z-basis, mapping every shot to one of C classes.

        Args:
            theta_vals (List[nn.Parameter]): List of circuit parameters.
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Tensor of expectation values with shape (batch_size, out_features).
        """
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected input tensor with {self.in_features} features (got {x.shape[1]})."
            )

        param_count = self.var_form.get_parameter_count()
        if len(theta_vals) != param_count:
            raise ValueError(
                f"Expected {param_count} variational parameters (got {len(theta_vals)})."
            )

        batch_size = x.shape[0]
        num_classes = self.out_features
        device = x.device

        # Convert nn.Parameters into a plain Python list of floats.
        theta_list: List[float] = [float(p.detach().item()) for p in theta_vals]

        # Pre-allocate the result tensor (batch_size × num_classes), on the same device as x.
        results = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=device)

        # Launch one cudaq.sample_async(...) per batch element.
        futures = []
        for i in range(batch_size):
            # Detach the i-th row of x, move to CPU, convert to plain Python list of floats.
            feature_row: List[float] = x[i].detach().cpu().tolist()

            # Launch sample_async; returns an AsyncSampleResult (a future-like object).
            # By default, qpu_id=0. If you have multiple GPUs/QPUs, you could pass qpu_id=i%num_qpus.
            fut = cudaq.sample_async(
                self.kernel,
                feature_row,
                theta_list,
                shots_count=self.shots
            )
            futures.append(fut)

        for i, fut in enumerate(futures):
            sample_res = fut.get()

            # Convert each measured bitstring (e.g. "011") → integer mod num_classes.
            bitstrings = sample_res.get_sequential_data()
            class_ids = torch.tensor(
                [int(bs, 2) % num_classes for bs in bitstrings],
                dtype=torch.long,
                device=device,
            )

            # Count how many times each class ID appeared, then normalize to probabilities.
            counts = torch.bincount(class_ids, minlength=num_classes).to(torch.float32)
            probs = counts / counts.sum()
            results[i] = probs

        return results

    @staticmethod
    def setup_context(
        ctx,
        inputs: tuple,
        output: torch.Tensor,
    ) -> None:
        r"""
        Save everything we need for backward.

        This is called *after* forward() returned, with exactly the
        inputs to forward and the single Tensor output.
        """
        theta, x, quantum_circuit = inputs
        # we don’t need grads for these non-tensors
        ctx.set_materialize_grads(False)
        ctx.quantum_circuit = quantum_circuit
        ctx.shift = quantum_circuit.shift
        # stash tensors for backward:
        ctx.save_for_backward(theta, x, output)

    @staticmethod
    def forward(
        theta: torch.Tensor, x: torch.Tensor, quantum_circuit: "QuantumFunction"
    ) -> torch.Tensor:
        r"""
        Execute the circuit to get expectation values.

        Args:
            theta (Tensor): 1-D tensor of circuit parameters.
            x (Tensor):     shape (batch, in_features), the data.
            quantum_circuit (QuantumFunction): instance carrying the kernel.
            shift (float):  finite-difference shift.

        Returns:
            Tensor of shape (batch, out_features) with expectations.
        """
        return quantum_circuit.run(theta, x)

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None]:
        r"""Backward pass: compute gradients via parameter-shift.

        Args:
            ctx: Autograd context containing saved tensors.
            grad_output (torch.Tensor): Gradient of loss w.r.t. output.

        Returns:
            Tuple[torch.Tensor, None, None, None]: Gradient w.r.t. theta, Nones.
        """
        theta, x, _ = ctx.saved_tensors
        device = theta.device
        gradients: torch.Tensor = torch.zeros_like(theta, device=device)
        for i in range(theta.numel()):
            theta_plus: torch.Tensor = theta.clone()
            theta_plus[i] += ctx.shift
            y_plus: torch.Tensor = ctx.quantum_circuit.run(theta_plus, x)
            theta_minus: torch.Tensor = theta.clone()
            theta_minus[i] -= ctx.shift
            y_minus: torch.Tensor = ctx.quantum_circuit.run(theta_minus, x)
            diff: torch.Tensor = (y_plus - y_minus) / (2 * ctx.shift)
            gradients[i] = torch.sum(grad_output * diff)
        return gradients, None, None


# Alias for nn.Modules
quantum_fn = QuantumFunction.apply

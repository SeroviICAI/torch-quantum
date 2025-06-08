import cudaq
import math
import itertools
import torchq.quantum as qu
import torchq.quantum.feature_maps as qfm
from typing import List, Union
from collections import OrderedDict


class PauliBlock(qu.Ansatz):
    r"""Parameterized Pauli block for data encoding.

    Constructs a block based on a given Pauli string that acts on specified qubits.
    The block first applies basis-change gates (Hadamard for X and RX(π/2) for Y),
    then entangles the non-identity operations with CX gates, applies a parameterized
    RZ rotation on the last non-identity qubit, and finally undoes the basis change.

    Circuit Diagram Example (for indices [0,1] with pauli_string "ZY" and alpha 2.0):

    .. code-block:: text

         ╭───────────╮                                             ╭────────────╮
    q0 : ┤ rx(1.571) ├──●───────────────────────────────────────●──┤ rx(-1.571) ├
         ╰───────────╯╭─┴─╮╭─────────────────────────────────╮╭─┴─╮╰────────────╯
    q1 : ─────────────┤ x ├┤ rz(2.0*(pi - x[0])*(pi - x[1])) ├┤ x ├──────────────
                      ╰───╯╰─────────────────────────────────╯╰───╯

    Args:
        indices (List[int]): Qubit indices on which to apply the block.
        pauli_string (str): Pauli string specifying operations (e.g. "XZI").
        alpha (float): Scaling factor for the RZ rotation.

    References:
    [1] Havlicek et al. Supervised learning with quantum enhanced feature spaces,
    `Nature 567, 209-212 (2019) <https://www.nature.com/articles/s41586-019-0980-2>`__.
    """

    def __init__(self, indices: List[int], pauli_string: str, alpha: float) -> None:
        self.indices: List[int] = indices
        # Reverse the pauli string to align with the ordering of the provided qubits.
        self.pauli_string: str = pauli_string[::-1]
        self.alpha: float = alpha
        super().__init__(len(indices))

    def build_kernel(self) -> cudaq.Kernel:
        r"""Build the quantum circuit kernel for the Pauli block.

        Returns:
            cudaq.Kernel: The kernel implementing the parameterized Pauli block.
        """
        param_count = self.get_parameter_count()
        kernel, qvec, params = cudaq.make_kernel(
            cudaq.qvector, list[float]
        )

        def self_product(vals: cudaq.QuakeValue) -> float:
            # Computes the product ∏ₖ (π - vₖ) for the parameter list.
            if param_count == 1:
                return vals[0]
            prod = 1.0
            for i in range(param_count):
                # index into vals
                prod *= math.pi - vals[i]
            return prod

        non_identity = [i for i, ch in enumerate(self.pauli_string) if ch != "I"]
        params = [params[self.indices[j]] for j in non_identity]

        # Pre-rotation: apply basis change for X and Y.
        for i, ch in enumerate(self.pauli_string):
            if ch == "X":
                kernel.h(qvec[self.indices[i]])
            elif ch == "Y":
                kernel.rx(math.pi / 2, qvec[self.indices[i]])

        # Apply entangling CX gates between successive non-identity operations.
        for i in range(len(non_identity) - 1):
            kernel.cx(
                qvec[self.indices[non_identity[i]]],
                qvec[self.indices[non_identity[i + 1]]],
            )

        # Apply parameterized RZ gate on the last non-identity qubit.
        if non_identity:
            kernel.rz(
                self.alpha * self_product(params),
                qvec[self.indices[non_identity[-1]]],
            )

        # Uncompute the entanglement by reversing the CX gates.
        for i in reversed(range(len(non_identity) - 1)):
            kernel.cx(
                qvec[self.indices[non_identity[i]]],
                qvec[self.indices[non_identity[i + 1]]],
            )

        # Post-rotation: revert the basis change.
        for i, ch in enumerate(self.pauli_string):
            if ch == "X":
                kernel.h(qvec[self.indices[i]])
            elif ch == "Y":
                kernel.rx(-math.pi / 2, qvec[self.indices[i]])
        return kernel

    def get_parameter_count(self) -> int:
        r"""Return the number of parameters in the Pauli block.

        Returns:
            int: Count of non-identity operations in the pauli string.
        """
        return sum(ch != "I" for ch in self.pauli_string)


class PauliFeatureMapLayer(qu.Ansatz):
    r"""Pauli feature map layer for encoding classical data into quantum states.

    This layer first applies a Hadamard transform to all qubits, then for each Pauli string
    provided it generates groups of qubits based on the specified entanglement strategy and
    applies a corresponding Pauli block.

    Args:
        num_qubits (int): Number of qubits.
        paulis (List[str]): List of Pauli strings that define each block.
        entanglement (Union[str, List[List[int]]], optional): Strategy for grouping qubits.
            Supported: ``"full"`` for all combinations or custom grouping. Defaults to ``"full"``.
        alpha (float, optional): Scaling factor for the parameterized RZ rotations. Defaults to 2.0.

    References:
    [1] Havlicek et al. Supervised learning with quantum enhanced feature spaces,
    `Nature 567, 209-212 (2019) <https://www.nature.com/articles/s41586-019-0980-2>`__.
    """

    @staticmethod
    def get_entangler_map(
        num_qubits: int, block_size: int, entanglement: Union[str, List[List[int]]]
    ) -> List[List[int]]:
        r"""Generate qubit groups (entangler map) for a given block size.

        Args:
            num_qubits (int): Total number of qubits.
            block_size (int): Number of qubits per block (length of the Pauli string).
            entanglement (Union[str, List[List[int]]]): Entanglement strategy or custom groups.

        Returns:
            List[List[int]]: List of qubit index groups.
        """

        if block_size == 1:
            return [[i] for i in range(num_qubits)]
        if isinstance(entanglement, str):
            if entanglement.lower() == "full":
                return [
                    list(c)
                    for c in itertools.combinations(range(num_qubits), block_size)
                ]
            else:
                raise ValueError(f"Unsupported entanglement strategy: {entanglement}")
        if isinstance(entanglement, list):
            if isinstance(entanglement[0], int):
                if len(entanglement) != block_size:
                    raise ValueError(
                        "Provided entanglement list does not match block size"
                    )
                return [entanglement]
            for sub in entanglement:
                if len(sub) != block_size:
                    raise ValueError(
                        "One of the provided entanglement sublists does not match block size"
                    )
            return entanglement
        return []

    def __init__(
        self,
        num_qubits: int,
        paulis: List[str],
        entanglement: Union[str, List[List[int]]] = "full",
        alpha: float = 2.0,
    ) -> None:
        max_pauli_len = max(len(p) for p in paulis)
        if num_qubits < max_pauli_len:
            raise ValueError(
                f"num_qubits (number of qubits) must be at least equal to the maximum pauli length. "
                f"Got in_features={num_qubits} and maximum pauli length={max_pauli_len}."
            )
        self.num_qubits = num_qubits
        self.paulis = paulis
        self.entanglement = entanglement
        self.alpha = alpha
        self.blocks: OrderedDict[str, qu.Ansatz] = OrderedDict()
        super().__init__(num_qubits)

    def build_kernel(self) -> cudaq.Kernel:
        r"""Build the quantum circuit kernel for the Pauli feature map layer.

        Returns:
            cudaq.Kernel: The kernel that applies an initial Hadamard on all qubits
            followed by a series of Pauli blocks.
        """
        kernel, qvec, x = cudaq.make_kernel(cudaq.qvector, List[float])

        kernel.h(qvec)
        for pauli in self.paulis:
            block_size: int = len(pauli)
            entangler_list: List[List[int]] = self.get_entangler_map(
                self.num_qubits, block_size, self.entanglement
            )
            for indices in entangler_list:
                pb: PauliBlock = PauliBlock(indices, pauli, self.alpha)
                self.blocks[f"pauli_{pauli}_{indices}"] = pb
                kernel.apply_call(pb.kernel, qvec, x)
        return kernel

    def get_parameter_count(self) -> int:
        r"""Return the total number of parameters in the Pauli feature map layer.

        Returns:
            int: Sum of parameters over all blocks.
        """
        return self.num_qubits


class PauliFeatureMap(qfm.FeatureMap):
    r"""Pauli feature map for quantum data encoding.

    Constructs a feature map consisting of multiple layers of Pauli feature map layers.
    Each layer applies a set of parameterized Pauli blocks to encode classical data into
    a quantum state.

    Args:
        in_features (int): Number of qubits (input features).
        num_layers (int): Number of layers.
        paulis (List[str]): List of Pauli strings that define the blocks.
        entanglement (Union[str, List[List[int]]], optional): Entanglement strategy.
            Defaults to "full".
        alpha (float, optional): Scaling factor for the parameterized rotations.
            Defaults to 2.0.
    """

    def __init__(
        self,
        in_features: int,
        num_layers: int,
        paulis: List[str],
        entanglement: Union[str, List[List[int]]] = "full",
        alpha: float = 2.0,
    ) -> None:
        max_pauli_len: int = max(len(p) for p in paulis)
        if in_features < max_pauli_len:
            raise ValueError(
                f"in_features must be at least equal to the maximum pauli length. Got in_features={in_features} "
                f"and maximum pauli length={max_pauli_len}."
            )
        self.paulis: List[str] = paulis
        self.entanglement: Union[str, List[List[int]]] = entanglement
        self.alpha: float = alpha
        super().__init__(in_features, num_layers)

    def build_kernel(self) -> cudaq.Kernel:
        r"""Build the quantum circuit kernel for the Pauli feature map.

        Returns:
            cudaq.Kernel: The constructed feature map circuit kernel.
        """
        kernel, qvec, x = cudaq.make_kernel(cudaq.qvector, List[float])
        for layer in range(self.num_layers):
            pfm_layer: PauliFeatureMapLayer = PauliFeatureMapLayer(
                self.num_qubits, self.paulis, self.entanglement, self.alpha
            )
            self.layers[f"layer_{layer}"] = pfm_layer
            kernel.apply_call(pfm_layer.kernel, qvec, x)
        return kernel

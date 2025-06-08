import cudaq
import torchq.quantum as qu
from typing import Union, List


class RotationBlock(qu.Ansatz):
    r"""Rotation block ansatz for single-qubit rotations.

    Applies a parameterized rotation gate on each qubit. Supported rotation gates
    are ``"rx"``, ``"ry"`` and ``"rz"``. If a single gate is provided as a string,
    the same gate is applied on all qubits; otherwise a list of gates must match the
    number of qubits.

    Circuit Diagram Example (for 2 qubits with "ry" rotations):

    Args:
        num_qubits (int): Number of qubits.
        gates (Union[str, List[str]]): Rotation gate(s) to apply. If a string is provided,
            the same gate is used for all qubits. Otherwise, the list length must equal
            ``num_qubits``.

    Raises:
        ValueError: If the number of provided gates does not match ``num_qubits``.
    """

    def __init__(self, num_qubits: int, gates: Union[str, List[str]]) -> None:
        self.num_qubits: int = num_qubits
        if isinstance(gates, str):
            self.gates: List[str] = [gates.lower()] * num_qubits
        else:
            if len(gates) != num_qubits:
                raise ValueError(
                    f"num_qubits must equal the number of gates. Got num_qubits={num_qubits} and "
                    f"gates length={len(gates)}."
                )
            self.gates = [g.lower() for g in gates]
        super().__init__(num_qubits)

    def build_kernel(self) -> cudaq.Kernel:
        r"""Build the quantum circuit kernel for the rotation block.

        Returns:
            cudaq.Kernel: The kernel that applies the designated rotation gate on each qubit.
        """
        # Create a kernel with one float parameter per rotation.
        kernel, qvec, thetas = cudaq.make_kernel(
            cudaq.qvector, List[float]
        )
        for i in range(self.num_qubits):
            gate = self.gates[i]
            if gate == "rx":
                kernel.rx(thetas[i], qvec[i])
            elif gate == "ry":
                kernel.ry(thetas[i], qvec[i])
            elif gate == "rz":
                kernel.rz(thetas[i], qvec[i])
            else:
                raise ValueError(f"Unsupported rotation gate: {gate}")
        return kernel

    def get_parameter_count(self) -> int:
        r"""Return the number of parameters for the rotation block.

        Returns:
            int: Equal to the number of qubits.
        """
        return self.num_qubits


class EntanglementBlock(qu.Ansatz):
    r"""Entanglement block ansatz for multi-qubit entanglement.

    Constructs a circuit applying an entangling gate (e.g. CNOT or CZ) between each
    pair of qubits defined by the ``pairs`` argument.

    Circuit Diagram Example (using CNOT on qubits 0 and 1):

    Args:
        pairs (List[List[int]]): List of pairs of qubit indices to entangle.
        gate (str): Entanglement gate to use (e.g. ``"cx"`` or ``"cz"``).
    """

    def __init__(self, pairs: List[List[int]], gate: str) -> None:
        num_qubits: int = max(max(pair) for pair in pairs) + 1 if pairs else 0
        self.gate: str = gate.lower()
        self.pairs: List[List[int]] = pairs
        super().__init__(num_qubits)

    def build_kernel(self) -> cudaq.Kernel:
        r"""Build the quantum circuit kernel for the entanglement block.

        Returns:
            cudaq.Kernel: The kernel that applies the designated entangling gate
            between the specified qubit pairs.
        """
        kernel, qvec = cudaq.make_kernel(cudaq.qvector)
        for pair in self.pairs:
            i, j = pair
            if self.gate in ["cx", "cnot"]:
                kernel.cx(qvec[i], qvec[j])
            elif self.gate in ["cz", "cphase"]:
                kernel.cz(qvec[i], qvec[j])
            else:
                raise ValueError(f"Unsupported entanglement gate: {self.gate}")
        return kernel

    def get_parameter_count(self) -> int:
        r"""Return the number of parameters for the entanglement block.

        Entanglement blocks are not parameterized.

        Returns:
            int: Always 0.
        """
        return 0

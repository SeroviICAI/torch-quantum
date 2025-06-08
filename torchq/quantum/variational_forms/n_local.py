import cudaq
import torchq.quantum as qu
import torchq.quantum.variational_forms as qvf
from typing import Union, List
from collections import OrderedDict


class NLocalLayer(qu.Ansatz):
    r"""N-local layer ansatz combining rotation and entanglement blocks.

    This layer applies one or more rotation blocks followed by entanglement blocks,
    and optionally an additional final rotation block. It is intended as a building
    block for variational circuits.

    Circuit Diagram Example (for 3 qubits, two rotation blocks then full entanglement):

    .. code-block:: text

         ╭─────────╮╭─────────╮
    q0 : ┤ ry(θ_1) ├┤ rz(θ_4) ├──●────●───────
         ├─────────┤├─────────┤╭─┴─╮  │
    q1 : ┤ ry(θ_2) ├┤ rz(θ_5) ├┤ x ├──┼────●──
         ├─────────┤├─────────┤╰───╯╭─┴─╮╭─┴─╮
    q2 : ┤ ry(θ_3) ├┤ rz(θ_6) ├─────┤ x ├┤ x ├
         ╰─────────╯╰─────────╯     ╰───╯╰───╯

    Args:
        num_qubits (int): Number of qubits.
        rotation_blocks (Union[str, List[str], List[List[str]]]): Rotation gate(s) specification.
            Either a single gate (applied on all qubits) or a list (or list of lists) specifying
            gates per block.
        entanglement_blocks (Union[str, List[str]]): Entanglement gate(s) specification.
            Either a single gate or a list of gates.
        entanglement (Union[str, List[List[int]]], optional): Entanglement strategy.
            Supported strategies include ``"full"``, ``"linear"``, or ``"reverse_linear"``
            (or custom pairs). Defaults to ``"full"``.
        apply_final_rotation (bool, optional): Whether to apply an additional rotation block
            after the entanglement. Defaults to False.
    """

    def __init__(
        self,
        num_qubits: int,
        rotation_blocks: Union[str, List[str], List[List[str]]],
        entanglement_blocks: Union[str, List[str]],
        entanglement: Union[str, List[List[int]]] = "full",
        apply_final_rotation: bool = False,
    ) -> None:
        self.num_qubits: int = num_qubits
        if isinstance(rotation_blocks, str):
            self.rotation_blocks: Union[List[str], List[List[str]]] = [rotation_blocks]
        else:
            self.rotation_blocks = rotation_blocks

        if isinstance(entanglement_blocks, str):
            self.entanglement_blocks: List[str] = [entanglement_blocks]
        else:
            self.entanglement_blocks = entanglement_blocks

        self.entanglement: Union[str, List[List[int]]] = entanglement
        self.apply_final_rotation: bool = apply_final_rotation
        self.blocks: OrderedDict[str, qu.Ansatz] = OrderedDict()
        super().__init__(num_qubits)

    @staticmethod
    def get_entangler_pairs(
        num_qubits: int, entanglement: Union[str, List[List[int]]]
    ) -> List[List[int]]:
        r"""Generate qubit pairs for entanglement based on a specified strategy.

        Args:
            num_qubits (int): Total number of qubits.
            entanglement (Union[str, List[List[int]]]): Either a string specifying a known strategy
                (e.g. ``"full"``, ``"linear"``, ``"reverse_linear"``) or a custom list of pairs.

        Returns:
            List[List[int]]: List of qubit index pairs to be entangled.
        """
        if isinstance(entanglement, str):
            entanglement_lower: str = entanglement.lower()
            if entanglement_lower == "full":
                pairs = []
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        pairs.append([i, j])
                return pairs
            elif entanglement_lower == "linear":
                return [[i, i + 1] for i in range(num_qubits - 1)]
            elif entanglement_lower == "reverse_linear":
                return [[i - 1, i] for i in range(num_qubits - 1, 0, -1)]
        if isinstance(entanglement, list):
            if isinstance(entanglement[0], int):
                return [entanglement]
            return entanglement
        return []

    def build_kernel(self) -> cudaq.Kernel:
        r"""Build the quantum circuit kernel for the N-local layer.

        Sequentially applies rotation blocks, entanglement blocks, and, if specified,
        a final rotation block.

        Returns:
            cudaq.Kernel: The constructed circuit kernel.
        """
        # Create a kernel with one float parameter per rotation.
        kernel, qvec, thetas = cudaq.make_kernel(
            cudaq.qvector, List[float]
        )

        ptr: int = 0
        for idx in range(len(self.rotation_blocks)):
            rb: qu.RotationBlock = qu.RotationBlock(
                self.num_qubits, self.rotation_blocks[idx]
            )
            self.blocks[f"rotation_{idx}"] = rb
            count: int = rb.get_parameter_count()
            kernel.apply_call(rb.kernel, qvec, thetas[ptr : ptr + count])
            ptr += count

        ent_pairs: List[List[int]] = (
            NLocalLayer.get_entangler_pairs(self.num_qubits, self.entanglement)
            if self.entanglement is not None
            else []
        )
        for idx, gate in enumerate(self.entanglement_blocks):
            eb: qu.EntanglementBlock = qu.EntanglementBlock(ent_pairs, gate)
            self.blocks[f"entanglement_{idx}"] = eb
            kernel.apply_call(eb.kernel, qvec)

        if self.apply_final_rotation:
            for idx in range(len(self.rotation_blocks)):
                frb: qu.RotationBlock = qu.RotationBlock(
                    self.num_qubits, self.rotation_blocks[idx]
                )
                self.blocks[f"final_rotation_{idx}"] = frb
                count = frb.get_parameter_count()
                kernel.apply_call(frb.kernel, qvec, thetas[ptr : ptr + count])
                ptr += count
        return kernel

    def get_parameter_count(self) -> int:
        r"""Return the total number of parameters in the N-local layer.

        Returns:
            int: Number of parameters from the rotation blocks (and optionally final rotations).
        """
        count: int = len(self.rotation_blocks) * self.num_qubits
        if self.apply_final_rotation:
            count += len(self.rotation_blocks) * self.num_qubits
        return count


class NLocal(qvf.VariationalForm):
    r"""N-local variational form ansatz.

    Constructs an N-local variational circuit made of multiple N-local layers. Each layer
    consists of rotation and entanglement blocks. Optionally, the final layer may include an extra
    rotation stage unless skipped.

    Args:
        num_qubits (int): Number of qubits.
        num_layers (int): Number of variational layers.
        rotation_blocks (Union[str, List[str], List[List[str]]]): Rotation gate(s) specification.
        entanglement_blocks (Union[str, List[str]]): Entanglement gate(s) specification.
        entanglement (Union[str, List[List[int]]], optional): Entanglement strategy.
            Defaults to "full".
        skip_final_rotation (bool, optional): If True, the final rotation layer is skipped.
            Defaults to False.
    """

    def __init__(
        self,
        num_qubits: int,
        num_layers: int,
        rotation_blocks: Union[str, List[str], List[List[str]]],
        entanglement_blocks: Union[str, List[str]],
        entanglement: Union[str, List[List[int]]] = "full",
        skip_final_rotation: bool = False,
    ) -> None:
        if isinstance(rotation_blocks, str):
            self.rotation_blocks: List[str] = [rotation_blocks]
        else:
            self.rotation_blocks = rotation_blocks

        if isinstance(entanglement_blocks, str):
            self.entanglement_blocks: List[str] = [entanglement_blocks]
        else:
            self.entanglement_blocks = entanglement_blocks

        self.entanglement: Union[str, List[List[int]]] = entanglement
        self.skip_final_rotation: bool = skip_final_rotation
        super().__init__(num_qubits, num_layers)

    def build_kernel(self) -> cudaq.Kernel:
        r"""Build the quantum circuit kernel for the N-local variational form.

        Returns:
            cudaq.Kernel: The constructed variational circuit kernel.
        """
        kernel, qvec, thetas = cudaq.make_kernel(
            cudaq.qvector, List[float]
        )
        ptr: int = 0
        for layer in range(self.num_layers):
            apply_final: bool = (layer == self.num_layers - 1) and (
                not self.skip_final_rotation
            )
            nl_layer: NLocalLayer = NLocalLayer(
                self.num_qubits,
                self.rotation_blocks,
                self.entanglement_blocks,
                self.entanglement,
                apply_final_rotation=apply_final,
            )
            self.layers[f"layer_{layer}"] = nl_layer
            count: int = nl_layer.get_parameter_count()
            kernel.apply_call(nl_layer.kernel, qvec, thetas[ptr : ptr + count])
            ptr += count
        return kernel

    def get_parameter_count(self) -> int:
        r"""Return the total number of parameters in the N-local variational form.

        Returns:
            int: Total parameter count across all layers.
        """
        total: int
        if self.skip_final_rotation:
            total = self.num_layers * len(self.rotation_blocks) * self.num_qubits
        else:
            total = (self.num_layers - 1) * len(
                self.rotation_blocks
            ) * self.num_qubits + 2 * len(self.rotation_blocks) * self.num_qubits
        return total

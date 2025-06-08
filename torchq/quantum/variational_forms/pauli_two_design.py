import cudaq
import math
import numpy as np
import torchq.quantum as qu
import torchq.quantum.variational_forms as qvf
from typing import List, Union
from collections import OrderedDict


class PauliTwoDesignLayer(qu.Ansatz):
    def __init__(
        self,
        num_qubits: int,
        seed: Union[int, None] = 2025,
        apply_final_rotation: bool = False,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.entanglement_blocks = "cz"
        self.apply_final_rotation = apply_final_rotation
        self.blocks: OrderedDict[qu.Ansatz] = OrderedDict()
        super().__init__(num_qubits)

    @staticmethod
    def get_entangler_pairs(num_qubits: int) -> List[List[int]]:
        entanglement: List[List[int]] = []
        for i in range(0, num_qubits - 1, 2):
            entanglement.append([i, i + 1])
        for i in range(1, num_qubits - 1, 2):
            entanglement.append([i, i + 1])
        return entanglement

    def build_kernel(self) -> cudaq.Kernel:
        kernel, qvec, thetas = cudaq.make_kernel(
            cudaq.qvector, List[float]
        )

        ptr = 0
        rotation_block = self.rng.choice(["rx", "ry", "rz"], self.num_qubits).tolist()
        rb = qu.RotationBlock(self.num_qubits, rotation_block)
        self.blocks[f"rotation_0"] = rb
        count = rb.get_parameter_count()
        kernel.apply_call(rb.kernel, qvec, thetas[: ptr + count])
        ptr += count

        ent_pairs = PauliTwoDesignLayer.get_entangler_pairs(self.num_qubits)
        eb = qu.EntanglementBlock(ent_pairs, self.entanglement_blocks)
        self.blocks[f"entanglement_0"] = eb
        kernel.apply_call(eb.kernel, qvec)

        if self.apply_final_rotation:
            final_rotation_block = self.rng.choice(
                ["rx", "ry", "rz"], self.num_qubits
            ).tolist()
            frb = qu.RotationBlock(self.num_qubits, final_rotation_block)
            self.blocks[f"final_rotation_0"] = frb
            count = frb.get_parameter_count()
            kernel.apply_call(frb.kernel, qvec, thetas[ptr:])
            ptr += count
        return kernel

    def get_parameter_count(self) -> int:
        return 2 * self.num_qubits if self.apply_final_rotation else self.num_qubits


class PauliTwoDesign(qvf.VariationalForm):
    def __init__(
        self, num_qubits: int, num_layers: int = 3, seed: Union[int, None] = None
    ) -> None:
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.seed = seed

        super().__init__(num_qubits, num_layers)

    def build_kernel(self) -> cudaq.Kernel:
        kernel, qvec, thetas = cudaq.make_kernel(
            cudaq.qvector, List[float]
        )

        for q in range(self.num_qubits):
            kernel.ry(math.pi / 4, qvec[q])

        ptr = 0

        for layer in range(self.num_layers):
            layer_seed = self.seed + layer if self.seed is not None else None
            apply_final = layer == self.num_layers - 1
            ptd_layer = PauliTwoDesignLayer(
                self.num_qubits, seed=layer_seed, apply_final_rotation=apply_final
            )
            count = ptd_layer.get_parameter_count()
            kernel.apply_call(ptd_layer.kernel, qvec, thetas[ptr : ptr + count])
            ptr += count
        return kernel

    def get_parameter_count(self) -> int:
        return (self.num_layers - 1) * self.num_qubits + 2 * self.num_qubits

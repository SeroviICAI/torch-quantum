import torchq.quantum.variational_forms as qvf
from typing import Union, List


class EfficientSU2(qvf.NLocal):
    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 3,
        su2_gates: Union[str, List[str]] = None,
        entanglement: Union[str, List[List[int]]] = "full",  # "reverse_linear",
        skip_final_rotation: bool = False,
    ) -> None:
        if su2_gates is None:
            su2_gates = ["ry", "rz"]

        entanglement_blocks = "cx" if num_qubits > 1 else ""
        super().__init__(
            num_qubits=num_qubits,
            num_layers=num_layers,
            rotation_blocks=su2_gates,
            entanglement_blocks=entanglement_blocks,
            entanglement=entanglement,
            skip_final_rotation=skip_final_rotation,
        )

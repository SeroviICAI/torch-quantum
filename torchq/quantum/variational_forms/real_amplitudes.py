import torchq.quantum.variational_forms as qvf
from typing import Union, List


class RealAmplitudes(qvf.NLocal):
    def __init__(
        self,
        num_qubits: int,
        num_layers: int = 3,
        entanglement: Union[str, List[List[int]]] = "full",
        skip_final_rotation: bool = False,
    ) -> None:
        rotation_blocks = "ry"
        entanglement_blocks = "cx" if num_qubits > 1 else ""
        super().__init__(
            num_qubits=num_qubits,
            num_layers=num_layers,
            rotation_blocks=rotation_blocks,
            entanglement_blocks=entanglement_blocks,
            entanglement=entanglement,
            skip_final_rotation=skip_final_rotation,
        )

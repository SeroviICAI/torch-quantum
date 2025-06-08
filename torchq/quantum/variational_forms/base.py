import torchq.quantum as qu
from collections import OrderedDict


class VariationalForm(qu.Ansatz):
    r"""Base class for variational form ansatz circuits.

    This class serves as a container for variational circuits that typically consist of
    multiple layers of parameterized quantum operations.

    Attributes:
        num_layers (int): Number of variational layers.
        layers (OrderedDict[qu.Ansatz]): Ordered dictionary holding each layer's ansatz.
    """

    def __init__(self, num_qubits: int, num_layers: int = 1) -> None:
        r"""Initialize the variational form.

        Args:
            num_qubits (int): Number of qubits for the circuit.
            num_layers (int, optional): Number of layers. Defaults to 1.
        """
        self.num_layers: int = num_layers
        self.layers: OrderedDict[qu.Ansatz] = OrderedDict()
        super().__init__(num_qubits)

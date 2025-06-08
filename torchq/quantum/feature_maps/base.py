import torchq.quantum as qu
from collections import OrderedDict


class FeatureMap(qu.Ansatz):
    r"""Base class for quantum feature maps.

    Encodes classical data into quantum states using parameterized circuits.
    This class defines the basic structure for feature maps.

    Attributes:
        num_layers (int): Number of layers in the feature map.
        layers (OrderedDict[qu.Ansatz]): Ordered dictionary of feature map layers.
    """

    def __init__(self, num_qubits: int, num_layers: int = 1) -> None:
        r"""Initialize the feature map.

        Args:
            num_qubits (int): Number of qubits.
            num_layers (int, optional): Number of layers. Defaults to 1.
        """
        self.num_layers: int = num_layers
        self.layers: OrderedDict[str, qu.Ansatz] = OrderedDict()
        super().__init__(num_qubits)
    
    def get_parameter_count(self) -> int:
        r"""Return the total number of parameters in the feature map.

        Returns:
            int: Total parameter count across all layers.
        """
        return self.num_qubits

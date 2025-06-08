import cudaq


class Ansatz:
    r"""Base class for quantum circuit ansatzes.

    This class defines the common interface for quantum ansatz circuits in torchquantum.
    Subclasses must override the `build_kernel()` and `get_parameter_count()` methods.

    Attributes:
        num_qubits (int): Number of qubits the ansatz operates on.
        _kernel (cudaq.Kernel): The quantum circuit (kernel) built by `build_kernel()`.

    .. note::
        This class should be subclassed. Child classes must implement the
        :meth:`build_kernel` and :meth:`get_parameter_count` methods.
    """

    def __init__(self, num_qubits: int) -> None:
        r"""Initialize the ansatz.

        Args:
            num_qubits (int): Number of qubits for the circuit.
        """
        self.num_qubits: int = num_qubits
        self._kernel: cudaq.Kernel = self.build_kernel()

    @property
    def kernel(self):
        r"""Return the quantum circuit kernel.

        Returns:
            cudaq.Kernel: The circuit kernel.
        """
        return self._kernel

    def build_kernel(self) -> cudaq.Kernel:
        r"""Construct the quantum circuit kernel.

        Subclasses must override this method.

        Raises:
            NotImplementedError: Always, unless overridden.

        Returns:
            cudaq.Kernel: The constructed quantum circuit.
        """
        raise NotImplementedError("Subclasses must implement build_kernel().")

    def get_parameter_count(self) -> int:
        r"""Return the total number of tunable parameters in the ansatz.

        Subclasses must override this method.

        Raises:
            NotImplementedError: Always, unless overridden.

        Returns:
            int: The number of parameters.
        """
        raise NotImplementedError("Subclasses must implement get_parameter_count().")

import torchq.quantum.feature_maps as qfm


class ZFeatureMap(qfm.PauliFeatureMap):
    def __init__(
        self,
        in_features: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__(
            in_features=in_features,
            num_layers=num_layers,
            paulis=["Z"],
            entanglement="full",
            alpha=2.0,
        )

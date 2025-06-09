TRAINING_PARAMS = {
    "batch_size": 8,
    "num_workers": 3,
    "learning_rate": 0.1,
    "epochs": 20,
    "seed": 42,
}

QNN_PARAMS = {
    "in_features": 4,
    "out_features": 3,
    "num_layers": 9,
    "shots": 2048,
    "feature_map": "z",
    "var_form": "efficientsu2",
    "reupload": False,
}

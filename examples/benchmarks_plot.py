import os
import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    """
    Generate and save the following figures:
      1. Training comparison (accuracy & loss) for classical vs quantum on Iris.
      2. Quantum NN training (accuracy & loss) across Iris, Wine, and Digits.
    """
    # 1 & 2: Load Iris data for classical and quantum models
    base_dirs = {
        "quantum":   "results/data/benchmarks/quantum_neural_network",
        "classical": "results/data/benchmarks/classical_neural_network"
    }
    prefixes = {
        "quantum":   "quantum_neural_network",
        "classical": "multilayer_perceptron"
    }
    legend_names = {
        "quantum":   "Quantum Neural Network",
        "classical": "Classical Neural Network"
    }
    splits = ["train", "val"]
    metrics = ["acc", "loss"]

    data = {m: {} for m in base_dirs}
    for m in base_dirs:
        for split in splits:
            for metric in metrics:
                key = f"{split}_{metric}"
                path = os.path.join(
                    base_dirs[m],
                    split,
                    f"{prefixes[m]}_iris_{split}_{metric}.csv",
                )
                df = pd.read_csv(path)
                data[m][key] = df[["Step", "Value"]]

    # 1. Training comparison
    print("Generating Figure 1 (Iris training comparison)...")
    fig1, ax1 = plt.subplots(1, 2, figsize=(8, 4), dpi=1000)
    # Accuracy
    for m, color in zip(base_dirs, ("tab:blue", "tab:red")):
        ax1[0].plot(
            data[m]["train_acc"]["Step"],
            data[m]["train_acc"]["Value"],
            label=legend_names[m],
            color=color,
        )
    ax1[0].set_title("Training Accuracy")
    ax1[0].set_xlabel("Step")
    ax1[0].set_ylabel("Accuracy")
    ax1[0].grid(True)
    ax1[0].legend(loc="lower right")
    # Loss
    for m, color in zip(base_dirs, ("tab:blue", "tab:red")):
        ax1[1].plot(
            data[m]["train_loss"]["Step"],
            data[m]["train_loss"]["Value"],
            label=legend_names[m],
            color=color,
        )
    ax1[1].set_title("Training Loss")
    ax1[1].set_xlabel("Step")
    ax1[1].set_ylabel("Loss")
    ax1[1].grid(True)
    ax1[1].legend(loc="lower right")
    fig1.tight_layout()
    fig1.savefig("results/images/benchmarks/iris_training_comparison.png")
    plt.close(fig1)

    # 2. Quantum NN training across multiple datasets
    print("Generating Figure 2 (Quantum multi-dataset training)...")
    datasets = ["iris", "wine", "digits"]
    q_data = {}
    for ds in datasets:
        path_acc = os.path.join(
            base_dirs["quantum"],
            "train",
            f"{prefixes['quantum']}_{ds}_train_acc.csv",
        )
        path_loss = os.path.join(
            base_dirs["quantum"],
            "train",
            f"{prefixes['quantum']}_{ds}_train_loss.csv",
        )
        df_acc = pd.read_csv(path_acc)
        df_loss = pd.read_csv(path_loss)
        q_data[ds] = {
            "acc": df_acc[["Step", "Value"]],
            "loss": df_loss[["Step", "Value"]],
        }

    fig3, ax3 = plt.subplots(1, 2, figsize=(8, 4), dpi=1000)
    # Accuracy
    for ds, color in zip(datasets, ("tab:blue", "tab:green", "tab:orange")):
        ax3[0].plot(
            q_data[ds]["acc"]["Step"],
            q_data[ds]["acc"]["Value"],
            label=ds.capitalize(),
            color=color,
        )
    ax3[0].set_title("Quantum NN Training Accuracy")
    ax3[0].set_xlabel("Step")
    ax3[0].set_ylabel("Accuracy")
    ax3[0].grid(True)
    ax3[0].legend(loc="lower right")
    # Loss
    for ds, color in zip(datasets, ("tab:blue", "tab:green", "tab:orange")):
        ax3[1].plot(
            q_data[ds]["loss"]["Step"],
            q_data[ds]["loss"]["Value"],
            label=ds.capitalize(),
            color=color,
        )
    ax3[1].set_title("Quantum NN Training Loss")
    ax3[1].set_xlabel("Step")
    ax3[1].set_ylabel("Loss")
    ax3[1].grid(True)
    ax3[1].legend(loc="lower right")
    fig3.tight_layout()
    fig3.savefig("results/images/benchmarks/quantum_multi_dataset_training.png")
    plt.close(fig3)


if __name__ == "__main__":
    main()

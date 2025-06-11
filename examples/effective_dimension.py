import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cudaq

# Import modules from the TorchQuantum repository
from torchq.models import QNN
from examples.classic_models import ParamMatchedFCN
from examples.utils import count_parameters, set_seed

# Import Fisher information functions
from torchq.func.fisher import fisher, fisher_norm, eff_dim


def plot_hist_with_inset(panel, values, colour, title):
    vals_np = values.numpy()
    w = np.ones_like(vals_np) / len(vals_np)  # normalised weights
    counts, bins, _ = panel.hist(
        vals_np, bins=6, weights=w, color=colour, edgecolor="k"
    )
    panel.set(title=title, xlabel="eigenvalue size")
    if "classical" in title:
        panel.set_ylabel("normalised counts")
    panel.set_ylim(0, 1)

    # Inset showing distribution of first-bin values
    first_bin_mask = (vals_np >= bins[0]) & (vals_np < bins[1])
    inset = panel.inset_axes([0.55, 0.55, 0.4, 0.3])
    inset.hist(
        vals_np[first_bin_mask],
        bins=6,
        weights=np.ones_like(vals_np[first_bin_mask])
        / max(1, len(vals_np[first_bin_mask])),
        color=colour,
        edgecolor="none",
    )
    inset.set_xlabel("eig. size", fontsize=7)
    inset.set_ylabel("norm. counts", fontsize=7)
    inset.tick_params(axis="both", labelsize=6)
    inset.set_title("first bin", fontsize=8)
    inset.set_ylim(0, 1)


def main() -> None:
    r"""Generate plots reproducing key results from 'The power of quantum neural networks'.

    This script creates and saves the following figures:
      1. Figure 2 - Fisher information spectrum histograms for a classical FCN and a QNN ($s_{in}=4$, $s_{out}=3$).
      2. Figure 3 - Normalised effective dimension vs number of data points (n) for a FCN and QNN.

    All experiments use only the QNN and ParamMatchedFCN models (no hybrid or "easy" quantum models).
    QNN parameters follow the configuration from train.py: in_features=$s_{in}$, out_features=$s_{out}$, num_layers=6
    (unless otherwise noted for specific experiments), shots=2048, feature_map="z", var_form="realamplitudes", reupload=False.
    """
    # Ensure reproducibility
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set quantum simulation backend based on device
    if device.type == "cuda":
        cudaq.set_target("nvidia", option="fp32")
    else:
        cudaq.set_target("qpp-cpu")

    # 1. Figure 2: Fisher information eigenvalue spectrum histograms for FCN vs QNN (d≈28, s_in=4, s_out=3).
    print("Generating Figure 2 (Fisher information spectrum)...")
    s_in = 4
    s_out = 3
    n_layers = 6
    # Initialize QNN and classical model with parameter count 28
    # Use QNN with default parameters (num_layers=3) and count its parameters
    model_qnn = QNN(
        in_features=s_in,
        out_features=s_out,
        num_layers=n_layers,
        shots=2048,
        feature_map="z",
        var_form="realamplitudes",
        reupload=False,
    ).to(device)
    d_qnn = count_parameters(model_qnn)
    # Create a ParamMatchedFCN with at least d_qnn parameters
    model_fcn = ParamMatchedFCN(in_features=s_in, out_features=s_out, d=d_qnn).to(
        device
    )
    d_fcn = count_parameters(model_fcn)
    print(f"QNN param count: {d_qnn}, FCN param count: {d_fcn}")

    # Compute eigenvalues of Fisher information for 100 random parameter instances of each model
    num_param_samples = 100
    eigvals_qnn, eigvals_fcn = [], []
    fhat_qnn_list, fhat_fcn_list = [], []

    # Use the same number of data samples for Fisher computation (e.g., 100) for each random draw
    num_data_samples = 100
    # Sample inputs from a standard normal distribution
    for _ in tqdm(range(num_param_samples), desc="FCN Fisher samples"):
        model_c = ParamMatchedFCN(in_features=s_in, out_features=s_out, d=d_qnn).to(
            device
        )
        X_rand = torch.randn(num_data_samples, s_in)
        fhat_c, _ = fisher_norm(fisher(model_c, X_rand))
        fhat_fcn_list.append(fhat_c[0])
        eigs = torch.linalg.eigvalsh(fhat_c[0]).cpu().numpy()
        eigvals_fcn.extend(eigs)

    for _ in tqdm(range(num_param_samples), desc="QNN Fisher samples"):
        # Randomize QNN parameters by reinitializing a new model (to ensure independent random draws)
        model_q = QNN(
            in_features=s_in,
            out_features=s_out,
            num_layers=n_layers,
            shots=2048,
            feature_map="z",
            var_form="realamplitudes",
            reupload=False,
        ).to(device)
        # Generate random input batch
        X_rand = torch.randn(num_data_samples, s_in)
        # Compute Fisher information matrix for this parameter set
        fhat_q, _ = fisher_norm(fisher(model_q, X_rand))
        fhat_qnn_list.append(fhat_q[0])
        # Eigenvalues of Fisher (use real part since F is symmetric PSD)
        eigs = torch.linalg.eigvalsh(fhat_q[0]).cpu().numpy()
        eigvals_qnn.extend(eigs)

    # Store raw data
    eigvals_fcn, eigvals_qnn= map(
        torch.tensor, (eigvals_fcn, eigvals_qnn)
    )
    fhat_fcn_batch, fhat_qnn_batch = map(
        torch.stack, (fhat_fcn_list, fhat_qnn_list)
    )
    torch.save(eigvals_fcn, "results/data/information_geometry/eigvals_fcn.pt")
    torch.save(eigvals_qnn, "results/data/information_geometry/eigvals_qnn.pt")
    torch.save(fhat_fcn_list, "results/data/information_geometry/fhat_fcn.pt")
    torch.save(fhat_qnn_list, "results/data/information_geometry/fhat_qnn.pt")

    # Plot histograms of eigenvalue spectra
    fig2, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=1000)
    plot_hist_with_inset(ax[0], eigvals_fcn, "tab:red", "classical neural network")
    plot_hist_with_inset(ax[1], eigvals_qnn, "tab:blue", "quantum neural network")
    fig2.tight_layout()
    plt.savefig("results/images/fisher_information_spectrum.png")
    plt.close(fig2)
    print("Figure 2 saved in to results/images/fisher_information_spectrum.png")

    # 2. Figure 3: Normalised effective dimension vs number of data points for QNN vs FCN (d≈28, s_in=4, s_out=2).
    print("Generating Figure 3 (Normalised effective dimension vs number of data)...")

    # Define range of number of data points (n) to evaluate effective dimension
    n_values = np.array(
        [1000, 2000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000]
    )

    # Compute effective dimension for each n (returns tensor of shape [len(n_values)])
    eff_qnn = eff_dim(fhat_qnn_batch, n_values.tolist()).cpu().numpy()
    eff_fcn = eff_dim(fhat_fcn_batch, n_values.tolist()).cpu().numpy()

    # Normalize effective dimension by model parameter count d (to yield d_eff/d)
    eff_qnn_norm = eff_qnn / d_qnn
    eff_fcn_norm = eff_fcn / d_fcn

    # Plot the effective dimension vs n curves
    fig3, ax3 = plt.subplots(figsize=(6, 4), dpi=1000)
    ax3.plot(n_values, eff_qnn_norm, color="tab:blue", label="quantum neural network")
    ax3.plot(n_values, eff_fcn_norm, color="tab:red", label="classical neural network")
    ax3.set_xlabel("number of data")
    ax3.set_ylabel("normalised effective dimension $d_{\\text{eff}}/d$")
    ax3.legend(loc="lower right")
    ax3.grid(True)
    ax3.set_xlim([0, n_values.max()])
    ax3.set_ylim([0, 1])
    fig3.tight_layout()
    plt.savefig("results/images/information_geometry/normalised_effective_dimension.png")
    plt.close(fig3)
    print("Figure 3 saved to results/images/information_geometry/normalised_effective_dimension.png")


if __name__ == "__main__":
    main()

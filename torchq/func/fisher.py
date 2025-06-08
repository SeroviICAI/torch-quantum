import math
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.func import functional_call
from typing import Sequence, Tuple

__all__ = ["fisher", "fisher_norm", "eff_dim"]


def _make_flat_to_param_dict(model: nn.Module):
    """Return a converter from *flat θ* → {param_name: tensor‐view}."""
    slices, offset = [], 0
    for p in model.parameters():
        cnt = p.numel()
        slices.append((offset, p.shape))
        offset += cnt

    def flat_to_param_dict(flat: torch.Tensor):
        pd = {}
        for (n, _), (off, shape) in zip(model.named_parameters(), slices):
            length = int(torch.tensor(shape).prod().item())
            pd[n] = flat[off : off + length].view(shape)
        return pd

    return flat_to_param_dict


def _probabilities(out: torch.Tensor) -> torch.Tensor:
    """
    Ensure *out* is a proper probability vector.

    * If the model already returns non-negative rows that sum to 1 – do nothing.
    * Otherwise apply soft-max.
    """
    if torch.all(out >= 0) and torch.allclose(out.sum(-1), out.new_ones(out.size(0))):
        return out
    return torch.softmax(out, dim=-1)


def fisher(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    num_thetas: int = 1,
    param_range: Tuple[float, float] = (-1.0, 1.0),
    damping: float = 0.0,
) -> torch.Tensor:
    r"""
    Approximate the Fisher information **E₍x,θ₎[F(x;θ)]** (Empirical Fisher Information matrix).

    • Draw *num_thetas* independent parameter vectors θ ~ U[param_range].
    • For each θ and each input x, compute

        F₍x,θ₎ = ∑ₖ gₖ gₖᵀ with gₖ = √pₖ(x;θ) ∇θ log pₖ(x;θ).

      (In code we obtain gₖ by back-prop on ``log pₖ`` and then scale by √pₖ.)
    • Average F₍x,θ₎ over the *inputs* for an empirical Fisher, keep **one** matrix
      per θ.  The resulting tensor has shape ``(num_thetas, D, D)``.

    Args
        model: Any ``nn.Module`` mapping ``(B, …) → (B, C)`` with *probabilities* or logits.
        inputs: Tensor shaped ``(B, in_features)``.
        num_thetas: How many random θ's to average over (≥1).
        param_range: (low, high) for uniform sampling of every parameter.
        damping: Optional λI Tikhonov term for numerical stability.

    Returns
        fishers: Tensor ``(num_thetas, D, D)`` - one Fisher matrix per sampled θ.
    """
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    B = inputs.size(0)

    # util to convert flat vec → model-param dict
    to_dict = _make_flat_to_param_dict(model)
    D = sum(p.numel() for p in model.parameters())

    fishers = torch.zeros(num_thetas, D, D, device=device)

    low, high = param_range

    with torch.no_grad():
        base_params = parameters_to_vector(model.parameters()).detach()

    for t in range(num_thetas):
        theta = torch.empty_like(base_params).uniform_(low, high).requires_grad_(True)

        sample_fisher = torch.zeros(D, D, device=device)

        for i in range(B):
            x = inputs[i : i + 1]  # keep batch dim =1
            # forward with *functional_call* so we do not mutate the real params
            out = functional_call(model, to_dict(theta), (x,))
            p = _probabilities(out).squeeze(0)  # (C,)
            log_p = torch.log(p + 1e-40)  # avoid log(0)

            # accumulate ∑ₖ gₖ gₖᵀ
            for k in range(p.numel()):
                model.zero_grad(set_to_none=True)
                (grad_flat,) = torch.autograd.grad(
                    log_p[k], theta, retain_graph=True, create_graph=False
                )
                gk = grad_flat * torch.sqrt(p[k])
                sample_fisher += torch.outer(gk, gk)

        sample_fisher /= float(B)  # empirical over inputs
        if damping > 0.0:
            sample_fisher += damping * torch.eye(D, device=device)
        fishers[t] = sample_fisher.detach()

    return fishers


def fisher_norm(
    fisher: torch.Tensor,
) -> Tuple[torch.Tensor, float]:
    r"""
    Normalize the Fisher matrix so that its trace equals D.

    Args:
        fisher: (D, D) or (num_thetas, D, D) Fisher matrices

    Returns:
        fhat: (num_thetas, D, D) normalized Fishers (trace = D)
        tr:   trace of original Fishers
    """
    # make sure we always have a batch dim
    batched = fisher.dim() == 3
    fishers = fisher if batched else fisher.unsqueeze(0)

    d = fishers.size(-1)
    # trace for every matrix in the batch → shape (T,)
    traces = fishers.diagonal(dim1=-2, dim2=-1).sum(-1)
    fhats = (d / traces.view(-1, 1, 1)) * fishers

    # squeeze back if original input was a single matrix
    fhats = fhats if batched else fhats[0]
    return fhats, traces.mean().item()


def eff_dim(
    fhat: torch.Tensor,
    dataset_sizes: Sequence[int],
    gamma: float = 1.0,
) -> torch.Tensor:
    r"""
    Effective dimension **d_{gamma,n}** (Abbas et al., Eq.(2)) for every *n*.

        d_{gamma,n}(M) =
            2 · [ logsumexp_t r_t  - log T ]  /  log(gamma n / (2π log n))
        with  r_t = ½ · log det( I + (gamma n)/(2π log n) · F̂_t )

    where *t* indexes the *num_thetas* Fisher samples.

    Args:
        fhat: Normalised Fisher(s) ``shape = (T, D, D)`` or ``(D, D)``.
        dataset_sizes: sequence of dataset sizes (n values)
        gamma: ositive scalar in (0,1]; defaults to 1 (paper sets gamma as tuning knob).

    Returns:
        Tensor of shape (len(dataset_sizes),) with effective dimensions.
    """
    if fhat.dim() == 2:
        fhat = fhat.unsqueeze(0)  # => (1, D, D)

    T, D, _ = fhat.shape
    device = fhat.device
    effdims = []

    for n in dataset_sizes:
        scale = gamma * n / (2 * math.pi * math.log(max(n, 3)))  # avoid log(1)≈0
        log_Ms = []

        for t in range(T):
            M = torch.eye(D, device=device) + scale * fhat[t]
            sign, logdet = torch.linalg.slogdet(M)
            if sign <= 0:
                # PSD matrix should never give sign ≤0; add ε and retry
                eps = 1e-12
                M = M + eps * torch.eye(D, device=device)
                sign, logdet = torch.linalg.slogdet(M)
            log_Ms.append(0.5 * logdet)  # r_t

        r = torch.stack(log_Ms)  # (T,)
        logmean = torch.logsumexp(r, 0) - math.log(T)
        denom = math.log(scale)
        effdims.append(2.0 * logmean / denom)

    return torch.stack(effdims)

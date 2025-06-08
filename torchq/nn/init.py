import torch


def beta_initialization(tensor: torch.Tensor, a: float = 2.0, b: float = 5.0) -> None:
    r"""Initialize a tensor using a scaled Beta distribution.

    Samples from a Beta(a, b) distribution, scales the values to the range [-1, 1],
    and copies the result into the provided tensor.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
        a (float, optional): Alpha parameter of the Beta distribution. Defaults to 2.0.
        b (float, optional): Beta parameter of the Beta distribution. Defaults to 5.0.
    """
    beta_dist = torch.distributions.Beta(torch.tensor(a), torch.tensor(b))
    beta_sample = beta_dist.sample(tensor.shape)
    with torch.no_grad():
        tensor.copy_(2 * beta_sample - 1)

import torch

from mvae_mnist.model import MVAE
from mvae_mnist.train import elbo_mnist


def test_elbo_returns_finite_scalar() -> None:
    model = MVAE(latent_dim=8)
    image = torch.rand(4, 1, 28, 28)
    label = torch.tensor([0, 1, 2, 3])

    gen_image, gen_label, mu, logvar = model(image_modal=image, label_modal=label)
    loss = elbo_mnist(gen_image, image, gen_label, label, mu, logvar)

    assert loss.ndim == 0
    assert torch.isfinite(loss)

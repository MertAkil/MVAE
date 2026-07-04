import torch

from mvae_mnist.model import MVAE
from mvae_mnist.utils import condition_posterior


def test_prior_conditioning_returns_unit_gaussian_parameters() -> None:
    model = MVAE(latent_dim=8)

    mu, logvar, image, label = condition_posterior(model)

    assert image is None
    assert label is None
    assert mu.shape == (1, 8)
    assert logvar.shape == (1, 8)
    assert torch.allclose(mu, torch.zeros_like(mu))
    assert torch.allclose(logvar, torch.zeros_like(logvar))


def test_label_zero_is_a_valid_condition() -> None:
    model = MVAE(latent_dim=8)

    mu, logvar, image, label = condition_posterior(model, condition_label=0)

    assert image is None
    assert label.item() == 0
    assert mu.shape == (1, 8)
    assert logvar.shape == (1, 8)


def test_label_conditioning_for_nonzero_digit() -> None:
    model = MVAE(latent_dim=8)

    mu, logvar, _, label = condition_posterior(model, condition_label=6)

    assert label.item() == 6
    assert mu.shape == (1, 8)
    assert logvar.shape == (1, 8)


def test_joint_conditioning_accepts_supplied_image_tensor() -> None:
    model = MVAE(latent_dim=8)
    image = torch.rand(1, 1, 28, 28)

    mu, logvar, returned_image, label = condition_posterior(
        model,
        condition_image=image,
        condition_label=5,
    )

    assert returned_image is image
    assert label.item() == 5
    assert mu.shape == (1, 8)
    assert logvar.shape == (1, 8)

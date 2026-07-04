import pytest
import torch

from mvae_mnist.model import MVAE


def test_forward_shapes_for_available_modalities() -> None:
    model = MVAE(latent_dim=8)
    image = torch.rand(4, 1, 28, 28)
    label = torch.tensor([0, 1, 2, 3])

    gen_image, gen_label, mu, logvar = model(image_modal=image, label_modal=None)
    assert gen_image.shape == (4, 784)
    assert gen_label is None
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)

    gen_image, gen_label, mu, logvar = model(image_modal=None, label_modal=label)
    assert gen_image is None
    assert gen_label.shape == (4, 10)
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)

    gen_image, gen_label, mu, logvar = model(image_modal=image, label_modal=label)
    assert gen_image.shape == (4, 784)
    assert gen_label.shape == (4, 10)
    assert mu.shape == (4, 8)
    assert logvar.shape == (4, 8)


def test_prepare_poe_rejects_missing_modalities() -> None:
    model = MVAE(latent_dim=8)

    with pytest.raises(ValueError, match="At least one modality"):
        model.prepare_poe(image_modal=None, label_modal=None)


def test_prior_expert_uses_requested_device_and_dtype() -> None:
    model = MVAE(latent_dim=8).double()
    image = torch.rand(2, 1, 28, 28, dtype=torch.float64)

    mu, logvar = model.prepare_poe(image_modal=image, label_modal=None)

    assert mu.device == image.device
    assert logvar.device == image.device
    assert mu.dtype == image.dtype
    assert logvar.dtype == image.dtype

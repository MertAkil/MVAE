"""Neural network modules for the MNIST multimodal VAE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class MVAE(nn.Module):
    """Product-of-experts multimodal VAE for MNIST images and digit labels."""

    def __init__(self, latent_dim: int):
        super().__init__()
        self.image_encoder = ImageEncoder(latent_dim)
        self.image_decoder = ImageDecoder(latent_dim)
        self.label_encoder = LabelEncoder(latent_dim)
        self.label_decoder = LabelDecoder(latent_dim)
        self.z_dim = latent_dim

    def forward(
        self,
        image_modal: Tensor | None = None,
        label_modal: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
        """Encode available modalities and reconstruct the observed modalities."""

        mu, logvar = self.prepare_poe(image_modal=image_modal, label_modal=label_modal)
        poe_mu, poe_logvar = self.compute_poe(mu, logvar)
        z = self.reparameterize(poe_mu, poe_logvar)

        gen_image = self.image_decoder(z) if image_modal is not None else None
        gen_label = self.label_decoder(z) if label_modal is not None else None

        return gen_image, gen_label, poe_mu, poe_logvar

    @staticmethod
    def reparameterize(mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    @staticmethod
    def prior_expert(
        size: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        return torch.zeros(size, device=device, dtype=dtype), torch.zeros(
            size,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def compute_poe(mu: Tensor, logvar: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
        """Compute a diagonal Gaussian product of experts."""

        var = torch.exp(logvar) + eps
        precision = 1.0 / var
        poe_cov = 1.0 / torch.sum(precision, dim=0)
        poe_mu = poe_cov * torch.sum(mu * precision, dim=0)
        poe_logvar = torch.log(poe_cov)
        return poe_mu, poe_logvar

    def prepare_poe(
        self,
        image_modal: Tensor | None,
        label_modal: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Build the expert stack from the prior plus all observed modalities."""

        if image_modal is None and label_modal is None:
            raise ValueError("At least one modality must be provided.")

        n_samples = image_modal.size(0) if image_modal is not None else label_modal.size(0)
        device = image_modal.device if image_modal is not None else label_modal.device
        dtype = image_modal.dtype if image_modal is not None else next(self.parameters()).dtype

        if (
            image_modal is not None
            and label_modal is not None
            and image_modal.device != label_modal.device
        ):
            raise ValueError("Image and label modalities must be on the same device.")

        mu, logvar = self.prior_expert((1, n_samples, self.z_dim), device=device, dtype=dtype)

        if image_modal is not None:
            img_mu, img_logvar = self.image_encoder(image_modal)
            mu = torch.cat((mu, img_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, img_logvar.unsqueeze(0)), dim=0)

        if label_modal is not None:
            lbl_mu, lbl_logvar = self.label_encoder(label_modal)
            mu = torch.cat((mu, lbl_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, lbl_logvar.unsqueeze(0)), dim=0)

        return mu, logvar


class ImageEncoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.l_input = nn.Linear(784, 512)
        self.l_hidden = nn.Linear(512, 512)
        self.out_mu = nn.Linear(512, z_dim)
        self.out_logvar = nn.Linear(512, z_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = F.relu(self.l_input(x.view(-1, 784)))
        x = F.relu(self.l_hidden(x))
        return self.out_mu(x), self.out_logvar(x)


class ImageDecoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.l_input = nn.Linear(z_dim, 512)
        self.l_hidden = nn.Linear(512, 512)
        self.l_output = nn.Linear(512, 784)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.l_input(x))
        x = F.relu(self.l_hidden(x))
        return self.l_output(x)


class LabelEncoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.l_embed = nn.Embedding(10, 512)
        self.l_input = nn.Linear(512, 512)
        self.l_hidden = nn.Linear(512, 512)
        self.out_mu = nn.Linear(512, z_dim)
        self.out_logvar = nn.Linear(512, z_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = F.relu(self.l_input(self.l_embed(x.long())))
        x = F.relu(self.l_hidden(x))
        return self.out_mu(x), self.out_logvar(x)


class LabelDecoder(nn.Module):
    def __init__(self, z_dim: int):
        super().__init__()
        self.l_input = nn.Linear(z_dim, 512)
        self.l_hidden = nn.Linear(512, 512)
        self.l_output = nn.Linear(512, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.l_input(x))
        x = F.relu(self.l_hidden(x))
        return self.l_output(x)

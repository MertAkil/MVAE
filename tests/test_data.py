from pathlib import Path

import torch
from torch.utils.data import Dataset

from mvae_mnist.data import build_mnist_dataloaders


class FakeMNIST(Dataset):
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool,
        transform: object | None,
    ) -> None:
        self.root = Path(root)
        self.train = train
        self.download = download
        self.transform = transform
        self.n_examples = 12 if train else 5
        self.targets = torch.arange(self.n_examples) % 10

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = torch.full((1, 28, 28), fill_value=float(index) / 255.0)
        return image, int(self.targets[index])


def test_data_loaders_use_train_val_split_and_real_test_split(tmp_path: Path) -> None:
    loaders = build_mnist_dataloaders(
        data_dir=tmp_path,
        batch_size=2,
        train_size=8,
        val_size=4,
        seed=123,
        dataset_factory=FakeMNIST,
    )

    assert len(loaders.train.dataset) == 8
    assert len(loaders.val.dataset) == 4
    assert len(loaders.test.dataset) == 5
    assert loaders.test.dataset.train is False

    image, label = next(iter(loaders.train))
    assert image.shape == (2, 1, 28, 28)
    assert label.shape == (2,)

import torch
from torchvision import datasets, transforms


def get_data_loader(
    data_dir,
    batch_size=32,
    train=True,
    image_size=32,
):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        download=True,
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
    )

    return loader
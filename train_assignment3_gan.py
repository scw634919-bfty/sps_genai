import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.model import get_model
from helper_lib.trainer import train_mnist_gan
from helper_lib.generator import generate_mnist_gan_samples


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    mnist_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    data_loader = DataLoader(
        mnist_dataset,
        batch_size=64,
        shuffle=True
    )

    model = get_model("MNISTGAN")

    model = train_mnist_gan(
        model=model,
        data_loader=data_loader,
        device=device,
        epochs=1,
        z_dim=100,
    )

    os.makedirs("checkpoints/assignment3_gan", exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "z_dim": 100,
        },
        "checkpoints/assignment3_gan/mnist_gan.pth"
    )

    print("Saved checkpoint to checkpoints/assignment3_gan/mnist_gan.pth")

    generate_mnist_gan_samples(
        model=model,
        num_samples=16,
        z_dim=100,
        device=device,
        save_path="mnist_gan_samples.png",
    )


if __name__ == "__main__":
    main()
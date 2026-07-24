import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.generator import generate_diffusion_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_diffusion


def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    train_dataset = torch.utils.data.Subset(
    train_dataset,
    range(256),
)

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    model = get_model("Diffusion").to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.network.parameters(),
        lr=1e-3,
    )

    model = train_diffusion(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=1,
    )

    os.makedirs("checkpoints/diffusion", exist_ok=True)

    checkpoint_path = (
        "checkpoints/diffusion/"
        "diffusion_cifar10.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print(
        f"Diffusion checkpoint saved to "
        f"{checkpoint_path}"
    )

    generate_diffusion_samples(
        model=model,
        device=device,
        num_samples=16,
        diffusion_steps=20,
        image_size=32,
        save_path="diffusion_samples.png",
    )


if __name__ == "__main__":
    main()
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper_lib.generator import generate_energy_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_energy_model


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

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
    )

    model = get_model("Energy").to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
    )

    model = train_energy_model(
        model=model,
        data_loader=train_loader,
        optimizer=optimizer,
        device=device,
        epochs=1,
        sampling_steps=20,
    )

    os.makedirs(
        "checkpoints/energy",
        exist_ok=True,
    )

    checkpoint_path = (
        "checkpoints/energy/"
        "energy_cifar10.pth"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )

    print(
        f"Energy checkpoint saved to "
        f"{checkpoint_path}"
    )

    generate_energy_samples(
        model=model,
        device=device,
        num_samples=16,
        sampling_steps=60,
        image_size=32,
        save_path="energy_samples.png",
    )


if __name__ == "__main__":
    main()
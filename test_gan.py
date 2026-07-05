import torch

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_gan_samples


def main():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    data_loader = get_data_loader(
        data_dir="./data",
        batch_size=32,
        train=True,
        image_size=64,
    )

    model = get_model("GAN")

    model = train_gan(
        model=model,
        data_loader=data_loader,
        device=device,
        epochs=1,
        z_dim=100,
    )

    generate_gan_samples(
        model=model,
        num_samples=16,
        z_dim=100,
        device=device,
        save_path="gan_samples.png",
    )


if __name__ == "__main__":
    main()
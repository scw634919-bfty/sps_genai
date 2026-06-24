import torch
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.generator import generate_samples
from helper_lib.model import get_model
from helper_lib.trainer import train_vae_model, vae_loss_function


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


device = get_device()
print(f"Using device: {device}")

train_loader = get_data_loader(
    "./data",
    batch_size=64,
    train=True,
)

vae = get_model("VAE")

optimizer = optim.Adam(
    vae.parameters(),
    lr=0.001,
)

trained_vae = train_vae_model(
    model=vae,
    data_loader=train_loader,
    criterion=vae_loss_function,
    optimizer=optimizer,
    device=device,
    epochs=10,
)

generate_samples(
    model=trained_vae,
    device=device,
    num_samples=16,
)
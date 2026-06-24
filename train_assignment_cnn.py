import torch
import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


device = get_device()
print(f"Using device: {device}")

train_loader = get_data_loader(
    data_dir="./data",
    batch_size=64,
    train=True,
    image_size=64,
)

val_loader = get_data_loader(
    data_dir="./data",
    batch_size=64,
    train=False,
    image_size=64,
)

model = get_model("AssignmentCNN")

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)

trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=1,
    checkpoint_dir="checkpoints/assignment_cnn",
)

print("AssignmentCNN training completed.")
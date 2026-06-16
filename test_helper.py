import torch.nn as nn
import torch.optim as optim

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_model
from helper_lib.evaluator import evaluate_model


train_loader = get_data_loader(
    "./data",
    batch_size=64,
    train=True,
)

test_loader = get_data_loader(
    "./data",
    batch_size=64,
    train=False,
)

model = get_model("CNN")

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
)

trained_model = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=1,
    checkpoint_dir="checkpoints",
)

avg_loss, accuracy = evaluate_model(
    trained_model,
    test_loader,
    criterion,
)

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Accuracy: {accuracy:.2f}%")
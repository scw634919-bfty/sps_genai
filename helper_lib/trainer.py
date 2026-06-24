import torch
import torch.nn.functional as F
from tqdm import tqdm

from .checkpoints import save_checkpoint


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints",
):
    model = model.to(device)

    best_accuracy = 0.0

    for epoch in range(1, epochs + 1):
        # Training
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
        )

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

            progress_bar.set_postfix(
                loss=loss.item()
            )

        avg_train_loss = train_loss / train_total
        train_accuracy = 100 * train_correct / train_total

        # Validation
        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                predictions = outputs.argmax(dim=1)
                val_correct += (
                    predictions == labels
                ).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / val_total
        val_accuracy = 100 * val_correct / val_total

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        # Save checkpoint for every epoch
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            loss=avg_val_loss,
            accuracy=val_accuracy,
            checkpoint_dir=checkpoint_dir,
        )

        # Save best-performing model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_val_loss,
                accuracy=val_accuracy,
                checkpoint_dir=f"{checkpoint_dir}/best",
            )

    return model

def vae_loss_function(reconstruction, inputs, mu, logvar):
    reconstruction_loss = F.binary_cross_entropy(
        reconstruction,
        inputs,
        reduction="sum",
    )

    kl_divergence = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    return reconstruction_loss + kl_divergence


def train_vae_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
):
    model = model.to(device)
    model.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0

        progress_bar = tqdm(
            data_loader,
            desc=f"Epoch {epoch}/{epochs}",
        )

        for images, _ in progress_bar:
            images = images.to(device)

            optimizer.zero_grad()

            reconstruction, mu, logvar = model(images)

            loss = criterion(
                reconstruction,
                images,
                mu,
                logvar,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        average_loss = total_loss / len(data_loader.dataset)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Average Loss: {average_loss:.4f}"
        )

    print("Finished VAE Training")

    return model
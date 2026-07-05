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

def train_gan(
    model,
    data_loader,
    device="cpu",
    epochs=5,
    z_dim=100,
    lr=0.0002,
    beta1=0.5,
):
    import torch
    import torch.nn as nn
    import torch.optim as optim

    model.to(device)
    generator = model.generator
    critic = model.critic

    generator.train()
    critic.train()

    criterion = nn.BCEWithLogitsLoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_c = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, 0.999))

    for epoch in range(epochs):
        total_g_loss = 0.0
        total_c_loss = 0.0

        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # -------------------------
            # Train Critic / Discriminator
            # -------------------------
            optimizer_c.zero_grad()

            real_outputs = critic(real_images)
            real_loss = criterion(real_outputs, real_labels)

            noise = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(noise)
            fake_outputs = critic(fake_images.detach())
            fake_loss = criterion(fake_outputs, fake_labels)

            c_loss = real_loss + fake_loss
            c_loss.backward()
            optimizer_c.step()

            # -------------------------
            # Train Generator
            # -------------------------
            optimizer_g.zero_grad()

            noise = torch.randn(batch_size, z_dim, device=device)
            fake_images = generator(noise)
            fake_outputs = critic(fake_images)

            g_loss = criterion(fake_outputs, real_labels)
            g_loss.backward()
            optimizer_g.step()

            total_c_loss += c_loss.item()
            total_g_loss += g_loss.item()

        avg_c_loss = total_c_loss / len(data_loader)
        avg_g_loss = total_g_loss / len(data_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Critic Loss: {avg_c_loss:.4f}, "
            f"Generator Loss: {avg_g_loss:.4f}"
        )

    return model
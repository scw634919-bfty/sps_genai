import math

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def generate_samples(
    model,
    device="cpu",
    num_samples=16,
):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        z = torch.randn(
            num_samples,
            model.latent_dim,
            device=device,
        )

        samples = model.decode(z)

        columns = math.ceil(math.sqrt(num_samples))

        grid = make_grid(
            samples,
            nrow=columns,
        )

        image = grid.permute(1, 2, 0).cpu().numpy()

        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

def generate_gan_samples(
    model,
    num_samples=16,
    z_dim=100,
    device="cpu",
    save_path="gan_samples.png",
):
    import torch
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    model.to(device)
    model.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, z_dim, device=device)
        fake_images = model.generator(noise)

    # Convert from [-1, 1] to [0, 1] because Generator uses Tanh()
    fake_images = (fake_images + 1) / 2

    grid = make_grid(fake_images, nrow=4)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

    print(f"GAN samples saved to {save_path}")

    return fake_images

def generate_mnist_gan_samples(
    model,
    num_samples=16,
    z_dim=100,
    device="cpu",
    save_path="mnist_gan_samples.png",
):
    import torch
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid

    model.to(device)
    model.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, z_dim, device=device)
        fake_images = model.generator(noise)

    # Generator uses Tanh(), so output is in [-1, 1].
    # Convert it back to [0, 1] for saving/viewing.
    fake_images = (fake_images + 1) / 2

    grid = make_grid(fake_images, nrow=4)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).cpu(), cmap="gray")
    plt.axis("off")
    plt.savefig(save_path)
    plt.close()

    print(f"MNIST GAN samples saved to {save_path}")

    return fake_images
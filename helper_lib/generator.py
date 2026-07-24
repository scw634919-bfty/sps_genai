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

def generate_diffusion_samples(
    model,
    device="cpu",
    num_samples=16,
    diffusion_steps=20,
    image_size=32,
    save_path="diffusion_samples.png",
):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        initial_noise = torch.randn(
            num_samples,
            3,
            image_size,
            image_size,
            device=device,
        )

        generated_images = model.generate(
            num_images=num_samples,
            diffusion_steps=diffusion_steps,
            image_size=image_size,
            initial_noise=initial_noise,
        )

    columns = math.ceil(math.sqrt(num_samples))

    grid = make_grid(
        generated_images.cpu(),
        nrow=columns,
        padding=2,
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(
        grid.permute(1, 2, 0).numpy()
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.close()

    print(
        f"Diffusion samples saved to {save_path}"
    )

    return generated_images

def generate_energy_samples(
    model,
    device="cpu",
    num_samples=16,
    image_size=32,
    sampling_steps=60,
    step_size=10.0,
    noise_scale=0.005,
    save_path="energy_samples.png",
):
    model = model.to(device)
    model.eval()

    samples = torch.rand(
        num_samples,
        3,
        image_size,
        image_size,
        device=device,
    )

    for _ in range(sampling_steps):
        samples = samples.detach()
        samples.requires_grad_(True)

        noise = torch.randn_like(samples) * noise_scale

        energies = model(samples)
        total_energy = energies.sum()

        gradients = torch.autograd.grad(
            outputs=total_energy,
            inputs=samples,
            create_graph=False,
        )[0]

        samples = samples - step_size * gradients + noise
        samples = torch.clamp(samples, 0.0, 1.0)

    samples = samples.detach().cpu()

    columns = math.ceil(math.sqrt(num_samples))

    grid = make_grid(
        samples,
        nrow=columns,
        padding=2,
    )

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        save_path,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Energy samples saved to {save_path}")

    return samples
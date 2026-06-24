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
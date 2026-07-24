import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

def swish(x):
    return x * torch.sigmoid(x)

class FCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 32 * 3, 200)
        self.fc2 = nn.Linear(200, 150)
        self.fc3 = nn.Linear(150, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)

        return x


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            padding=1,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1,
        )

        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class EnhancedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3,
            16,
            kernel_size=3,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            64,
            128,
            kernel_size=3,
            padding=1,
        )
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()

        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        self.fc_decode = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                32, 3, kernel_size=4, stride=2, padding=1
            ),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):
        x = self.fc_decode(z)
        x = x.view(-1, 128, 4, 4)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

class AssignmentCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2,
            ),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()

        self.z_dim = z_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.9),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(x.size(0), self.z_dim, 1, 1)
        return self.model(x)


class GAN(nn.Module):
    def __init__(self, z_dim=100):
        super(GAN, self).__init__()
        self.generator = Generator(z_dim=z_dim)
        self.critic = Critic()

    def forward(self, z):
        return self.generator(z)

class MNISTGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super(MNISTGenerator, self).__init__()

        self.z_dim = z_dim

        self.fc = nn.Linear(z_dim, 7 * 7 * 128)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), 128, 7, 7)
        x = self.model(x)
        return x


class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super(MNISTDiscriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1)
        )

    def forward(self, x):
        return self.model(x)


class MNISTGAN(nn.Module):
    def __init__(self, z_dim=100):
        super(MNISTGAN, self).__init__()

        self.generator = MNISTGenerator(z_dim=z_dim)
        self.discriminator = MNISTDiscriminator()

    def forward(self, z):
        return self.generator(z)

def linear_diffusion_schedule(
    diffusion_times,
    min_rate=1e-4,
    max_rate=0.02,
):
    diffusion_times = diffusion_times.to(dtype=torch.float32)

    betas = min_rate + diffusion_times * (
        max_rate - min_rate
    )

    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)

    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    signal_rates = torch.cos(
        diffusion_times * math.pi / 2
    )

    noise_rates = torch.sin(
        diffusion_times * math.pi / 2
    )

    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(
    diffusion_times,
    min_signal_rate=0.02,
    max_signal_rate=0.95,
):
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()

    start_angle = torch.acos(
        torch.tensor(
            max_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device,
        )
    )

    end_angle = torch.acos(
        torch.tensor(
            min_signal_rate,
            dtype=torch.float32,
            device=diffusion_times.device,
        )
    )

    diffusion_angles = start_angle + diffusion_times_flat * (
        end_angle - start_angle
    )

    signal_rates = torch.cos(
        diffusion_angles
    ).reshape(original_shape)

    noise_rates = torch.sin(
        diffusion_angles
    ).reshape(original_shape)

    return noise_rates, signal_rates


class SinusoidalEmbedding(nn.Module):
    def __init__(self, num_frequencies=16):
        super().__init__()

        self.num_frequencies = num_frequencies

        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                num_frequencies,
            )
        )

        self.register_buffer(
            "angular_speeds",
            2.0
            * math.pi
            * frequencies.view(1, 1, 1, -1),
        )

    def forward(self, x):
        x = x.expand(
            -1,
            1,
            1,
            self.num_frequencies,
        )

        sin_part = torch.sin(
            self.angular_speeds * x
        )

        cos_part = torch.cos(
            self.angular_speeds * x
        )

        return torch.cat(
            [sin_part, cos_part],
            dim=-1,
        )

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.batch_norm1 = nn.BatchNorm2d(in_channels)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        if in_channels == out_channels:
            self.residual_connection = nn.Identity()
        else:
            self.residual_connection = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = self.residual_connection(x)

        x = self.batch_norm1(x)
        x = F.silu(x)
        x = self.conv1(x)

        x = self.batch_norm2(x)
        x = F.silu(x)
        x = self.conv2(x)

        return x + residual


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        block_depth=2,
    ):
        super().__init__()

        blocks = []

        current_channels = in_channels

        for _ in range(block_depth):
            blocks.append(
                ResidualBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                )
            )

            current_channels = out_channels

        self.blocks = nn.ModuleList(blocks)

        self.downsample = nn.AvgPool2d(
            kernel_size=2,
            stride=2,
        )

    def forward(self, x):
        skip_connections = []

        for block in self.blocks:
            x = block(x)
            skip_connections.append(x)

        x = self.downsample(x)

        return x, skip_connections

class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        block_depth=2,
    ):
        super().__init__()

        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        blocks = []

        current_channels = in_channels + skip_channels

        for _ in range(block_depth):
            blocks.append(
                ResidualBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                )
            )

            current_channels = out_channels + skip_channels

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, skip_connections):
        x = self.upsample(x)

        for block in self.blocks:
            skip = skip_connections.pop()

            x = torch.cat(
                [x, skip],
                dim=1,
            )

            x = block(x)

        return x


class UNet(nn.Module):
    def __init__(
        self,
        image_size=32,
        image_channels=3,
        widths=(32, 64, 96),
        block_depth=2,
        embedding_dim=32,
    ):
        super().__init__()

        self.image_size = image_size
        self.image_channels = image_channels

        self.noise_embedding = SinusoidalEmbedding(
            num_frequencies=embedding_dim // 2
        )

        self.input_projection = nn.Conv2d(
            image_channels,
            widths[0],
            kernel_size=1,
        )

        self.embedding_projection = nn.Conv2d(
            embedding_dim,
            widths[0],
            kernel_size=1,
        )

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    in_channels=widths[index],
                    out_channels=widths[index + 1],
                    block_depth=block_depth,
                )
                for index in range(len(widths) - 1)
            ]
        )

        self.middle_blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=widths[-1],
                    out_channels=widths[-1],
                )
                for _ in range(block_depth)
            ]
        )

        self.up_blocks = nn.ModuleList(
    [
        UpBlock(
            in_channels=widths[index + 1],
            skip_channels=widths[index + 1],
            out_channels=widths[index],
            block_depth=block_depth,
        )
        for index in reversed(
            range(len(widths) - 1)
        )
    ]
)

        self.output_projection = nn.Conv2d(
            widths[0],
            image_channels,
            kernel_size=1,
        )

        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, noisy_images, noise_variances):
        noise_embedding = self.noise_embedding(
            noise_variances
        )

        noise_embedding = noise_embedding.permute(
            0,
            3,
            1,
            2,
        )

        noise_embedding = self.embedding_projection(
            noise_embedding
        )

        noise_embedding = F.interpolate(
            noise_embedding,
            size=(
                noisy_images.shape[2],
                noisy_images.shape[3],
            ),
            mode="nearest",
        )

        x = self.input_projection(noisy_images)
        x = x + noise_embedding

        skip_connections = []

        for down_block in self.down_blocks:
            x, block_skips = down_block(x)
            skip_connections.extend(block_skips)

        for middle_block in self.middle_blocks:
            x = middle_block(x)

        for up_block in self.up_blocks:
            x = up_block(x, skip_connections)

        return self.output_projection(x)   

class DiffusionModel(nn.Module):
    def __init__(
        self,
        network,
        schedule_fn=offset_cosine_diffusion_schedule,
        ema_decay=0.999,
    ):
        super().__init__()

        self.network = network
        self.ema_network = copy.deepcopy(network)

        self.ema_network.eval()

        for parameter in self.ema_network.parameters():
            parameter.requires_grad = False

        self.schedule_fn = schedule_fn
        self.ema_decay = ema_decay

        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0

    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std

    def normalize(self, images):
        return (
            images - self.normalizer_mean
        ) / self.normalizer_std

    def denormalize(self, images):
        images = (
            images * self.normalizer_std
            + self.normalizer_mean
        )

        return torch.clamp(
            images,
            min=0.0,
            max=1.0,
        )

    def denoise(
        self,
        noisy_images,
        noise_rates,
        signal_rates,
        training=True,
    ):
        if training:
            network = self.network
        else:
            network = self.ema_network

        predicted_noises = network(
            noisy_images,
            noise_rates**2,
        )

        predicted_images = (
            noisy_images
            - noise_rates * predicted_noises
        ) / signal_rates

        return predicted_noises, predicted_images

    def update_ema(self):
        with torch.no_grad():
            for ema_parameter, parameter in zip(
                self.ema_network.parameters(),
                self.network.parameters(),
            ):
                ema_parameter.mul_(self.ema_decay)
                ema_parameter.add_(
                    parameter,
                    alpha=1.0 - self.ema_decay,
                )

    def train_step(
        self,
        images,
        optimizer,
        loss_fn,
    ):
        self.network.train()

        images = self.normalize(images)

        noises = torch.randn_like(images)

        diffusion_times = torch.rand(
            images.size(0),
            1,
            1,
            1,
            device=images.device,
        )

        noise_rates, signal_rates = self.schedule_fn(
            diffusion_times
        )

        noisy_images = (
            signal_rates * images
            + noise_rates * noises
        )

        predicted_noises, _ = self.denoise(
            noisy_images,
            noise_rates,
            signal_rates,
            training=True,
        )

        loss = loss_fn(
            predicted_noises,
            noises,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.update_ema()

        return loss.item()

    def test_step(
        self,
        images,
        loss_fn,
    ):
        self.ema_network.eval()

        images = self.normalize(images)

        noises = torch.randn_like(images)

        diffusion_times = torch.rand(
            images.size(0),
            1,
            1,
            1,
            device=images.device,
        )

        noise_rates, signal_rates = self.schedule_fn(
            diffusion_times
        )

        noisy_images = (
            signal_rates * images
            + noise_rates * noises
        )

        with torch.no_grad():
            predicted_noises, _ = self.denoise(
                noisy_images,
                noise_rates,
                signal_rates,
                training=False,
            )

            loss = loss_fn(
                predicted_noises,
                noises,
            )

        return loss.item()

    def reverse_diffusion(
        self,
        initial_noise,
        diffusion_steps=20,
    ):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise

        predicted_images = None

        for step in range(diffusion_steps):
            diffusion_times = torch.ones(
                initial_noise.size(0),
                1,
                1,
                1,
                device=initial_noise.device,
            )

            diffusion_times = diffusion_times * (
                1.0 - step * step_size
            )

            noise_rates, signal_rates = self.schedule_fn(
                diffusion_times
            )

            predicted_noises, predicted_images = (
                self.denoise(
                    current_images,
                    noise_rates,
                    signal_rates,
                    training=False,
                )
            )

            next_diffusion_times = torch.clamp(
                diffusion_times - step_size,
                min=0.0,
            )

            next_noise_rates, next_signal_rates = (
                self.schedule_fn(
                    next_diffusion_times
                )
            )

            current_images = (
                next_signal_rates * predicted_images
                + next_noise_rates * predicted_noises
            )

        return predicted_images

    def generate(
        self,
        num_images=16,
        diffusion_steps=20,
        image_size=32,
        initial_noise=None,
    ):
        device = next(self.network.parameters()).device

        if initial_noise is None:
            initial_noise = torch.randn(
                num_images,
                self.network.image_channels,
                image_size,
                image_size,
                device=device,
            )

        self.ema_network.eval()

        with torch.no_grad():
            generated_images = self.reverse_diffusion(
                initial_noise,
                diffusion_steps=diffusion_steps,
            )

        return self.denormalize(generated_images)

    def forward(
        self,
        noisy_images,
        noise_variances,
    ):
        return self.network(
            noisy_images,
            noise_variances,
        ) 

class EnergyModel(nn.Module):
    def __init__(self, image_channels=3):
        super().__init__()

        self.conv1 = nn.Conv2d(
            image_channels,
            16,
            kernel_size=5,
            stride=2,
            padding=2,
        )

        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv3 = nn.Conv2d(
            32,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.conv4 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(
            64 * 2 * 2,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            1,
        )

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))

        x = self.flatten(x)

        x = swish(self.fc1(x))

        return self.fc2(x)

def get_model(model_name):
    if model_name == "FCNN":
        return FCNN()

    if model_name == "CNN":
        return CNN()

    if model_name == "EnhancedCNN":
        return EnhancedCNN()

    if model_name == "VAE":
        return VAE()

    if model_name == "AssignmentCNN":
        return AssignmentCNN()

    if model_name == "Generator":
        return Generator()

    if model_name == "Critic":
        return Critic()

    if model_name == "GAN":
        return GAN()

    if model_name == "MNISTGenerator":
        return MNISTGenerator()

    if model_name == "MNISTDiscriminator":
        return MNISTDiscriminator()

    if model_name == "MNISTGAN":
        return MNISTGAN()

    if model_name == "Diffusion":
        unet = UNet(
            image_size=32,
            image_channels=3,
            widths=(32, 64, 96),
            block_depth=2,
            embedding_dim=32,
        )

        return DiffusionModel(
            network=unet,
            schedule_fn=offset_cosine_diffusion_schedule,
        )

    if model_name == "Energy":
        return EnergyModel(
            image_channels=3,
        )

    raise ValueError(
        "model_name must be 'FCNN', 'CNN', 'EnhancedCNN', 'VAE', "
        "'AssignmentCNN', 'Generator', 'Critic', 'GAN', "
        "'MNISTGenerator', 'MNISTDiscriminator', 'MNISTGAN', "
        "'Diffusion', or 'Energy'"
    )
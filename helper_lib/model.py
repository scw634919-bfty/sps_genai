import torch
import torch.nn as nn
import torch.nn.functional as F


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

    raise ValueError(
    "model_name must be 'FCNN', 'CNN', 'EnhancedCNN', 'VAE', "
    "'AssignmentCNN', 'Generator', 'Critic', 'GAN', "
    "'MNISTGenerator', 'MNISTDiscriminator', or 'MNISTGAN'"
    )
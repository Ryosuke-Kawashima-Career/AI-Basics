"""This file makes a Variational Auto Encoder for MNIST dataset.
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_dataloaders(
    batch_size: int = 128,
    data_dir: str = "./data",
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, datasets.MNIST]:
    """Load and flatten MNIST train/validation datasets."""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    valid_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, valid_dataset


def torch_log(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable natural log to prevent log(0) = -inf / NaN."""
    return torch.log(torch.clamp(x, min=1e-10, max=1e10))


class VAE(nn.Module):
    def __init__(self, z_dim: int = 2, in_dim: int = 28 * 28, hidden_dim: int = 200) -> None:
        """
        z_dim: the hidden dimension of this VAE
        """
        super().__init__()
        if z_dim < 2:
            raise ValueError("z dimensionality must be larger or equal to 2.")
        self.z_dim = z_dim

        # Encoder: maps input x -> Gaussian parameters (mean, std)
        self.dense_enc1 = nn.Linear(in_dim, hidden_dim)
        self.dense_enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense_encmean = nn.Linear(hidden_dim, z_dim)
        self.dense_encvar = nn.Linear(hidden_dim, z_dim)

        # Decoder: maps latent z -> Bernoulli parameter y in (0, 1)
        self.dense_dec1 = nn.Linear(z_dim, hidden_dim)
        self.dense_dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense_dec3 = nn.Linear(hidden_dim, in_dim)

    def encoder(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Infers the latent variables(z) from a Gaussian distribution.
        Args:
            x (torch.Tensor): (b, c*h*w)
        Returns:
            mean (torch.Tensor): (b, z_dim)
            std (torch.Tensor): (b, z_dim)
        """
        h = F.relu(self.dense_enc1(x))
        h = F.relu(self.dense_enc2(h))
        mean = self.dense_encmean(h)
        std = F.softplus(self.dense_encvar(h))
        return mean, std

    def sample_z(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Reparameterization technique for backward propagation.
        Args:
            mean: (b, z_dim)
            std: (b, z_dim)
        Returns:
            z: (b, z_dim)
        """
        if self.training:
            epsilon = torch.randn_like(mean)
            return mean + std * epsilon
        return mean

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """Generates images (x) from a Bernoulli distribution.
        Args:
            z (torch.Tensor): (b, z_dim) a vector of latent variables
        Returns:
            x (torch.Tensor): Reconstructed images (b, c*h*w)
        """
        h = F.relu(self.dense_dec1(z))
        h = F.relu(self.dense_dec2(h))
        x = torch.sigmoid(self.dense_dec3(h))
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Flattened input image (b, c*h*w)
        Returns:
            y: Reconstructed image (b, c*h*w)
            z: Latent variables (b, z_dim)
        """
        mean, std = self.encoder(x)
        z = self.sample_z(mean, std)
        y = self.decoder(z)
        return y, z

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the objective function (Negative ELBO = KL + Reconstruction loss).
        Args:
            x (torch.Tensor): (b, c*h*w)
        Returns:
            KL (torch.Tensor): Regularization scalar (,)
            reconstruction_loss (torch.Tensor): Reconstruction error scalar (,)
        """
        mean, std = self.encoder(x)

        # Analytical KL divergence between N(mean, std^2) and N(0, I)
        kl = -0.5 * torch.mean(
            torch.sum(1.0 + torch_log(std**2) - mean**2 - std**2, dim=1)
        )

        z = self.sample_z(mean, std)
        y = self.decoder(z)

        # Bernoulli log-likelihood reconstruction term
        reconstruction = torch.mean(
            torch.sum(x * torch_log(y) + (1.0 - x) * torch_log(1.0 - y), dim=1)
        )
        return kl, -reconstruction

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        n_epochs: int = 15,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
    ) -> None:
        """Train the VAE and evaluate on the validation set each epoch."""
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for epoch in range(n_epochs):
            losses = []
            kl_losses = []
            reconstruction_losses = []

            self.train()
            for x, _ in train_loader:
                x = x.to(device)
                optimizer.zero_grad()
                kl_loss, reconstruction_loss = self.loss(x)
                loss = kl_loss + reconstruction_loss
                loss.backward()
                optimizer.step()

                losses.append(loss.detach().cpu().item())
                kl_losses.append(kl_loss.detach().cpu().item())
                reconstruction_losses.append(reconstruction_loss.detach().cpu().item())

            losses_val = []
            self.eval()
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(device)
                    kl_loss, reconstruction_loss = self.loss(x)
                    loss = kl_loss + reconstruction_loss
                    losses_val.append(loss.detach().cpu().item())

            print(
                f"EPOCH: {epoch + 1:02d}    "
                f"Train Loss: {np.mean(losses):.4f} "
                f"(KL: {np.mean(kl_losses):.4f}, Recon: {np.mean(reconstruction_losses):.4f})    "
                f"Valid Loss: {np.mean(losses_val):.4f}"
            )

    def validate(
        self,
        valid_dataset: datasets.MNIST,
        device: str | torch.device = "cpu",
        num_images: int = 40,
    ) -> None:
        """Validates the reconstructed images of MNIST data."""
        self.eval()
        fig = plt.figure(figsize=(10, 4))
        fig.suptitle("Reconstructed Validation Images")
        with torch.no_grad():
            for i in range(num_images):
                x, _ = valid_dataset[i]
                x = x.unsqueeze(0).to(device)
                y, _ = self(x)
                im = y.view(28, 28).detach().cpu().numpy()
                ax = fig.add_subplot(4, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(im, cmap="gray")
        plt.tight_layout()
        plt.show()
        plt.close(fig)

    def visualize_latent(
        self,
        val_loader: torch.utils.data.DataLoader,
        device: str | torch.device = "cpu",
        reduc: str = "PCA",
    ) -> None:
        """Plot 2D latent space representation colored by digit class."""
        self.eval()
        z_list = []
        t_list = []
        with torch.no_grad():
            for x, t in val_loader:
                x = x.to(device)
                _, z = self(x)
                z_list.append(z.detach().cpu().numpy())
                t_list.append(t.numpy())

        z_val = np.concatenate(z_list, axis=0)
        labels = np.concatenate(t_list, axis=0)

        if self.z_dim > 2:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE

            if reduc.upper() == "TSNE":
                z_reduc = TSNE(n_components=2, random_state=42).fit_transform(z_val).T
            elif reduc.upper() == "PCA":
                z_reduc = PCA(n_components=2, random_state=42).fit_transform(z_val).T
            else:
                raise ValueError("Please choose dimensionality reduction method from 'TSNE' or 'PCA'.")
        else:
            z_reduc = z_val.T

        colors = [
            "khaki", "lightgreen", "cornflowerblue", "violet", "sienna",
            "darkturquoise", "slateblue", "orange", "darkcyan", "tomato",
        ]

        fig = plt.figure(figsize=(6, 6))
        plt.scatter(z_reduc[0], z_reduc[1], s=1.0, c=[colors[t] for t in labels])
        for i in range(10):
            plt.scatter([], [], c=colors[i], label=str(i))
        plt.legend(title="Digit")
        plt.title(f"Latent Space (z_dim={self.z_dim})")
        plt.tight_layout()
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, valid_dataset = get_dataloaders(batch_size=128)

    vae = VAE(z_dim=2)
    vae.fit(train_loader, val_loader, n_epochs=15, lr=1e-3, device=device)
    vae.validate(valid_dataset, device=device)
    vae.visualize_latent(val_loader, device=device)

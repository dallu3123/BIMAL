import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPActionVAE(nn.Module):
    """
    A simple yet effective MLP-based Variational AutoEncoder for Action sequences.
    Flattens the temporal dimension (T) and action dimension (D), then compresses them
    into a continuous latent space. Very fast and easy to train.
    """

    def __init__(self, action_dim: int, seq_len: int, latent_dim: int, hidden_dim: int = 512) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder: (B, T * D) -> (B, 2 * latent_dim)
        self.encoder = nn.Sequential(
            nn.Linear(action_dim * seq_len, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim * 2),  # Outputs both mean and log-variance
        )

        # Decoder: (B, latent_dim) -> (B, T * D)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim * seq_len),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Action sequence of shape (B, T, D)
        Returns:
            mu: Mean of the latent distribution (B, latent_dim)
            logvar: Log-variance of the latent distribution (B, latent_dim)
        """
        b = x.shape[0]
        x_flat = x.view(b, -1)
        stats = self.encoder(x_flat)
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Samples z from the normal distribution N(mu, var) using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent vector of shape (B, latent_dim)
        Returns:
            Reconstructed action sequence of shape (B, T, D)
        """
        b = z.shape[0]
        x_recon_flat = self.decoder(z)
        x_recon = x_recon_flat.view(b, self.seq_len, self.action_dim)
        return x_recon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        Returns:
            x_recon: Reconstructed actions
            mu: Latent mean
            logvar: Latent log-variance
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 1e-4) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Helper function to compute VAE loss (Reconstruction + KL Divergence).
        """
        x_recon, mu, logvar = self(x)

        # 1. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(x_recon, x)

        # 2. KL Divergence Loss
        # KL(N(mu, var) || N(0, 1)) = -0.5 * sum(1 + log(var) - mu^2 - var)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()

        total_loss = recon_loss + kl_weight * kl_loss

        metrics = {
            "vae_loss": total_loss.item(),
            "vae_recon_loss": recon_loss.item(),
            "vae_kl_loss": kl_loss.item(),
        }

        return total_loss, metrics


class Conv1DActionVAE(nn.Module):
    """
    A 1D Convolutional VAE.
    Better at preserving temporal structure in the action trajectories compared to MLP.
    Highly recommended for longer action horizons (e.g., T >= 32).
    """

    def __init__(self, action_dim: int, seq_len: int, latent_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(action_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.Mish(),
        )

        # Calculate flattened dimension after convolutions
        # Example for seq_len=16: 16 -> 8 -> 4. So length is seq_len // 4 (approx)
        self._conv_out_len = (seq_len + 1) // 2
        self._conv_out_len = (self._conv_out_len + 1) // 2

        self.encoder_linear = nn.Linear(hidden_dim * 4 * self._conv_out_len, latent_dim * 2)

        # Decoder
        self.decoder_linear = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 4 * self._conv_out_len),
            nn.Mish(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose1d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.Mish(),
            nn.ConvTranspose1d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.Mish(),
            nn.Conv1d(hidden_dim, action_dim, kernel_size=3, padding=1),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D) -> Conv1d expects (B, C, L), so permute to (B, D, T)
        x = x.permute(0, 2, 1)
        h = self.encoder_conv(x)
        h_flat = h.reshape(h.size(0), -1)
        stats = self.encoder_linear(h_flat)
        mu, logvar = torch.chunk(stats, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h_flat = self.decoder_linear(z)
        # Reshape back to (B, C, L)
        h = h_flat.view(h_flat.size(0), -1, self._conv_out_len)
        x_recon = self.decoder_conv(h)

        # Ensure the sequence length matches exactly
        if x_recon.shape[-1] != self.seq_len:
            x_recon = F.interpolate(x_recon, size=self.seq_len, mode="linear", align_corners=False)

        # (B, D, T) -> (B, T, D)
        return x_recon.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def compute_loss(self, x: torch.Tensor, kl_weight: float = 1e-4) -> tuple[torch.Tensor, dict[str, float]]:
        x_recon, mu, logvar = self(x)
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, {
            "vae_loss": total_loss.item(),
            "vae_recon_loss": recon_loss.item(),
            "vae_kl_loss": kl_loss.item(),
        }

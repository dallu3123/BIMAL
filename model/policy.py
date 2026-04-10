import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vita import SimpleFlowNet
from model.vision import ResNetObserver, MultiCameraObserver
from model.action_vae import MLPActionVAE


def compute_contrastive_loss(
    image_features: torch.Tensor, action_features: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """
    InfoNCE contrastive loss between image and action features.
    Encourages matched (image, action) pairs to be close in latent space.
    """
    image_features = F.normalize(image_features, dim=1)
    action_features = F.normalize(action_features, dim=1)

    logits = torch.matmul(image_features, action_features.T) / temperature

    labels = torch.arange(image_features.size(0), device=logits.device)
    loss_i2a = F.cross_entropy(logits, labels)
    loss_a2i = F.cross_entropy(logits.T, labels)

    return (loss_i2a + loss_a2i) / 2


class VITAPolicy(nn.Module):
    """
    Top-level VITA Policy integrating Vision Encoder, Action VAE, and Flow Network.

    Data flows as follows:
    [Training]
      1. Images -> Vision Encoder -> Vision Latent (x0)
      2. Actions -> Action VAE Encoder -> Action Latent (x1)
      3. Flow Matching Loss: sample t, interpolate xt, predict velocity
      4. (Optional) Decode Flow Latents: sample from flow, compute consistency & contrastive losses
      5. Loss = Flow Loss + VAE Loss + Consistency Loss + Contrastive Losses

    [Inference]
      1. Images -> Vision Encoder -> Vision Latent (x0)
      2. Simulate ODE from t=0 to 1 starting at x0 -> Action Latent (x1)
      3. Action Latent (x1) -> Action VAE Decoder -> Action Sequence
    """

    def __init__(
        self,
        action_dim: int,
        seq_len: int,
        num_cameras: int = 3,
        latent_dim: int = 512,
        flow_hidden_dim: int = 1024,
        flow_num_layers: int = 6,
        num_sampling_steps: int = 10,
        # VITA core parameters
        decode_flow_latents: bool = True,
        consistency_weight: float = 1.0,
        enc_contrastive_weight: float = 0.0,
        flow_contrastive_weight: float = 0.0,
        sigma: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_sampling_steps = num_sampling_steps

        # VITA core settings
        self.decode_flow_latents = decode_flow_latents
        self.consistency_weight = consistency_weight
        self.enc_contrastive_weight = enc_contrastive_weight
        self.flow_contrastive_weight = flow_contrastive_weight
        self.sigma = sigma

        # 1. Vision Encoder (ResNet18 baseline for lightweight training)
        self.vision_encoder = MultiCameraObserver(
            encoder=ResNetObserver(resnet_type="resnet18", out_dim=latent_dim),
            num_cameras=num_cameras,
            feature_dim=latent_dim,
        )

        # 2. Action VAE (Compresses B x T x D into B x Latent)
        self.action_vae = MLPActionVAE(
            action_dim=action_dim,
            seq_len=seq_len,
            latent_dim=latent_dim,
        )

        # 3. Flow Network (Predicts velocity)
        self.flow_net = SimpleFlowNet(
            input_dim=latent_dim,
            hidden_dim=flow_hidden_dim,
            output_dim=latent_dim,
            num_layers=flow_num_layers,
        )

    def _flow_sample(self, start: torch.Tensor) -> torch.Tensor:
        """
        Sample from the flow network using Euler ODE solver.
        Used during training (decode_flow_latents) and inference.
        """
        b = start.shape[0]
        device = start.device
        xt = start
        dt = 1.0 / self.num_sampling_steps

        for i in range(self.num_sampling_steps):
            t_val = i / self.num_sampling_steps
            t = torch.full((b,), t_val, device=device)
            vt = self.flow_net(xt, t)
            xt = xt + vt * dt

        return xt

    def compute_loss(self, images_dict: dict[str, torch.Tensor], actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Computes Flow Matching Loss + VAE Loss + Consistency Loss + Contrastive Losses.
        """
        b = actions.shape[0]
        device = actions.device

        # 1. Vision Latents (x0) - Start of the flow
        obs_latents = self.vision_encoder(images_dict)

        # 2. Action Latents (x1) - End of the flow
        x_recon, mu, logvar = self.action_vae(actions)
        action_latents = self.action_vae.reparameterize(mu, logvar)

        # VAE Loss (Reconstruction + KL)
        recon_loss = F.mse_loss(x_recon, actions)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        vae_loss = recon_loss + 1e-4 * kl_loss

        # 3. Flow Matching Loss
        x0 = obs_latents
        x1 = action_latents

        # Add noise if sigma > 0 (latent noise std)
        if self.sigma > 0:
            x0 = x0 + self.sigma * torch.randn_like(x0)

        t = torch.rand(b, device=device)
        t_expand = t.view(b, 1)

        xt = (1 - t_expand) * x0 + t_expand * x1
        target_v = x1 - x0
        pred_v = self.flow_net(xt, t)

        flow_loss = F.mse_loss(pred_v, target_v)

        # Total Loss (base)
        total_loss = flow_loss + vae_loss

        metrics = {
            "total_loss": 0.0,
            "flow_loss": flow_loss.item(),
            "vae_recon_loss": recon_loss.item(),
            "vae_kl_loss": kl_loss.item(),
        }

        # 4. Encoder Contrastive Loss
        if self.enc_contrastive_weight > 0:
            enc_contrast = compute_contrastive_loss(obs_latents, action_latents)
            total_loss = total_loss + self.enc_contrastive_weight * enc_contrast
            metrics["enc_contrastive_loss"] = enc_contrast.item()

        # 5. Decode Flow Latents: consistency & flow contrastive losses
        if self.decode_flow_latents and (self.consistency_weight > 0 or self.flow_contrastive_weight > 0):
            action_latents_pred = self._flow_sample(obs_latents)

            # Consistency Loss (FLC): predicted action latents should match encoder action latents
            if self.consistency_weight > 0:
                consistency_loss = F.mse_loss(action_latents_pred, action_latents)
                total_loss = total_loss + self.consistency_weight * consistency_loss
                metrics["consistency_loss"] = consistency_loss.item()

            # Flow Contrastive Loss
            if self.flow_contrastive_weight > 0:
                flow_contrast = compute_contrastive_loss(obs_latents, action_latents_pred)
                total_loss = total_loss + self.flow_contrastive_weight * flow_contrast
                metrics["flow_contrastive_loss"] = flow_contrast.item()

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics

    @torch.no_grad()
    def get_action(self, images_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference: Generate an action sequence given multi-camera images.
        Uses Euler ODE solver to integrate the velocity field.
        """
        # 1. Vision Latents (x0)
        vision_latents = self.vision_encoder(images_dict)

        # 2. Flow Sampling (Euler ODE Solver)
        action_latents = self._flow_sample(vision_latents)

        # 3. Decode Action Latents to Actions
        actions_pred = self.action_vae.decode(action_latents)
        return actions_pred

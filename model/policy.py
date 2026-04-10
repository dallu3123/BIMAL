import torch
import torch.nn as nn
import torch.nn.functional as F

from model.vita import SimpleFlowNet
from model.vision import ResNetObserver, MultiCameraObserver
from model.action_vae import MLPActionVAE


class VITAPolicy(nn.Module):
    """
    Top-level VITA Policy integrating Vision Encoder, Action VAE, and Flow Network.
    
    Data flows as follows:
    [Training]
      1. Images -> Vision Encoder -> Vision Latent (x0)
      2. Actions -> Action VAE Encoder -> Action Latent (x1)
      3. Sample t ~ U[0, 1], interpolate xt = (1-t)*x0 + t*x1
      4. FlowNet predicts velocity v(xt, t). Target is (x1 - x0).
      5. Loss = MSE(v(xt, t), x1 - x0) + VAE_Loss
      
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
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_sampling_steps = num_sampling_steps

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

    def compute_loss(self, images_dict: dict[str, torch.Tensor], actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Computes Flow Matching Loss + VAE Loss for training.
        """
        b = actions.shape[0]
        device = actions.device

        # 1. Vision Latents (x0) - Start of the flow
        vision_latents = self.vision_encoder(images_dict)

        # 2. Action Latents (x1) - End of the flow
        x_recon, mu, logvar = self.action_vae(actions)
        action_latents = self.action_vae.reparameterize(mu, logvar)

        # VAE Loss (Reconstruction + KL)
        recon_loss = F.mse_loss(x_recon, actions)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        vae_loss = recon_loss + 1e-4 * kl_loss

        # 3. Flow Matching
        x0 = vision_latents
        x1 = action_latents

        # Sample timestep t uniformly from [0, 1]
        t = torch.rand(b, device=device)
        t_expand = t.view(b, 1)

        # Interpolate between x0 and x1
        xt = (1 - t_expand) * x0 + t_expand * x1

        # Target velocity is a straight line from x0 to x1
        target_v = x1 - x0

        # Predict velocity
        pred_v = self.flow_net(xt, t)

        # Flow Matching Loss
        flow_loss = F.mse_loss(pred_v, target_v)

        # Total Loss
        total_loss = flow_loss + vae_loss

        metrics = {
            "total_loss": total_loss.item(),
            "flow_loss": flow_loss.item(),
            "vae_recon_loss": recon_loss.item(),
            "vae_kl_loss": kl_loss.item(),
        }
        return total_loss, metrics

    @torch.no_grad()
    def get_action(self, images_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Inference: Generate an action sequence given multi-camera images.
        Uses Euler ODE solver to integrate the velocity field.
        """
        b = next(iter(images_dict.values())).shape[0]
        device = next(iter(images_dict.values())).device

        # 1. Vision Latents (x0)
        vision_latents = self.vision_encoder(images_dict)

        # 2. Flow Matching (Euler ODE Solver)
        xt = vision_latents  # Start at x0
        dt = 1.0 / self.num_sampling_steps

        for i in range(self.num_sampling_steps):
            t_val = i / self.num_sampling_steps
            t = torch.full((b,), t_val, device=device)
            
            # Predict velocity at xt, t
            vt = self.flow_net(xt, t)
            
            # Euler integration step
            xt = xt + vt * dt

        action_latents = xt  # This is our predicted x1

        # 3. Decode Action Latents to Actions
        actions_pred = self.action_vae.decode(action_latents)
        return actions_pred

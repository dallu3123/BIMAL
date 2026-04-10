import torch
import torch.nn as nn
import torchvision.models as models


class ResNetObserver(nn.Module):
    """
    ResNet-based Vision Encoder.
    Baseline for VITA, ACT, and Diffusion Policy.
    Extracts spatial features from images using a pre-trained ResNet.
    """

    def __init__(self, resnet_type: str = "resnet18", pretrained: bool = True, out_dim: int = 512) -> None:
        super().__init__()

        # Load pre-trained ResNet
        if resnet_type == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feat_dim = 512
        elif resnet_type == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            feat_dim = 512
        elif resnet_type == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            feat_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")

        # Remove the classification head (avgpool and fc) to get a flattened feature vector
        self.backbone.fc = nn.Identity()

        # Projection layer to map to the desired latent dimension
        self.proj = nn.Linear(feat_dim, out_dim)

        # Standard ImageNet normalization parameters
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (B, C, H, W) in [0, 1] range.
        Returns:
            Latent features of shape (B, out_dim).
        """
        # Normalize
        x = (images - self.mean) / self.std

        # Extract features
        x = self.backbone(x)  # (B, feat_dim)

        # Project to target dimension
        latent = self.proj(x)
        return latent


class DINOv2Observer(nn.Module):
    """
    ViT-based Vision Encoder using frozen DINOv2.
    Provides very strong pre-trained features for robotics without fine-tuning the backbone.
    Highly recommended for state-of-the-art generalization.
    """

    def __init__(self, model_size: str = "vits14", freeze: bool = True, out_dim: int = 512) -> None:
        super().__init__()

        # Available models: dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
        model_name = f"dinov2_{model_size}"

        # Load from PyTorch Hub
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Determine feature dimension based on model size
        if model_size == "vits14":
            feat_dim = 384
        elif model_size == "vitb14":
            feat_dim = 768
        elif model_size == "vitl14":
            feat_dim = 1024
        elif model_size == "vitg14":
            feat_dim = 1536
        else:
            raise ValueError(f"Unknown DINOv2 model size: {model_size}")

        # Projection layer (MLP is commonly used after ViT backbones)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, out_dim)
        )

        # Standard ImageNet normalization (DINOv2 uses the same)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Tensor of shape (B, C, H, W) in [0, 1] range.
                    Images should ideally be resized to multiples of 14 (e.g., 224x224).
        Returns:
            Latent features of shape (B, out_dim).
        """
        # Normalize
        x = (images - self.mean) / self.std

        # If frozen, we don't need to track gradients for the backbone
        if not self.backbone.training or not next(self.backbone.parameters()).requires_grad:
            with torch.no_grad():
                # DINOv2 forward pass (returns the CLS token by default)
                features = self.backbone(x)
        else:
            features = self.backbone(x)

        # Project
        latent = self.proj(features)
        return latent


class MultiCameraObserver(nn.Module):
    """
    Wrapper to handle multiple cameras (e.g., left wrist, right wrist, top-down)
    for Bimanual Manipulation (ALOHA/Piper).
    """

    def __init__(self, encoder: nn.Module, num_cameras: int = 3, feature_dim: int = 512) -> None:
        super().__init__()
        self.encoder = encoder
        self.num_cameras = num_cameras

        # A simple MLP to fuse multi-camera features into a single state latent
        self.fusion = nn.Sequential(
            nn.Linear(num_cameras * feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )

    def forward(self, images_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            images_dict: Dictionary containing image tensors for each camera.
                         e.g., {"cam_top": (B, C, H, W), "cam_left": (B, C, H, W), ...}
        Returns:
            Fused latent representation of shape (B, feature_dim).
        """
        features = []
        # Ensure consistent order of cameras when processing
        for cam_name in sorted(images_dict.keys()):
            img = images_dict[cam_name]
            feat = self.encoder(img)
            features.append(feat)

        # Concatenate features from all cameras
        concat_features = torch.cat(features, dim=-1)  # (B, num_cameras * feature_dim)

        # Fuse
        fused = self.fusion(concat_features)
        return fused

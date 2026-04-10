import math

import torch
import torch.nn as nn


class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Mlp(nn.Module):
    """
    Standard MLP replacing timm.layers.Mlp to keep dependencies minimal.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FlowNetLayer(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()

        def approx_gelu() -> nn.Module:
            return nn.GELU(approximate="tanh")

        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=approx_gelu,
            drop=dropout,
        )

        # Injects timestep t
        self.time_modulator = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * dim))
        self.dim = dim

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.constant_(self.time_modulator[-1].weight, 0)
        nn.init.constant_(self.time_modulator[-1].bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        features = self.time_modulator(t).view(b, 3, self.dim).unbind(1)
        gamma, scale, shift = features

        x_norm = self.norm(x)
        x_norm = x_norm.mul(scale.add(1)).add_(shift)
        x = x + self.mlp(x_norm).mul_(gamma)

        return x


class SimpleFlowNet(nn.Module):
    """
    VITA's Vision-to-Action Flow Network.

    Directly predicts velocity given current latent state 'x' and timestep 't'.
    In VITA, visual latents are x(0) and action latents are x(1).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_embed_dim: int = 256,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmbed(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(time_embed_dim * 4, hidden_dim),
        )

        self.layers = nn.ModuleList(
            [FlowNetLayer(dim=hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Basic initialization
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        t = self.time_embed(t)

        for block in self.layers:
            x = block(x, t)

        x = self.norm(x)
        x = self.out_proj(x)
        return x


class SimpleCondFlowNet(SimpleFlowNet):
    """
    VITA Flow Network extended with an explicit global condition.

    Useful if you want to explicitly condition on proprioception or language.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        time_embed_dim: int = 256,
        condition_dim: int = 512,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
        )

        self.cond_embed = nn.Linear(condition_dim, hidden_dim)

        self._init_weights()

    def forward(self, x: torch.Tensor, t: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        t = self.time_embed(t)

        c = self.cond_embed(global_cond)
        t = t + c  # Inject condition into modulation signal

        for block in self.layers:
            x = block(x, t)

        x = self.norm(x)
        x = self.out_proj(x)
        return x

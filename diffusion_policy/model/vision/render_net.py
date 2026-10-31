"""Render RGB using full LSTM wrench sequences and spatial cross-attention."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(math.gcd(8, out_channels), out_channels), nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(math.gcd(8, out_channels), out_channels),
        )
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, image):
        return F.silu(self.body(image) + self.skip(image))


class BottleneckResidualBlock(nn.Module):
    """Mix channels at 1x1, reserving the spatial convolution for a narrow tensor."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = max(8, out_channels // 2)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 1),
            nn.GroupNorm(math.gcd(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(math.gcd(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 1),
            nn.GroupNorm(math.gcd(8, out_channels), out_channels),
        )
        self.skip = (nn.Identity() if in_channels == out_channels else
                     nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, image):
        return F.silu(self.body(image) + self.skip(image))


class BottleneckDecoderBlock(nn.Module):
    """Reduce channels before resizing; retain the full-resolution skip."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        hidden = max(8, out_channels // 2)
        self.up_projection = nn.Conv2d(in_channels, hidden, 1)
        self.body = nn.Sequential(
            nn.Conv2d(hidden + out_channels, hidden, 1),
            nn.GroupNorm(math.gcd(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(math.gcd(8, hidden), hidden), nn.SiLU(),
            nn.Conv2d(hidden, out_channels, 1),
            nn.GroupNorm(math.gcd(8, out_channels), out_channels),
        )

    def forward(self, features, skip):
        features = F.interpolate(self.up_projection(features), size=skip.shape[-2:],
                                 mode="bilinear", align_corners=False)
        return F.silu(self.body(torch.cat([features, skip], dim=1)) + skip)


class WrenchCrossAttentionBlock(nn.Module):
    """Image queries read every LSTM timestep; residuals retain image features."""

    def __init__(self, embed_dim, conditioning_dim, num_heads, mlp_ratio, dropout,
                 self_attention=True):
        super().__init__()
        self.self_attention = (nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
        ) if self_attention else None)
        self.self_norm = nn.LayerNorm(embed_dim) if self_attention else None
        self.image_norm = nn.LayerNorm(embed_dim)
        self.wrench_norm = nn.LayerNorm(conditioning_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True,
            kdim=conditioning_dim, vdim=conditioning_dim,
        )
        self.output_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim), nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens, wrench_tokens):
        if self.self_attention is not None:
            query = self.self_norm(tokens)
            update, _ = self.self_attention(query, query, query, need_weights=False)
            tokens = tokens + self.dropout(update)
        update, _ = self.cross_attention(
            query=self.image_norm(tokens), key=self.wrench_norm(wrench_tokens),
            value=wrench_tokens, need_weights=False,
        )
        tokens = tokens + self.dropout(update)
        return tokens + self.ffn(self.output_norm(tokens))


class ConvAttentionBackbone(nn.Module):
    """Convolutional image pyramid, low-resolution attention, and skip decoder."""

    def __init__(self, base_channels, conditioning_dim, *, self_attention,
                 patch_size=16, embed_dim=128, depth=4, num_heads=4,
                 mlp_ratio=4.0, dropout=0.0, conv_block="standard"):
        super().__init__()
        if (patch_size < 2 or patch_size & (patch_size - 1) or depth < 1
                or num_heads < 1 or embed_dim < 4 or embed_dim % 4
                or embed_dim % num_heads or base_channels < 1 or mlp_ratio <= 0):
            raise ValueError("Render backbone needs a power-of-two patch_size >= 2, positive "
                             "channels/depth/heads/MLP ratio, and embed_dim divisible by 4 and heads")
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.conv_block = conv_block
        if conv_block not in {"standard", "bottleneck"}:
            raise ValueError("conv_block must be standard or bottleneck")
        block_type = ResidualBlock if conv_block == "standard" else BottleneckResidualBlock
        levels = int(math.log2(patch_size))
        channels = [min(base_channels * 2 ** level, embed_dim) for level in range(levels)]
        channels.append(embed_dim)
        self.feature_dim = channels[0]
        self.stem = ResidualBlock(3, channels[0])
        self.encoders = nn.ModuleList([
            block_type(previous, current) for previous, current in zip(channels, channels[1:])
        ])
        self.blocks = nn.ModuleList([
            WrenchCrossAttentionBlock(embed_dim, conditioning_dim, num_heads, mlp_ratio,
                                      dropout, self_attention=self_attention)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoders = nn.ModuleList([
            (ResidualBlock(channels[level + 1] + channels[level], channels[level])
             if conv_block == "standard" else
             BottleneckDecoderBlock(channels[level + 1], channels[level]))
            for level in reversed(range(levels))
        ])
        self.output = nn.Conv2d(channels[0], 3, 3, padding=1)

    def _position(self, height, width, reference):
        y, x = torch.meshgrid(torch.arange(height, device=reference.device),
                              torch.arange(width, device=reference.device), indexing="ij")
        frequency = 10000.0 ** (-torch.arange(self.embed_dim // 4, device=reference.device).float()
                               / (self.embed_dim // 4))
        x, y = x.flatten()[:, None] * frequency, y.flatten()[:, None] * frequency
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1).to(reference.dtype)

    def forward(self, image, wrench_tokens, camera_embedding):
        height, width = image.shape[-2:]
        p = self.patch_size
        image = F.pad(image, (0, (-width) % p, 0, (-height) % p), mode="replicate")
        features = self.stem(image)
        skips = []
        for encoder in self.encoders:
            skips.append(features)
            features = encoder(F.avg_pool2d(features, 2))
        gh, gw = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        tokens = tokens + self._position(gh, gw, tokens) + camera_embedding.to(tokens)
        for block in self.blocks:
            tokens = block(tokens, wrench_tokens)
        features = self.norm(tokens).transpose(1, 2).reshape(image.shape[0], self.embed_dim, gh, gw)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            if self.conv_block == "bottleneck":
                features = decoder(features, skip)
            else:
                features = decoder(torch.cat([
                    F.interpolate(features, size=skip.shape[-2:], mode="bilinear", align_corners=False),
                    skip,
                ], dim=1))
        features = features[:, :, :height, :width]
        return self.output(features), features


class RenderNet(nn.Module):
    architecture_version = 2

    def __init__(self, wrench_dim, base_channels=16, conditioning_dim=256, residual_scale=0.1,
                 backbone="transformer", transformer=None, high_attention_render=None,
                 architecture_version=2, wrench_history_steps=8, max_obs_steps=2,
                 wrench_layers=1, wrench_dropout=0.0, image_keys=("side", "wrist"),
                 initialize_wrench_from_teacher=True, conv_block="standard"):
        super().__init__()
        if architecture_version != 2:
            raise ValueError("Use build_render_net to load a version 1 checkpoint")
        if not math.isfinite(residual_scale) or residual_scale <= 0:
            raise ValueError("residual_scale must be positive and finite")
        if min(conditioning_dim, wrench_dim, wrench_history_steps, max_obs_steps, wrench_layers) < 1:
            raise ValueError("Wrench dimensions, sequence lengths and LSTM layers must be positive")
        if backbone not in {"unet", "transformer"}:
            raise ValueError("backbone must be 'unet' or 'transformer'")
        self.image_keys = tuple(sorted(str(key) for key in image_keys))
        if not self.image_keys or len(set(self.image_keys)) != len(self.image_keys):
            raise ValueError("image_keys must be nonempty and unique")
        self.residual_scale = float(residual_scale)
        self.backbone_name = str(backbone)
        self.conv_block = conv_block
        self.initialize_wrench_from_teacher = bool(initialize_wrench_from_teacher)
        self.wrench_history_steps = int(wrench_history_steps)
        self.max_obs_steps = int(max_obs_steps)
        self.wrench_projection = nn.Linear(wrench_dim, conditioning_dim)
        self.wrench_position = nn.Parameter(torch.zeros(1, 1, wrench_history_steps, conditioning_dim))
        self.obs_position = nn.Parameter(torch.zeros(1, max_obs_steps, 1, conditioning_dim))
        self.force_modality = nn.Parameter(torch.zeros(1, 1, 1, conditioning_dim))
        self.wrench_encoder = nn.LSTM(
            conditioning_dim, conditioning_dim, num_layers=wrench_layers, batch_first=True,
            dropout=wrench_dropout if wrench_layers > 1 else 0.0,
        )
        self.backbone = ConvAttentionBackbone(
            base_channels, conditioning_dim, self_attention=backbone == "transformer",
            conv_block=conv_block,
            **(transformer or {}),
        )
        self.camera_embedding = nn.Embedding(len(self.image_keys), self.backbone.embed_dim)
        for parameter in (self.wrench_position, self.obs_position, self.force_modality,
                          self.camera_embedding.weight):
            nn.init.normal_(parameter, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.high_attention_enabled = bool((high_attention_render or {}).get("enabled", False))
        self.mask_head = (nn.Conv2d(self.backbone.feature_dim, 1, 1)
                          if self.high_attention_enabled else None)
        self._compiled_backbone_forward = None

    def compile_backbone(self, **kwargs):
        # Compile only the image path. LSTM stays in cuDNN; state_dict keys stay
        # unchanged, and evaluation does not compile a second forward-only graph.
        self._compiled_backbone_forward = torch.compile(self.backbone.forward, **kwargs)

    @property
    def output(self):
        return self.backbone.output

    @torch.no_grad()
    def initialize_wrench_encoder(self, teacher_encoder):
        """Copy the complete teacher force path without keeping a teacher reference."""
        if not isinstance(teacher_encoder.force_encoder, nn.LSTM):
            raise ValueError("Teacher wrench initialization requires an LSTM teacher")
        self.wrench_projection.load_state_dict(teacher_encoder.wrench_projection.state_dict())
        self.wrench_encoder.load_state_dict(teacher_encoder.force_encoder.state_dict())
        for name in ("wrench_position", "obs_position", "force_modality"):
            target, source = getattr(self, name), getattr(teacher_encoder, name)
            if target.shape != source.shape:
                raise ValueError(f"Teacher {name} shape {source.shape} does not match RenderNet {target.shape}")
            target.copy_(source)

    def encode_wrench_history(self, wrench_history):
        """Return [B*To,K,D] LSTM outputs, retaining every history timestep."""
        if wrench_history.ndim != 4:
            raise ValueError("wrench_history must have shape [B, To, K, D]")
        batch, steps, history, dims = wrench_history.shape
        if not 0 < steps <= self.max_obs_steps or history != self.wrench_history_steps:
            raise ValueError("Wrench observation/history lengths do not match RenderNet configuration")
        force = (self.wrench_projection(wrench_history) + self.wrench_position
                 + self.obs_position[:, :steps] + self.force_modality)
        sequence, _ = self.wrench_encoder(force.reshape(batch * steps, history, -1))
        return sequence

    def forward(self, images, wrench_history, *, return_aux=False, use_attention_mask=None):
        """Output RGB only; downstream DINO receives no student token adapter."""
        use_mask = self.high_attention_enabled if use_attention_mask is None else bool(use_attention_mask)
        if use_mask and self.mask_head is None:
            raise ValueError("Attention rendering was disabled during training; no trained mask head exists")
        wrench_tokens = self.encode_wrench_history(wrench_history)
        batch, steps = wrench_history.shape[:2]
        rendered, masks = {}, {}
        for key, image in images.items():
            if key not in self.image_keys:
                raise KeyError(f"Camera {key!r} is absent from configured image_keys {self.image_keys}")
            if image.ndim != 5 or image.shape[:2] != (batch, steps) or image.shape[2] != 3:
                raise ValueError("RenderNet requires matching [B, To, 3, H, W] images")
            camera = self.camera_embedding.weight[self.image_keys.index(key)].reshape(1, 1, -1)
            backbone_forward = (self._compiled_backbone_forward
                                if self._compiled_backbone_forward is not None
                                and self.training and torch.is_grad_enabled() else self.backbone)
            raw_residual, features = backbone_forward(image.flatten(0, 1), wrench_tokens, camera)
            residual = raw_residual.tanh().reshape_as(image)
            if use_mask:
                mask = self.mask_head(features).sigmoid().reshape(batch, steps, 1, *image.shape[-2:])
                masks[key] = mask
                residual = residual * mask
            rendered[key] = (image + self.residual_scale * residual).clamp(0, 1)
        return (rendered, masks) if return_aux else rendered


def build_render_net(wrench_dim, config):
    """Old saved configs have no version; preserve their GRU/FiLM state layout."""
    options = dict(config)
    version = int(options.get("architecture_version", 1))
    if version == 1:
        from diffusion_policy.model.vision.legacy_render_net import LegacyRenderNet

        options.pop("architecture_version", None)
        return LegacyRenderNet(wrench_dim=wrench_dim, **options)
    return RenderNet(wrench_dim=wrench_dim, **options)

"""OceanEmbed network: ViT + FNO satellite-embedding encoder -> U-Net reconstruction decoder.

    surface inputs (C, H, W)
      |-- FNO branch  : spectral convolutions, global large-scale patterns     (width, H, W)
      |-- ViT branch  : patch tokens + self-attention, long-range interactions (dim, H/8, W/8)
      |-- CNN stem    : local features at H, H/2, H/4, H/8 (U-Net skip connections)
      v
    fusion at H/8  -> latent ocean embedding z (latent_dim, H/8, W/8)
      v
    U-Net decoder (upsample + skips) -> temperature (and salinity) at every depth level

`use_vit=False, use_fno=False` gives a plain U-Net, used as the ablation baseline.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
    )


class SpectralConv2d(nn.Module):
    """Fourier layer: multiply the lowest `modes` Fourier coefficients by learned weights."""

    def __init__(self, cin, cout, modes1, modes2):
        super().__init__()
        self.cout, self.m1, self.m2 = cout, modes1, modes2
        scale = 1.0 / (cin * cout)
        self.w1 = nn.Parameter(scale * torch.randn(cin, cout, modes1, modes2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(cin, cout, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        b, _, h, w = x.shape
        xf = torch.fft.rfft2(x.float())
        m1, m2 = min(self.m1, h // 2), min(self.m2, w // 2 + 1)
        out = torch.zeros(b, self.cout, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :m1, :m2], self.w1[:, :, :m1, :m2])
        out[:, :, -m1:, :m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -m1:, :m2], self.w2[:, :, :m1, :m2])
        return torch.fft.irfft2(out, s=(h, w))


class FNOBranch(nn.Module):
    def __init__(self, cin, width=32, modes=12, layers=4):
        super().__init__()
        self.lift = nn.Conv2d(cin, width, 1)
        self.spec = nn.ModuleList([SpectralConv2d(width, width, modes, modes) for _ in range(layers)])
        self.pw = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])

    def forward(self, x):
        x = self.lift(x)
        for s, p in zip(self.spec, self.pw):
            x = F.gelu(s(x) + p(x))
        return x


class ViTBranch(nn.Module):
    def __init__(self, cin, grid_hw, patch=8, dim=128, depth=4, heads=4, dropout=0.1):
        super().__init__()
        self.patch = patch
        self.embed = nn.Conv2d(cin, dim, patch, stride=patch)
        self.grid_hw = grid_hw
        self.pos = nn.Parameter(torch.zeros(1, dim, *grid_hw))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        t = self.embed(x)                                        # (B, dim, h, w)
        b, c, h, w = t.shape
        pos = self.pos if (h, w) == tuple(self.grid_hw) else F.interpolate(
            self.pos, size=(h, w), mode="bilinear", align_corners=False)
        t = (t + pos).flatten(2).transpose(1, 2)                  # (B, h*w, dim)
        t = self.norm(self.encoder(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


class OceanEmbedNet(nn.Module):
    def __init__(self, in_ch, n_depth, grid_hw, out_vars=1, use_vit=True, use_fno=True,
                 base=32, latent_dim=128, vit_dim=128, vit_depth=4, fno_width=32, fno_modes=12):
        super().__init__()
        self.use_vit, self.use_fno = use_vit, use_fno
        self.n_depth, self.out_vars = n_depth, out_vars
        H, W = grid_hw
        self.fno = FNOBranch(in_ch, fno_width, fno_modes) if use_fno else None
        self.vit = ViTBranch(in_ch, (H // 8, W // 8), 8, vit_dim, vit_depth) if use_vit else None

        self.e1 = block(in_ch, base)
        self.e2 = block(base, base * 2)
        self.e3 = block(base * 2, base * 4)
        self.e4 = block(base * 4, base * 4)
        fuse_in = base * 4 + (vit_dim if use_vit else 0) + (fno_width if use_fno else 0)
        self.fuse = nn.Sequential(block(fuse_in, latent_dim), nn.Dropout2d(0.1))

        self.u3 = block(latent_dim + base * 4, base * 4)
        self.u2 = block(base * 4 + base * 2, base * 2)
        self.u1 = block(base * 2 + base + (fno_width if use_fno else 0), base * 2)
        self.head = nn.Conv2d(base * 2, n_depth * out_vars, 1)

    def encode(self, x):
        """Returns the latent ocean embedding z (B, latent_dim, H/8, W/8) and skip features."""
        f = self.fno(x) if self.use_fno else None
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        e4 = self.e4(F.max_pool2d(e3, 2))
        parts = [e4]
        if self.use_vit:
            parts.append(self.vit(x))
        if self.use_fno:
            parts.append(F.avg_pool2d(f, 8))
        z = self.fuse(torch.cat(parts, 1))
        return z, (e1, e2, e3, f)

    def decode(self, z, skips):
        e1, e2, e3, f = skips
        up = lambda a, ref: F.interpolate(a, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.u3(torch.cat([up(z, e3), e3], 1))
        d2 = self.u2(torch.cat([up(d3, e2), e2], 1))
        top = [up(d2, e1), e1] + ([f] if self.use_fno else [])
        d1 = self.u1(torch.cat(top, 1))
        return self.head(d1)                                    # (B, out_vars*D, H, W)

    def forward(self, x):
        z, skips = self.encode(x)
        return self.decode(z, skips)


VARIANTS = {
    "unet": dict(use_vit=False, use_fno=False),          # ablation baseline
    "oceanembed": dict(use_vit=True, use_fno=True),       # full PPT architecture
}


def build_model(variant, in_ch, n_depth, grid_hw, out_vars):
    return OceanEmbedNet(in_ch, n_depth, grid_hw, out_vars, **VARIANTS[variant])

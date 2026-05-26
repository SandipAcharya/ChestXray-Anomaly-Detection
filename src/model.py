"""
CMRF-Net — Corrected PyTorch Implementation
Fixes applied after direct XML audit of Proposed Architecture.docx:
  - LeakyReLU α corrected to 0.2 (doc line 15)
  - DPEU output: 128ch → 256ch (doc Table 1)
  - DPEU branches restructured to match eq. 5 (FA/FB/FC)
  - HAS: 3 RepConv → 4 RepConv blocks (doc eq. 7)
  - HAS input: 128ch → 256ch
  - MSBC: MaxPool → AdaptiveAvgPool (doc eq. 13)
  - SUMM: output 128ch → 256ch (doc Table 1); adds MultiBranchMixer (doc eq. 19)
  - DRC: output 512ch → 256ch (doc Table 1)
  - Detection heads: BatchNorm → GroupNorm (doc eq. 89)
  - All channel dimensions verified against doc Table 1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU_SLOPE = 0.2  # doc line 15: α=0.2


class InputNormalization(nn.Module):
    """Stage 0 — Combined min-max + z-score normalization + GroupNorm(3)."""
    def __init__(self):
        super().__init__()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.gn = nn.GroupNorm(num_groups=3, num_channels=3)
        self.eps = 1e-6

    def forward(self, x):
        xmin = x.flatten(2).min(2)[0].view(x.size(0), x.size(1), 1, 1)
        xmax = x.flatten(2).max(2)[0].view(x.size(0), x.size(1), 1, 1)
        x_mm  = (x - xmin) / (xmax - xmin + self.eps)
        x_zs  = (x - self.mean) / (self.std + self.eps)
        return self.gn(0.5 * (x_mm + x_zs))


class SGEBlock(nn.Module):
    """Stage 1 — Shallow Gradient Extractor. Two 3×3 convs + stride-2 downsampling."""
    def __init__(self, in_ch=3, out_ch=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch // 2)
        self.conv2 = nn.Conv2d(out_ch // 2, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), LRELU_SLOPE)
        return F.leaky_relu(self.bn2(self.conv2(x)), LRELU_SLOPE)


class DPEUBlock(nn.Module):
    """
    Stage 2 — Dual-Path Expansion Unit. Output: 256ch @ 256×256 (doc Table 1).
    Three parallel branches per eq. 5 — each inner activation applied explicitly:
      FA = σ(Conv3×3(x))                            [spatial expansion]
      FB = σ(Conv3×3(σ(Conv1×1(x))))                [bottleneck expansion]
      FC = σ(Conv1×1(σ(Conv3×3(σ(Conv1×1(x))))))   [residual echo]
    Concat FA||FB||FC (320ch) → Conv1×1 → 256ch     (eq. 6)
    """
    def __init__(self, in_ch=64, out_ch=256):
        super().__init__()
        mid = out_ch // 4  # 64

        # FA: single Conv3×3 — spatial expansion (128ch out)
        self.fa_c3  = nn.Sequential(nn.Conv2d(in_ch, mid * 2, 3, padding=1, bias=False), nn.BatchNorm2d(mid * 2))

        # FB: Conv1×1 → [σ] → Conv3×3 — bottleneck expansion (128ch out)
        self.fb_c1  = nn.Sequential(nn.Conv2d(in_ch, mid,     1, bias=False), nn.BatchNorm2d(mid))
        self.fb_c3  = nn.Sequential(nn.Conv2d(mid,   mid * 2, 3, padding=1, bias=False), nn.BatchNorm2d(mid * 2))

        # FC: Conv1×1 → [σ] → Conv3×3 → [σ] → Conv1×1 — residual echo (64ch out)
        self.fc_c1a = nn.Sequential(nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid))
        self.fc_c3  = nn.Sequential(nn.Conv2d(mid,   mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid))
        self.fc_c1b = nn.Sequential(nn.Conv2d(mid,   mid, 1, bias=False), nn.BatchNorm2d(mid))

        # Compress: (128+128+64)=320 → 256ch (eq. 6)
        self.compress = nn.Sequential(
            nn.Conv2d(mid * 2 + mid * 2 + mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        # FA: σ(Conv3×3(x))
        fa = F.leaky_relu(self.fa_c3(x), LRELU_SLOPE)

        # FB: σ(Conv3×3(σ(Conv1×1(x)))) — inner σ applied explicitly
        fb = F.leaky_relu(self.fb_c1(x),  LRELU_SLOPE)
        fb = F.leaky_relu(self.fb_c3(fb), LRELU_SLOPE)

        # FC: σ(Conv1×1(σ(Conv3×3(σ(Conv1×1(x)))))) — all inner σ applied explicitly
        fc = F.leaky_relu(self.fc_c1a(x),  LRELU_SLOPE)
        fc = F.leaky_relu(self.fc_c3(fc),  LRELU_SLOPE)
        fc = F.leaky_relu(self.fc_c1b(fc), LRELU_SLOPE)

        # Concat and compress (eq. 6)
        return F.leaky_relu(self.compress(torch.cat([fa, fb, fc], 1)), LRELU_SLOPE)


class RepConv(nn.Module):
    """Re-parameterizable conv: 3×3 + 1×1 + identity branches."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.c3 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(out_ch)
        self.c1 = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        self.b1 = nn.BatchNorm2d(out_ch)
        self.identity = nn.BatchNorm2d(out_ch) if (in_ch == out_ch and stride == 1) else None

    def forward(self, x):
        out = self.b3(self.c3(x)) + self.b1(self.c1(x))
        if self.identity is not None:
            out = out + self.identity(x)
        return F.leaky_relu(out, LRELU_SLOPE)


class HASBlock(nn.Module):
    """
    Stage 3 — Hierarchical Aggregation Stack. FOUR RepConv blocks (doc eq. 7).
    SKIP-1: concat M2 (256ch) with FHAS(4) (256ch) → 512ch → 1×1 → 256ch.
    """
    def __init__(self, in_ch=256, out_ch=256):
        super().__init__()
        self.rep1 = RepConv(in_ch,  out_ch)
        self.rep2 = RepConv(out_ch, out_ch)
        self.rep3 = RepConv(out_ch, out_ch)
        self.rep4 = RepConv(out_ch, out_ch)
        # SKIP-1 concat: out_ch + in_ch → out_ch
        self.skip_conv = nn.Sequential(
            nn.Conv2d(out_ch + in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x, m2_skip):
        h = self.rep4(self.rep3(self.rep2(self.rep1(x))))
        return F.leaky_relu(self.skip_conv(torch.cat([h, m2_skip], 1)), LRELU_SLOPE)


class DWSepConv(nn.Module):
    """Depthwise Separable Convolution."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bd = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bp = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.leaky_relu(self.bp(self.pw(F.leaky_relu(self.bd(self.dw(x)), LRELU_SLOPE))), LRELU_SLOPE)


class PDEBlock(nn.Module):
    """Stage 4 — Progressive Depth Encoder. Output: 512ch @ 128×128."""
    def __init__(self, in_ch=256, out_ch=512):
        super().__init__()
        self.down  = nn.Conv2d(in_ch, out_ch // 2, 3, stride=2, padding=1, bias=False)
        self.bn    = nn.BatchNorm2d(out_ch // 2)
        self.ch_in = nn.Conv2d(out_ch // 2, out_ch // 2, 1, bias=False)
        self.bi    = nn.BatchNorm2d(out_ch // 2)
        self.dw1   = DWSepConv(out_ch // 2, out_ch // 2)
        self.dw2   = DWSepConv(out_ch // 2, out_ch)
        self.ch_re = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.br    = nn.BatchNorm2d(out_ch)
        self.short = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x):
        identity = self.short(x)
        h = F.leaky_relu(self.bn(self.down(x)), LRELU_SLOPE)
        h = F.leaky_relu(self.bi(self.ch_in(h)), LRELU_SLOPE)
        h = self.dw2(self.dw1(h))
        h = F.leaky_relu(self.br(self.ch_re(h)), LRELU_SLOPE)
        return F.leaky_relu(h + identity, LRELU_SLOPE)


class MSBCBlock(nn.Module):
    """
    Stage 5 — Multi-Scale Bottleneck Cluster. Output: 512ch @ 64×64.
    SPP uses AdaptiveAvgPool to spatial sizes [64,32,16,8] then interpolate back (doc eq. 13).
    SKIP-4: lateral DWConv stride=2 on S4.
    """
    def __init__(self, in_ch=512, out_ch=512):
        super().__init__()
        self.entry = DWSepConv(in_ch, out_ch, stride=2)  # 128→64
        self.pool_sizes = [32, 16, 8]                    # original 64×64 kept as-is
        # Compress 5 branches × out_ch = 5*512=2560 → out_ch
        self.compress = nn.Sequential(
            nn.Conv2d(out_ch * (1 + len(self.pool_sizes) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # SKIP-4 lateral
        self.lateral = DWSepConv(in_ch, out_ch, stride=2)
        self.fuse = DWSepConv(out_ch * 2, out_ch)

    def forward(self, x, s4_skip):
        feat = self.entry(x)                     # 64×64, 512ch
        branches = [feat]
        for sz in self.pool_sizes:
            pooled = F.adaptive_avg_pool2d(feat, (sz, sz))
            branches.append(F.interpolate(pooled, size=feat.shape[2:], mode='bilinear', align_corners=False))
        # lateral SKIP-4
        lat = self.lateral(s4_skip)              # stride-2 on original S4
        branches.append(lat)
        compressed = F.leaky_relu(self.compress(torch.cat(branches, 1)), LRELU_SLOPE)
        return self.fuse(torch.cat([compressed, lat], 1))


class FLRBlock(nn.Module):
    """
    Stage 6 — Feature Lift & Redistribution. Output: 256ch @ 128×128.
    Upsample MSBC (64→128). SKIP-3: downsample H3 (256×256→128×128) and concat.
    """
    def __init__(self, in_ch=512, out_ch=256):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.bu   = nn.BatchNorm2d(out_ch)
        # SKIP-3: stride-2 conv on H3 (256ch @ 256×256 → 256ch @ 128×128)
        self.skip_down = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.compress = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x, h3_skip):
        up   = F.leaky_relu(self.bu(self.up(x)), LRELU_SLOPE)
        skip = F.leaky_relu(self.skip_down(h3_skip), LRELU_SLOPE)
        return F.leaky_relu(self.compress(torch.cat([up, skip], 1)), LRELU_SLOPE)


class MultiBranchMixer(nn.Module):
    """
    Multi-branch mixer with {1×1, 3×3, 5×5 sep, dilated 3×3} (doc eq. 19 / line 65).
    Input: in_ch. Output: in_ch (each branch → in_ch//4).
    """
    def __init__(self, in_ch):
        super().__init__()
        b = in_ch // 4
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, b, 1, bias=False), nn.BatchNorm2d(b))
        self.b2 = nn.Sequential(nn.Conv2d(in_ch, b, 3, padding=1, bias=False), nn.BatchNorm2d(b))
        self.b3 = nn.Sequential(          # 5×5 depthwise sep
            nn.Conv2d(in_ch, in_ch, 5, padding=2, groups=in_ch, bias=False), nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, b, 1, bias=False), nn.BatchNorm2d(b),
        )
        self.b4 = nn.Sequential(nn.Conv2d(in_ch, b, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(b))
        self.proj = nn.Sequential(nn.Conv2d(in_ch, in_ch, 1, bias=False), nn.BatchNorm2d(in_ch))

    def forward(self, x):
        out = torch.cat([
            F.leaky_relu(self.b1(x), LRELU_SLOPE),
            F.leaky_relu(self.b2(x), LRELU_SLOPE),
            F.leaky_relu(self.b3(x), LRELU_SLOPE),
            F.leaky_relu(self.b4(x), LRELU_SLOPE),
        ], 1)
        return F.leaky_relu(self.proj(out), LRELU_SLOPE)


class SUMMBlock(nn.Module):
    """
    Stage 7 — Secondary Upscale & Multi-Branch Mixing. Output: 256ch @ 256×256.
    SKIP-2: element-wise add M2 lateral (doc eq. 19, Table 2).
    """
    def __init__(self, in_ch=256, out_ch=256):
        super().__init__()
        self.up      = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2, bias=False)
        self.bu      = nn.BatchNorm2d(out_ch)
        # SKIP-2 lateral: 1×1 on M2 (256ch)
        self.lateral = nn.Sequential(nn.Conv2d(out_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.mixer   = MultiBranchMixer(out_ch)
        self.out_conv = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False), nn.BatchNorm2d(out_ch))

    def forward(self, x, m2_skip):
        up  = F.leaky_relu(self.bu(self.up(x)), LRELU_SLOPE)
        lat = F.leaky_relu(self.lateral(m2_skip), LRELU_SLOPE)
        mix = self.mixer(up + lat)
        return F.leaky_relu(self.out_conv(mix), LRELU_SLOPE)


class DRBBlock(nn.Module):
    """
    Stage 8 — Downscale Reintegration Block. Output: 256ch @ 128×128.
    SKIP-5: downsample SUMM output, concat with F6 → RepConv.
    """
    def __init__(self, summ_ch=256, f6_ch=256, out_ch=256):
        super().__init__()
        self.down = nn.Conv2d(summ_ch, summ_ch, 3, stride=2, padding=1, bias=False)
        self.bd   = nn.BatchNorm2d(summ_ch)
        self.rep  = RepConv(summ_ch + f6_ch, out_ch)

    def forward(self, summ, f6):
        d = F.leaky_relu(self.bd(self.down(summ)), LRELU_SLOPE)
        return self.rep(torch.cat([d, f6], 1))


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling with rates [1,6,12,18] + global pool."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        def _branch(k, d):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(LRELU_SLOPE)
            )
        self.b1 = _branch(1, 1)
        self.b6 = _branch(3, 6)
        self.b12 = _branch(3, 12)
        self.b18 = _branch(3, 18)
        self.bg = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(LRELU_SLOPE),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 5, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.LeakyReLU(LRELU_SLOPE),
        )

    def forward(self, x):
        bg = F.interpolate(self.bg(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return self.project(torch.cat([self.b1(x), self.b6(x), self.b12(x), self.b18(x), bg], 1))


class DRCBlock(nn.Module):
    """
    Stage 9 — Deep Recombination Cluster. Output: 256ch @ 64×64.
    SKIP-6: concat DRB-down (256ch) with C5 (512ch) → ASPP → 256ch.
    """
    def __init__(self, drb_ch=256, c5_ch=512, out_ch=256):
        super().__init__()
        self.down = nn.Conv2d(drb_ch, drb_ch, 3, stride=2, padding=1, bias=False)
        self.bd   = nn.BatchNorm2d(drb_ch)
        self.proj = nn.Sequential(nn.Conv2d(drb_ch + c5_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.rep  = RepConv(out_ch, out_ch)
        self.aspp = ASPPModule(out_ch, out_ch)

    def forward(self, drb, c5):
        d = F.leaky_relu(self.bd(self.down(drb)), LRELU_SLOPE)
        fused = F.leaky_relu(self.proj(torch.cat([d, c5], 1)), LRELU_SLOPE)
        return self.aspp(self.rep(fused))


class RAOHHead(nn.Module):
    """
    Resolution-Aligned Output Head (doc eq. 89):
    Conv1×1[LeakyReLU(GroupNorm(Conv3×3(F)))]
    """
    def __init__(self, in_ch, num_anchors=3, num_classes=14):
        super().__init__()
        self.A = num_anchors
        self.C = num_classes
        out = num_anchors * (5 + num_classes)
        groups = 8 if in_ch % 8 == 0 else 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, bias=False),
            nn.GroupNorm(groups, in_ch),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv2d(in_ch, out, 1),
        )

    def forward(self, x):
        B, _, H, W = x.shape
        pred = self.conv(x).view(B, self.A, 5 + self.C, H, W)
        return pred.permute(0, 1, 3, 4, 2).contiguous()


class CMRFNet(nn.Module):
    """
    Cascade Multi-Resolution Feature Network (CMRF-Net).
    10-Stage encoder-decoder with 6 strategic skip connections,
    3-scale YOLO-style detection heads, and a global clinical triage head.
    Reference: Diagnostics (MDPI), DOI: 10.3390/diagnostics16010066
    """
    def __init__(self, num_classes=14, num_anchors=3, triage_classes=15):
        super().__init__()
        self.stage0 = InputNormalization()
        self.stage1 = SGEBlock(3, 64)                          # → 64ch @ 256×256
        self.stage2 = DPEUBlock(64, 256)                       # → 256ch @ 256×256
        self.stage3 = HASBlock(256, 256)                       # → 256ch @ 256×256
        self.stage4 = PDEBlock(256, 512)                       # → 512ch @ 128×128
        self.stage5 = MSBCBlock(512, 512)                      # → 512ch @ 64×64
        self.stage6 = FLRBlock(512, 256)                       # → 256ch @ 128×128
        self.stage7 = SUMMBlock(256, 256)                      # → 256ch @ 256×256
        self.stage8 = DRBBlock(256, 256, 256)                  # → 256ch @ 128×128
        self.stage9 = DRCBlock(256, 512, 256)                  # → 256ch @ 64×64

        # Three RAOH detection heads (doc Stage 10)
        self.head_high = RAOHHead(256, num_anchors, num_classes)  # 256×256
        self.head_med  = RAOHHead(256, num_anchors, num_classes)  # 128×128
        self.head_low  = RAOHHead(256, num_anchors, num_classes)  # 64×64

        # Global Triage Head: σ(MLP(GAP(F9))) — F9=DRC output, 256ch
        self.global_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, triage_classes),
        )

    def forward(self, x):
        x0 = self.stage0(x)                        # [B,  3, 512, 512]
        x1 = self.stage1(x0)                       # [B, 64, 256, 256]
        m2 = self.stage2(x1)                       # [B,256, 256, 256]
        h3 = self.stage3(m2, m2)                   # [B,256, 256, 256]  SKIP-1
        s4 = self.stage4(h3)                       # [B,512, 128, 128]
        c5 = self.stage5(s4, s4)                   # [B,512,  64,  64]  SKIP-4
        f6 = self.stage6(c5, h3)                   # [B,256, 128, 128]  SKIP-3
        r7 = self.stage7(f6, m2)                   # [B,256, 256, 256]  SKIP-2
        d8 = self.stage8(r7, f6)                   # [B,256, 128, 128]  SKIP-5
        z9 = self.stage9(d8, c5)                   # [B,256,  64,  64]  SKIP-6

        pred_high = self.head_high(r7)             # [B,A,256,256,5+C]
        pred_med  = self.head_med(d8)              # [B,A,128,128,5+C]
        pred_low  = self.head_low(z9)              # [B,A, 64, 64,5+C]
        triage    = torch.sigmoid(self.global_cls(z9))

        return pred_high, pred_med, pred_low, triage


if __name__ == "__main__":
    model = CMRFNet(num_classes=14, num_anchors=3, triage_classes=15).eval()
    x = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        ph, pm, pl, triage = model(x)
    print("✅ CMRF-Net forward pass successful")
    print(f"  High  head : {ph.shape}")    # [2,3,256,256,19]
    print(f"  Med   head : {pm.shape}")    # [2,3,128,128,19]
    print(f"  Low   head : {pl.shape}")    # [2,3, 64, 64,19]
    print(f"  Triage     : {triage.shape}")
    print(f"  Params     : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, num_groups=8):
        super().__init__()

        # ensure num_groups divides out_channels
        if out_channels % num_groups != 0:
            num_groups = 1  # fallback to LayerNorm-like behavior

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.dropout = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    def forward(self, x):
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.norm2(self.conv2(x)))
        return x



class UNet(nn.Module):
    """
    U-Net architecture for image-to-image prediction.
    Encoder-decoder structure with skip connections and upsampling.
    """
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        # Encoder path
        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)

        # Decoder path
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = ConvBlock(1024 + 512, 512)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = ConvBlock(512 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = ConvBlock(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = ConvBlock(128 + 64, 64)

        # Final output layer (1x1 convolution)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def center_crop(self, enc_feat, target_feat):
        _, _, h, w = target_feat.shape
        return TF.center_crop(enc_feat, [h, w])

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)
        p1 = self.pool1(c1)
        c2 = self.enc2(p1)
        p2 = self.pool2(c2)
        c3 = self.enc3(p2)
        p3 = self.pool3(c3)
        c4 = self.enc4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Decoder with skip connections
        u4 = self.up4(bn)
        c4 = self.center_crop(c4, u4)
        c5 = self.dec4(torch.cat([u4, c4], dim=1))
        u3 = self.up3(c5)
        c3 = self.center_crop(c3, u3)
        c6 = self.dec3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(c6)
        c2 = self.center_crop(c2, u2)
        c7 = self.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(c7)
        c1 = self.center_crop(c1, u1)
        c8 = self.dec1(torch.cat([u1, c1], dim=1))

        # Output
        return self.final(c8)





class CompositeLogitHuberTVLoss(nn.Module):
    def __init__(self, bin_edges, bin_weights, alpha=0.6, delta=0.3, eps=1e-4,
                 tv_weight=0, bias_penalty = 0.15):
            
        super().__init__()
        self.alpha = alpha
        self.delta = delta
        self.eps = eps
        self.tv_weight = tv_weight
        self.bias_penalty = bias_penalty
        
        self.register_buffer("bin_edges", torch.as_tensor(bin_edges, dtype=torch.float32))
        self.register_buffer("bin_weights", torch.as_tensor(bin_weights, dtype=torch.float32))
        

    @staticmethod
    def _tv_loss(y_prob, mask):
        if y_prob.dim() == 3:
            y_prob = y_prob.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        dy = (y_prob[:, :, 1:, :] - y_prob[:, :, :-1, :]).abs()
        mdy = (mask[:, :, 1:, :] * mask[:, :, :-1, :])
        dy = (dy * mdy).sum()
        denom_y = mdy.sum().clamp_min(1.0)

        dx = (y_prob[:, :, :, 1:] - y_prob[:, :, :, :-1]).abs()
        mdx = (mask[:, :, :, 1:] * mask[:, :, :, :-1])
        dx = (dx * mdx).sum()
        denom_x = mdx.sum().clamp_min(1.0)

        return 0.5 * (dy / denom_y + dx / denom_x)

    def forward(self, y_logits, y_true):
        if y_logits.dim() == 3:
            y_logits = y_logits.unsqueeze(1)
        if y_true.dim() == 3:
            y_true = y_true.unsqueeze(1)

        mask = (y_true != -1).float()
        y_true_clean = torch.where(mask > 0, y_true, torch.zeros_like(y_true))

        # ---- bin weights ----
        y_clip = torch.clamp(y_true_clean, 0.0, 1.0)
        idx = torch.bucketize(y_clip, self.bin_edges[1:-1], right=False)
        w = self.bin_weights[idx].detach()
        w_mask = w * mask
        denom = w_mask.sum().clamp_min(1.0)

        # ---- logit MSE ----
        y_eps = torch.clamp(y_clip, self.eps, 1.0 - self.eps)
        z_true = torch.log(y_eps) - torch.log(1.0 - y_eps)
        logit_mse = ((y_logits - z_true) ** 2 * w_mask).sum() / denom

        # ---- huber in prob space ----
        y_prob = torch.sigmoid(y_logits)
        diff = y_prob - y_true_clean
        abs_diff = diff.abs()

        quad = 0.5 * diff ** 2
        lin = self.delta * (abs_diff - 0.5 * self.delta)
        huber = torch.where(abs_diff <= self.delta, quad, lin)

        huber_loss = (huber * w_mask).sum() / denom

        tv = self._tv_loss(y_prob, mask)
        
        # Empirically grounded bias correction derived from observed per-bin bias profile:
        # 0.2-0.4: bias=-0.079 (under-pred) → push UP   | weight 1.0
        # 0.4-0.6: bias=+0.041 (over-pred)  → push DOWN | weight 0.5
        # 0.6-0.8: bias=+0.159 (over-pred)  → push DOWN | weight 2.0 (strongest)
        bin_02_04 = ((y_true >= 0.2) & (y_true < 0.4) & (mask > 0)).float()
        bin_04_06 = ((y_true >= 0.4) & (y_true < 0.6) & (mask > 0)).float()
        bin_06_08 = ((y_true >= 0.6) & (y_true < 0.8) & (mask > 0)).float()

        over_pred  = torch.clamp(y_prob - y_true_clean, min=0.0)
        under_pred = torch.clamp(y_true_clean - y_prob, min=0.0)

        bias_loss = (under_pred ** 2 * bin_02_04).sum() / bin_02_04.sum().clamp_min(1.0) * 1.0 \
                  + (over_pred  ** 2 * bin_04_06).sum() / bin_04_06.sum().clamp_min(1.0) * 0.5 \
                  + (over_pred  ** 2 * bin_06_08).sum() / bin_06_08.sum().clamp_min(1.0) * 2.0
        
        return self.alpha * logit_mse + (1.0 - self.alpha) * huber_loss \
             + self.tv_weight * tv + self.bias_penalty * bias_loss



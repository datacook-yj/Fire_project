"""
산불 예측 최종본 (6채널, Ablation Study 기준)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader


# ═══════════════════════════════════════════════════════
# 손실 함수
# ═══════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce)
        a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).view(-1)
        t = targets.view(-1)
        return 1.0 - (2.0*(p*t).sum()+self.smooth) / (p.sum()+t.sum()+self.smooth)


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss(alpha=0.75, gamma=2.0)
        self.dice  = DiceLoss(smooth=1.0)

    def forward(self, logits, targets):
        return 0.5*self.focal(logits, targets) + 0.5*self.dice(logits, targets)


# ═══════════════════════════════════════════════════════
# 모델 구성 블록
# ═══════════════════════════════════════════════════════

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x.mean(dim=[2, 3])).unsqueeze(-1).unsqueeze(-1)


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.block(x)


# ═══════════════════════════════════════════════════════
# U-Net 6채널 모델
# ═══════════════════════════════════════════════════════

class UNet_6Ch(nn.Module):
    """
    입력 채널: [화점, 기온, 습도, U풍속, V풍속, DEM고도]
    출력: (N, 1, 64, 64) logits
    """
    def __init__(self, in_channels: int = 6, base_ch: int = 32):
        super().__init__()
        self.enc1  = DoubleConv(in_channels, base_ch);  self.se1  = SEBlock(base_ch)
        self.enc2  = DoubleConv(base_ch, base_ch*2);    self.se2  = SEBlock(base_ch*2)
        self.pool1 = nn.MaxPool2d(2);                   self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_ch*2, base_ch*4)
        self.up2  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = DoubleConv(base_ch*4+base_ch*2, base_ch*2)
        self.up1  = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = DoubleConv(base_ch*2+base_ch, base_ch)
        self.final = nn.Conv2d(base_ch, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.se1(self.enc1(x))
        e2 = self.se2(self.enc2(self.pool1(e1)))
        b  = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b),  e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


# ═══════════════════════════════════════════════════════
# 데이터 증강
# ═══════════════════════════════════════════════════════

def augment_data(X: torch.Tensor, Y: torch.Tensor):
    X_aug = [X, torch.flip(X,[3]), torch.flip(X,[2]), torch.rot90(X,k=1,dims=[2,3])]
    Y_aug = [Y, torch.flip(Y,[3]), torch.flip(Y,[2]), torch.rot90(Y,k=1,dims=[2,3])]
    return torch.cat(X_aug, dim=0), torch.cat(Y_aug, dim=0)


# ═══════════════════════════════════════════════════════
# 학습 진입점
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = "/content"
    print(f"디바이스: {device}")

    X_np = np.load(os.path.join(DATA_DIR, 'X_train_final.npy'))
    Y_np = np.load(os.path.join(DATA_DIR, 'Y_train_final.npy'))
    print(f"원본 데이터: X={X_np.shape}, Y={Y_np.shape}")

    X_aug, Y_aug = augment_data(
        torch.tensor(X_np).float(),
        torch.tensor(Y_np).float()
    )
    print(f"증강 후  : X={X_aug.shape}, Y={Y_aug.shape}")

    loader = DataLoader(
        TensorDataset(X_aug.to(device), Y_aug.to(device)),
        batch_size=4, shuffle=True
    )

    model     = UNet_6Ch(in_channels=6, base_ch=32).to(device)
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-5)
    print(f"파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # ── Early Stopping 제거, best_loss 저장만 유지 ───────
    # 이유: model_5ch.py와 동일 epoch 수(200)로 공정한 Ablation Study 비교
    best_loss = float('inf')
    epochs    = 200
    print("\n[6ch] 학습 시작 (Early Stopping 없음, 200 epoch 완주)...")
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Best':>8} | {'LR':>10}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward(); optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(),
                       os.path.join(DATA_DIR, 'unet_best_model.pth'))

        if epoch % 20 == 0:
            print(f"{epoch:>6} | {avg_loss:>8.4f} | {best_loss:>8.4f} | "
                  f"{optimizer.param_groups[0]['lr']:>10.6f}")

    print(f"\n[6ch] 학습 완료! 최소 loss={best_loss:.4f}")
    save_path = os.path.join(DATA_DIR, 'unet_best_model.pth')
    status = "OK" if os.path.exists(save_path) else "없음"
    print(f"  [{status}] {save_path}")
"""Train and export a tiny lane segmentation model for Webots controller usage.

This script uses existing training batches (batch_*.npz) and auto-generates lane masks
from color heuristics (yellow + white lane markings). The resulting model is exported
as TorchScript and can be loaded by the controller via lane_seg_model_path.
"""

import argparse
import glob
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from config import DATA_CONFIG


def _build_lane_mask(camera_rgb: np.ndarray) -> np.ndarray:
    """Build a binary lane mask from a normalized RGB image (64x64x3, range [0,1])."""
    img = np.asarray(camera_rgb, dtype=np.float32)
    if img.shape != (64, 64, 3):
        return np.zeros((64, 64), dtype=np.float32)

    rgb = np.clip(img * 255.0, 0.0, 255.0)

    max_ch = np.max(rgb, axis=2)
    min_ch = np.min(rgb, axis=2)
    white_mask = (max_ch > 150.0) & ((max_ch - min_ch) < 35.0)

    yellow_mask = (rgb[:, :, 0] > 150.0) & (rgb[:, :, 1] > 120.0) & (rgb[:, :, 2] < 130.0)

    ref_yellow = np.array([203.0, 187.0, 95.0], dtype=np.float32)
    color_diff = np.sum(np.abs(rgb - ref_yellow), axis=2)
    yellow_ref_mask = color_diff < 95.0

    lane_mask = white_mask | yellow_mask | yellow_ref_mask

    # Keep strongest signal in lower image region where lane markings are expected.
    horizon_row = 18
    lane_mask[:horizon_row, :] = False

    return lane_mask.astype(np.float32)


class LaneSegDataset(Dataset):
    """Dataset of (camera, lane_mask) pairs."""

    def __init__(self, cameras: np.ndarray):
        # Keep memory low by storing half precision and casting at access time.
        self.cameras = cameras.astype(np.float16)
        self.masks = np.stack([_build_lane_mask(c) for c in self.cameras], axis=0).astype(np.float32)

    def __len__(self) -> int:
        return int(self.cameras.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.cameras[idx].astype(np.float32)
        mask = self.masks[idx]

        x = torch.from_numpy(img).permute(2, 0, 1)  # (3, 64, 64)
        y = torch.from_numpy(mask).unsqueeze(0)  # (1, 64, 64)
        return x, y


class TinyLaneSegNet(nn.Module):
    """Very small encoder-decoder segmentation model for 64x64 lane masks."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        b = self.bottleneck(p2)

        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))

        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))

        return self.head(d2)  # logits


def _dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = torch.sum(prob * target, dim=(1, 2, 3))
    union = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def _load_all_cameras(data_dir: str, max_samples: int = 12000, seed: int = 42) -> np.ndarray:
    batch_files = sorted(glob.glob(os.path.join(data_dir, "batch_*.npz")))
    reservoir: List[np.ndarray] = []
    seen = 0
    rng = np.random.default_rng(seed)

    for batch_file in batch_files:
        try:
            data = np.load(batch_file)
            if "camera" not in data:
                continue
            cam = np.asarray(data["camera"], dtype=np.float32)
            if cam.ndim == 4 and cam.shape[1:] == (64, 64, 3):
                if float(np.max(cam)) > 1.5:
                    cam = cam / 255.0
                cam = np.clip(cam, 0.0, 1.0)

                for sample in cam:
                    seen += 1
                    if len(reservoir) < max_samples:
                        reservoir.append(sample.copy())
                    else:
                        j = int(rng.integers(0, seen))
                        if j < max_samples:
                            reservoir[j] = sample.copy()
        except Exception:
            continue

    if not reservoir:
        raise ValueError(f"No usable camera data found in {data_dir}")

    all_cam = np.stack(reservoir, axis=0)
    print(f"[LaneSeg] Selected {len(all_cam)} samples from {seen} total frames")
    return all_cam


def train_and_export(
    data_dir: str,
    output_ts: str,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.15,
    max_samples: int = 12000,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LaneSeg] Device: {device}")

    cameras = _load_all_cameras(data_dir, max_samples=max_samples)
    dataset = LaneSegDataset(cameras)

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = TinyLaneSegNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = 0.6 * bce(logits, y) + 0.4 * _dice_loss_from_logits(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = 0.6 * bce(logits, y) + 0.4 * _dice_loss_from_logits(logits, y)
                val_loss += float(loss.item())
        val_loss /= max(1, len(val_loader))

        print(f"[LaneSeg] Epoch {epoch + 1}/{epochs} train={train_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    os.makedirs(os.path.dirname(output_ts), exist_ok=True)

    example = torch.randn(1, 3, 64, 64, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(output_ts)
    print(f"[LaneSeg] Exported TorchScript model: {output_ts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tiny lane segmentation model and export TorchScript")
    parser.add_argument("--data-dir", default=DATA_CONFIG["data_dir"], help="Directory containing batch_*.npz")
    parser.add_argument(
        "--output",
        default=os.path.join(DATA_CONFIG["models_dir"], "lane_segmentation_tiny.ts"),
        help="Output TorchScript path",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=12000, help="Max camera frames sampled for training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_export(
        data_dir=args.data_dir,
        output_ts=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()

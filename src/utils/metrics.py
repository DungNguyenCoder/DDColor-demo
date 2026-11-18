"""Độ đo cho bài toán tô màu: PSNR, độ sặc sỡ CF (Hasler & Süsstrunk 2003), ΔCF; FID tính ở evaluation/fid.py."""
import numpy as np


def psnr(pred_u8, gt_u8):
    d = pred_u8.astype(np.float64) - gt_u8.astype(np.float64)
    mse = np.mean(d * d)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def colorfulness(rgb_u8):
    """
    CF = σ_rgyb + 0,3 · μ_rgyb, với rg = R − G, yb = 0,5(R + G) − B (thang 0–255),
    σ_rgyb = √(σ_rg² + σ_yb²), μ_rgyb = √(μ_rg² + μ_yb²).
    Ảnh xám có CF = 0.
    """
    x = rgb_u8.astype(np.float64)
    rg = x[..., 0] - x[..., 1]
    yb = 0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))

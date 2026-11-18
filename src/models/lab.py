"""
Chuyển đổi sRGB <-> CIELAB (chuẩn trắng D65) tự cài đặt bằng NumPy.

Các bước (Chương 3 – mô hình màu):
  sRGB [0,1] --(bỏ gamma sRGB)--> RGB tuyến tính --(ma trận 3x3)--> XYZ --(hàm f)--> L*, a*, b*
L* thuộc [0, 100] là độ sáng cảm nhận; a* (lục–đỏ) và b* (lam–vàng) là hai kênh màu.
"""
import numpy as np

# Ma trận sRGB (tuyến tính) -> XYZ, chuẩn trắng D65 (IEC 61966-2-1)
M_RGB2XYZ = np.array([[0.412453, 0.357580, 0.180423],
                      [0.212671, 0.715160, 0.072169],
                      [0.019334, 0.119193, 0.950227]])
M_XYZ2RGB = np.linalg.inv(M_RGB2XYZ)
WHITE_D65 = M_RGB2XYZ @ np.ones(3)          # (0.950456, 1.0, 1.088754)
_EPS = (6 / 29) ** 3                         # 0.008856
_KAPPA = (29 / 6) ** 2 / 3                   # 7.787


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def _f(t):
    return np.where(t > _EPS, np.cbrt(t), _KAPPA * t + 16 / 116)


def _f_inv(t):
    return np.where(t > 6 / 29, t ** 3, (t - 16 / 116) / _KAPPA)


def rgb_to_lab(rgb):
    """rgb: HxWx3 trong [0,1] (sRGB). Trả về HxWx3 (L, a, b)."""
    xyz = srgb_to_linear(rgb) @ M_RGB2XYZ.T / WHITE_D65
    fx, fy, fz = _f(xyz[..., 0]), _f(xyz[..., 1]), _f(xyz[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def lab_to_rgb(lab):
    """lab: HxWx3. Trả về sRGB trong [0,1] (cắt ngoài gam màu)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200
    xyz = np.stack([_f_inv(fx), _f_inv(fy), _f_inv(fz)], axis=-1) * WHITE_D65
    return linear_to_srgb(xyz @ M_XYZ2RGB.T)


def to_gray_lab(rgb):
    """Ảnh xám dùng làm đầu vào: giữ L, đặt a = b = 0, đổi về RGB."""
    lab = rgb_to_lab(rgb)
    lab[..., 1:] = 0.0
    return lab_to_rgb(lab)

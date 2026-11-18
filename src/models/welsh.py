"""
Tô màu theo ảnh mẫu – Welsh, Ashikhmin, Mueller (SIGGRAPH 2002), tự cài đặt bằng NumPy.

Ý tưởng: điểm ảnh xám lấy màu (a, b) của điểm trong ảnh mẫu có "độ sáng + kết cấu" giống nhất.
  1. Đổi ảnh mẫu sang Lab; ảnh đích chỉ có L.
  2. Ánh xạ lại độ sáng của ảnh mẫu cho cùng trung bình, độ lệch chuẩn với ảnh đích (giống khớp histogram bậc 2).
  3. Đặc trưng mỗi điểm = 0,5 · L + 0,5 · độ lệch chuẩn của L trong cửa sổ 5 × 5.
  4. Lấy khoảng 200 điểm mẫu trên lưới có nhiễu (jittered grid) của ảnh mẫu.
  5. Với mỗi điểm ảnh đích, tìm điểm mẫu có đặc trưng gần nhất, chép (a, b), giữ nguyên L của ảnh đích.
Vì đặc trưng là một số thực, tìm láng giềng gần nhất bằng sắp xếp + tìm kiếm nhị phân.
"""
import numpy as np
from .lab import rgb_to_lab, lab_to_rgb


def box_mean(x, k=5):
    """Trung bình trong cửa sổ k x k bằng ảnh tích phân, đệm phản xạ ở mép."""
    r = k // 2
    p = np.pad(x, r, mode="reflect")
    s = np.cumsum(np.cumsum(p, 0), 1)
    s = np.pad(s, ((1, 0), (1, 0)))
    h, w = x.shape
    tot = s[k:k + h, k:k + w] - s[0:h, k:k + w] - s[k:k + h, 0:w] + s[0:h, 0:w]
    return tot / (k * k)


def local_std(x, k=5):
    m = box_mean(x, k)
    return np.sqrt(np.maximum(box_mean(x * x, k) - m * m, 0.0))


def jittered_samples(h, w, n=200, rng=None):
    rng = np.random.default_rng(0) if rng is None else rng
    g = int(np.ceil(np.sqrt(n * w / h)))
    gy = int(np.ceil(n / g))
    ys = ((np.arange(gy)[:, None] + rng.random((gy, g))) * h / gy).astype(int).ravel()
    xs = ((np.arange(g)[None, :] + rng.random((gy, g))) * w / g).astype(int).ravel()
    return np.clip(ys, 0, h - 1), np.clip(xs, 0, w - 1)


def colorize(gray_L, ref_rgb, n_samples=200, k=5, seed=0):
    """
    gray_L : HxW, độ sáng L* của ảnh đích (0..100).
    ref_rgb: HrxWrx3 ảnh mẫu màu trong [0,1].
    Trả về ảnh RGB [0,1] cùng kích thước ảnh đích.
    """
    ref_lab = rgb_to_lab(ref_rgb)
    Lr = ref_lab[..., 0]
    # 2. ánh xạ độ sáng
    mt, st = gray_L.mean(), gray_L.std() + 1e-6
    mr, sr = Lr.mean(), Lr.std() + 1e-6
    Lr_m = (Lr - mr) * (st / sr) + mt
    # 3. đặc trưng
    ft = 0.5 * gray_L + 0.5 * local_std(gray_L, k)
    fr = 0.5 * Lr_m + 0.5 * local_std(Lr_m, k)
    # 4. lấy mẫu
    ys, xs = jittered_samples(*Lr.shape, n=n_samples, rng=np.random.default_rng(seed))
    feat = fr[ys, xs]; ab = ref_lab[ys, xs, 1:]
    order = np.argsort(feat); feat = feat[order]; ab = ab[order]
    # 5. láng giềng gần nhất trên trục đặc trưng
    idx = np.searchsorted(feat, ft.ravel())
    idx = np.clip(idx, 1, len(feat) - 1)
    left, right = feat[idx - 1], feat[idx]
    idx = idx - (ft.ravel() - left < right - ft.ravel())
    out = np.empty(gray_L.shape + (3,))
    out[..., 0] = gray_L
    out[..., 1:] = ab[idx].reshape(gray_L.shape + (2,))
    return lab_to_rgb(out)


# ----------------------------------------------------------------------------
# Chọn ảnh mẫu tự động: ảnh trong kho có ảnh xám giống nhất
# ----------------------------------------------------------------------------
def gray_descriptor(L, size=32, bins=16):
    """Đặc trưng toàn cục của ảnh xám: ảnh thu nhỏ 32x32 chuẩn hóa + histogram 16 mức."""
    h, w = L.shape
    ys = (np.arange(size) * h / size).astype(int); xs = (np.arange(size) * w / size).astype(int)
    th = L[ys][:, xs].ravel()
    th = (th - th.mean()) / (th.std() + 1e-6)
    hist = np.histogram(L, bins=bins, range=(0, 100))[0].astype(np.float64)
    hist = hist / hist.sum()
    v = np.concatenate([th / np.sqrt(th.size), 2.0 * hist])
    return v / (np.linalg.norm(v) + 1e-9)

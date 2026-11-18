"""
Xử lý ảnh giả màu (Chương 3, mục 3.3 của học phần), tự cài đặt bằng NumPy.
  - Cắt cường độ sáng và mã hóa màu (intensity slicing): chia [0, 255] thành k khoảng, mỗi khoảng một màu.
  - Biến đổi cường độ sáng thành màu: ba hàm biến đổi độc lập cho ba kênh R, G, B,
    ở đây dùng hàm sin lệch pha như ví dụ trong Gonzalez & Woods (mục 6.3):
        R = |sin(2π f r + φ_R)|, G = |sin(2π f r + φ_G)|, B = |sin(2π f r + φ_B)|,  r ∈ [0, 1].
Giả màu nhằm làm nổi bật mức xám cho người xem, không nhằm khôi phục màu thật.
"""
import numpy as np


def intensity_slicing(gray_u8, levels=None, colors=None):
    levels = [64, 128, 192] if levels is None else levels
    colors = np.array([[0, 0, 160], [0, 170, 0], [230, 200, 0], [220, 30, 30]] if colors is None else colors, np.uint8)
    idx = np.digitize(gray_u8, levels)
    return colors[idx]


def sine_transform(gray_u8, f=1.0, phases=(0.0, np.pi / 3, 2 * np.pi / 3)):
    r = gray_u8.astype(np.float64) / 255.0
    ch = [np.abs(np.sin(np.pi * f * r + p)) for p in phases]
    return np.clip(np.rint(np.stack(ch, -1) * 255), 0, 255).astype(np.uint8)

"""Đọc ảnh đầu vào cho giao diện và kiểm thử: xoay theo EXIF, bỏ kênh alpha, đổi ảnh 16 bit, giới hạn kích thước."""
import numpy as np
from PIL import Image, ImageOps

MAX_SIDE = 2048


def read_image(f, max_side=MAX_SIDE):
    """
    f: đường dẫn hoặc đối tượng file (ví dụ file tải lên từ Streamlit).
    Trả về (ảnh RGB uint8 H×W×3, kích thước gốc (W, H)).
    """
    im = Image.open(f)
    im = ImageOps.exif_transpose(im)                      # ảnh điện thoại: xoay đúng chiều theo EXIF
    orig = im.size
    if im.mode.startswith("I;16"):                        # ảnh xám 16 bit: 0..65535 -> 0..255, giữ độ sáng
        im = Image.fromarray(np.rint(np.asarray(im, dtype=np.float64) / 257).astype(np.uint8))
    elif im.mode in ("I", "F"):                           # số nguyên 32 bit / số thực: co dải giá trị về 0..255
        a = np.asarray(im, dtype=np.float64)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) * 255 if hi > lo else np.zeros_like(a)
        im = Image.fromarray(np.rint(a).astype(np.uint8))
    if im.mode in ("RGBA", "LA", "P", "PA"):
        im = im.convert("RGBA")                           # đặt lên nền trắng
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    im = im.convert("RGB")
    if max(im.size) > max_side:                           # ảnh quá lớn: thu nhỏ để không tràn bộ nhớ
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.LANCZOS)
    return np.asarray(im), orig

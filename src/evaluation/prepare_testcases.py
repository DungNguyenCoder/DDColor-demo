"""
Tạo bộ ảnh test case (data/testcases):
  color/  ảnh màu có ảnh chuẩn (chuyển xám để thử, so với bản gốc)
  gray/   ảnh xám thật, không có ảnh chuẩn (ảnh chụp đen trắng, tài liệu, ảnh hiển vi...)
  func/   ảnh cho kiểm thử chức năng (ảnh lớn, ảnh nhỏ, kích thước lẻ, PNG có alpha, JPG, ảnh 1 kênh)
Nguồn: bộ ảnh mẫu của scikit-image (giấy phép trong skimage/data/README) và COCO val2017.
"""
import os, sys
import numpy as np
from PIL import Image
import skimage.data as sd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
T = os.path.join(ROOT, "data", "testcases")


def save(img, sub, name):
    os.makedirs(os.path.join(T, sub), exist_ok=True)
    Image.fromarray(img).save(os.path.join(T, sub, name))


def main():
    for n in ["astronaut", "coffee", "chelsea", "rocket", "immunohistochemistry", "hubble_deep_field", "colorwheel"]:
        save(getattr(sd, n)(), "color", n + ".png")
    for n in ["camera", "moon", "coins", "page", "text", "clock", "cell"]:
        save(getattr(sd, n)(), "gray", n + ".png")
    base = sd.coffee()
    save(np.asarray(Image.fromarray(base).resize((4000, 3000), Image.BICUBIC)), "func", "large_4000x3000.png")
    save(np.asarray(Image.fromarray(base).resize((32, 32), Image.BICUBIC)), "func", "tiny_32x32.png")
    save(np.asarray(Image.fromarray(base).resize((401, 299), Image.BICUBIC)), "func", "odd_401x299.png")
    save(sd.logo(), "func", "rgba_logo.png")
    Image.fromarray(base).save(os.path.join(T, "func", "photo.jpg"), quality=85)
    save(sd.camera(), "func", "single_channel.png")
    print("Đã tạo", sum(len(f) for _, _, f in os.walk(T)), "ảnh")


if __name__ == "__main__":
    main()

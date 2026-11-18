"""
Chuẩn bị tập kiểm tra từ COCO val2017 (5.000 ảnh, cũng là tập kiểm tra COCO-Stuff trong bài DDColor).
  data/raw/val2017.zip  ->  data/coco256/gt/<id>.png   (cạnh ngắn 256, cắt giữa 256 x 256, PNG)
Ghi thêm data/coco256/meta.csv: tên ảnh, kích thước gốc, ảnh gốc có phải ảnh xám không.
"""
import os, sys, zipfile, io
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SIZE = 256


def center_crop_resize(img, size=SIZE):
    w, h = img.size
    s = size / min(w, h)
    img = img.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    w, h = img.size
    l, t = (w - size) // 2, (h - size) // 2
    return img.crop((l, t, l + size, t + size))


def main():
    out = os.path.join(ROOT, "data", "coco256", "gt"); os.makedirs(out, exist_ok=True)
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "raw", "val2017.zip"))
    names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
    rows = []
    for n in names:
        im = Image.open(io.BytesIO(z.read(n)))
        mode = im.mode
        rgb = im.convert("RGB")
        a = np.asarray(rgb).astype(int)
        is_gray = mode == "L" or (np.abs(a[..., 0] - a[..., 1]).max() <= 2 and np.abs(a[..., 1] - a[..., 2]).max() <= 2)
        stem = os.path.splitext(os.path.basename(n))[0]
        center_crop_resize(rgb).save(os.path.join(out, stem + ".png"))
        rows.append({"image": stem, "w": im.size[0], "h": im.size[1], "mode": mode, "gray_origin": bool(is_gray)})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(ROOT, "data", "coco256", "meta.csv"), index=False)
    print(len(df), "ảnh; ảnh gốc xám:", int(df.gray_origin.sum()))


if __name__ == "__main__":
    main()

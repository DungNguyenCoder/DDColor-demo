"""
Kiểm thử chức năng chương trình demo:
  F01–F05: gọi trực tiếp các hàm với ảnh đặc biệt (lớn, nhỏ, kích thước lẻ, PNG có alpha, JPG, ảnh 1 kênh).
  F06–F10: chạy giao diện Streamlit ở chế độ không cửa sổ (streamlit.testing.AppTest).
Xuất results/functional_tests.csv
"""
import os, sys, time, traceback
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
FUNC = os.path.join(ROOT, "data", "testcases", "func")
rows = []


def rec(i, desc, ok, note):
    rows.append({"ID": i, "Nội dung": desc, "Kết quả": "Đạt" if ok else "Không đạt", "Ghi chú": note}); print(i, ok, note, flush=True)


def main():
    sys.path.insert(0, os.path.join(ROOT, "src", "app"))
    from src.models.ddcolor_runner import DDColorRunner
    dd = DDColorRunner("large", input_size=512)
    import importlib.util
    # dùng hàm đọc ảnh của giao diện (xử lý alpha, ảnh lớn)
    spec_read = lambda p: np.asarray(Image.open(p).convert("RGB"))

    def app_read(path):
        im = Image.open(path)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA"); bg = Image.new("RGBA", im.size, (255, 255, 255, 255)); im = Image.alpha_composite(bg, im)
        im = im.convert("RGB")
        if max(im.size) > 2048:
            s = 2048 / max(im.size); im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        return np.asarray(im)

    for i, (f, desc) in enumerate([("large_4000x3000.png", "Ảnh lớn 4000 × 3000"), ("tiny_32x32.png", "Ảnh rất nhỏ 32 × 32"),
                                   ("odd_401x299.png", "Ảnh kích thước lẻ 401 × 299"), ("rgba_logo.png", "PNG có kênh alpha"),
                                   ("photo.jpg", "Ảnh JPG"), ("single_channel.png", "Ảnh 1 kênh (xám)")], 1):
        try:
            x = app_read(os.path.join(FUNC, f))
            t0 = time.perf_counter(); out, ms = dd.colorize(x); total = (time.perf_counter() - t0) * 1000
            rec(f"F0{i}", desc, out.shape == x.shape and out.dtype == np.uint8,
                f"đầu vào {x.shape[1]}×{x.shape[0]}, đầu ra cùng kích thước; mạng {ms:.0f} ms, cả pipeline {total:.0f} ms")
        except Exception as e:
            rec(f"F0{i}", desc, False, f"{type(e).__name__}: {e}")
    try:
        from streamlit.testing.v1 import AppTest
        app = os.path.join(ROOT, "src", "app", "app.py")
        at = AppTest.from_file(app, default_timeout=600).run()
        labels = [m.label for m in at.metric]
        rec("F07", "Ảnh màu có ảnh chuẩn: hiển thị kết quả và độ đo", not at.exception and "PSNR (dB)" in labels and "Độ sặc sỡ CF" in labels,
            f"các chỉ số hiển thị: {', '.join(labels)}")
        rec("F08", "Tải ảnh kết quả", not at.exception and len(at.download_button) >= 1, "có nút tải ảnh PNG")
        at.sidebar.radio[0].set_value("Ảnh xám thật").run()
        rec("F09", "Ảnh xám thật, không có ảnh chuẩn", not at.exception and any("Không có ảnh chuẩn" in c.value for c in at.caption), "hiện thông báo thay cho PSNR")
        at.sidebar.radio[0].set_value("Tải ảnh lên").run()
        rec("F10", "Chưa chọn ảnh", not at.exception and any("Chọn hoặc tải" in i.value for i in at.info), "hiện hướng dẫn, chương trình không dừng")
        at2 = AppTest.from_file(app, default_timeout=600).run()
        at2.selectbox(key="method").set_value("Welsh 2002 (ảnh mẫu)").run()
        rec("F11", "Chọn phương pháp Welsh và ảnh mẫu", not at2.exception and any("Ảnh mẫu" in s.label for s in at2.selectbox), "hiện ô chọn ảnh mẫu, chạy không lỗi")
    except Exception as e:
        traceback.print_exc(); rec("F07", "Giao diện", False, f"{type(e).__name__}: {e}")
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "functional_tests.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

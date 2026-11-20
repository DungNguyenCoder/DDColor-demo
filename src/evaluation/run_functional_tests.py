"""
Kiểm thử chức năng chương trình demo:
  F01–F06, F15–F16: đọc ảnh đặc biệt bằng chính hàm đọc ảnh của giao diện (src/utils/image_io.py) rồi tô màu bằng DDColor-L.
  F07–F14: chạy giao diện Streamlit ở chế độ không cửa sổ (streamlit.testing.AppTest).
Xuất results/functional_tests.csv
"""
import os, sys, time, traceback
import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.utils.image_io import read_image  # noqa: E402

FUNC = os.path.join(ROOT, "data", "testcases", "func")
APP = os.path.join(ROOT, "src", "app", "app.py")
rows = []


def rec(i, desc, ok, note):
    rows.append({"ID": i, "Nội dung": desc, "Kết quả": "Đạt" if ok else "Không đạt", "Ghi chú": note}); print(i, ok, note, flush=True)


def image_cases(dd):
    big = os.path.join(FUNC, "large_4000x3000.png")
    if not os.path.exists(big):          # ảnh ~20 MB không đưa lên git; tạo lại giống prepare_testcases.py
        import skimage.data as sd
        Image.fromarray(sd.coffee()).resize((4000, 3000), Image.BICUBIC).save(big)
    cases = [("F01", "large_4000x3000.png", "Ảnh lớn 4000 × 3000"), ("F02", "tiny_32x32.png", "Ảnh rất nhỏ 32 × 32"),
             ("F03", "odd_401x299.png", "Ảnh kích thước lẻ 401 × 299"), ("F04", "rgba_logo.png", "PNG có kênh alpha"),
             ("F05", "photo.jpg", "Ảnh JPG"), ("F06", "single_channel.png", "Ảnh 1 kênh (xám)")]
    for i, f, desc in cases:
        try:
            x, _ = read_image(os.path.join(FUNC, f))
            t0 = time.perf_counter(); out, ms = dd.colorize(x); total = (time.perf_counter() - t0) * 1000
            rec(i, desc, out.shape == x.shape and out.dtype == np.uint8,
                f"đầu vào {x.shape[1]}×{x.shape[0]}, đầu ra cùng kích thước; mạng {ms:.0f} ms, cả pipeline {total:.0f} ms")
        except Exception as e:
            rec(i, desc, False, f"{type(e).__name__}: {e}")
    # F15: ảnh điện thoại lưu nằm ngang 600×400 nhưng EXIF yêu cầu xoay 90° -> phải đọc thành ảnh đứng 400×600
    try:
        x, _ = read_image(os.path.join(FUNC, "exif_rotated.jpg"))
        out, _ = dd.colorize(x)
        rec("F15", "Ảnh điện thoại có thẻ xoay EXIF", x.shape[:2] == (600, 400) and out.shape == x.shape,
            f"đọc thành ảnh đứng {x.shape[1]}×{x.shape[0]} đúng chiều hiển thị, tô màu được")
    except Exception as e:
        rec("F15", "Ảnh điện thoại có thẻ xoay EXIF", False, f"{type(e).__name__}: {e}")
    # F16: ảnh xám 16 bit phải giữ đúng độ sáng (so với bản 8 bit gốc)
    try:
        x, _ = read_image(os.path.join(FUNC, "gray16.png"))
        ref = np.asarray(Image.open(os.path.join(FUNC, "single_channel.png")).convert("RGB"))
        diff = int(np.abs(x.astype(int) - ref.astype(int)).max())
        out, _ = dd.colorize(x)
        rec("F16", "Ảnh xám 16 bit", diff <= 1 and out.shape == x.shape, f"độ sáng lệch tối đa {diff} mức so với bản 8 bit, tô màu được")
    except Exception as e:
        rec("F16", "Ảnh xám 16 bit", False, f"{type(e).__name__}: {e}")


def ui_cases():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(APP, default_timeout=900).run()
    labels = [m.label for m in at.metric]
    rec("F07", "Ảnh màu có ảnh chuẩn: hiển thị kết quả và độ đo", not at.exception and "PSNR (dB)" in labels and "Độ sặc sỡ CF" in labels,
        f"các chỉ số hiển thị: {', '.join(labels)}")
    rec("F08", "Tải ảnh kết quả", not at.exception and len(at.download_button) >= 1, "có nút tải ảnh PNG")
    at.sidebar.radio[0].set_value("Ảnh xám thật").run()
    rec("F09", "Ảnh xám thật, không có ảnh chuẩn", not at.exception and any("Không có ảnh chuẩn" in c.value for c in at.caption), "hiện thông báo thay cho PSNR")
    at.sidebar.radio[0].set_value("Tải ảnh lên").run()
    rec("F10", "Chưa chọn ảnh", not at.exception and any("Chọn hoặc tải" in i.value for i in at.info), "hiện hướng dẫn, chương trình không dừng")

    at = AppTest.from_file(APP, default_timeout=900).run()
    at.selectbox(key="method").set_value("Welsh 2002 (ảnh mẫu)").run()
    rec("F11", "Chọn phương pháp Welsh và ảnh mẫu", not at.exception and any("Ảnh mẫu" in s.label for s in at.selectbox), "hiện ô chọn ảnh mẫu, chạy không lỗi")

    # F12: lần lượt chọn mọi phương pháp ở tab Tô màu
    at = AppTest.from_file(APP, default_timeout=900).run()
    methods = list(at.selectbox(key="method").options)
    bad = []
    for m in methods:
        at.selectbox(key="method").set_value(m).run()
        if at.exception:
            bad.append(m)
    rec("F12", "Chạy lần lượt mọi phương pháp", not bad and len(methods) == 7, f"{len(methods)} phương pháp chạy không lỗi" if not bad else "lỗi: " + ", ".join(bad))

    # F13: tab So sánh, bấm nút chạy tất cả
    btn = [b for b in at.button if b.label == "Chạy tất cả phương pháp"][0]
    btn.click().run()
    n = len(at.dataframe[0].value) if len(at.dataframe) else 0
    rec("F13", "So sánh tất cả phương pháp", not at.exception and n == 7, f"bảng so sánh có {n} dòng (thời gian, CF, PSNR)")

    # F14: đổi kích thước đầu vào của mạng
    bad = []
    for s in [256, 384, 512]:
        at.sidebar.select_slider[0].set_value(s).run()
        if at.exception:
            bad.append(s)
    rec("F14", "Đổi kích thước đầu vào 256/384/512", not bad, "chạy không lỗi ở cả ba kích thước" if not bad else f"lỗi ở {bad}")


def main():
    from src.models.ddcolor_runner import DDColorRunner
    image_cases(DDColorRunner("large", input_size=512))
    try:
        ui_cases()
    except Exception as e:
        traceback.print_exc(); rec("F07", "Giao diện", False, f"{type(e).__name__}: {e}")
    df = pd.DataFrame(rows)
    df = df.assign(_k=df.ID.str[1:].astype(int)).sort_values("_k").drop(columns="_k")
    df.to_csv(os.path.join(ROOT, "results", "functional_tests.csv"), index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()

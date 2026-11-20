"""
Demo tô màu ảnh xám – DDColor (ICCV 2023).
Chạy:  python -m streamlit run src/app/app.py
"""
import io, os, sys, time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.models.lab import rgb_to_lab, lab_to_rgb
from src.models import welsh
from src.utils.metrics import psnr, colorfulness
from src.utils.image_io import read_image as _read, MAX_SIDE

st.set_page_config(page_title="DDColor – Tô màu ảnh xám", layout="wide")
TC = os.path.join(ROOT, "data", "testcases")


@st.cache_resource
def get_ddcolor(name):
    """Mỗi bản DDColor chỉ nạp một lần; kích thước đầu vào truyền khi tô màu nên đổi kích thước không nạp thêm mô hình."""
    from src.models.ddcolor_runner import DDColorRunner
    return DDColorRunner(name)


@st.cache_resource
def get_cic():
    from src.models.cic import CICRunner
    return CICRunner()


def to_u8(x):
    return np.clip(np.rint(x * 255), 0, 255).astype(np.uint8)


def gray_of(rgb):
    lab = rgb_to_lab(rgb / 255.0); lab[..., 1:] = 0
    return to_u8(lab_to_rgb(lab))


def read_image(f):
    try:
        a, (w, h) = _read(f)
    except Exception as e:                                         # tệp hỏng hoặc không phải ảnh
        st.error(f"Không đọc được ảnh: {e}")
        return None
    if max(w, h) > MAX_SIDE:
        st.info(f"Ảnh lớn {w} × {h} được thu nhỏ về {a.shape[1]} × {a.shape[0]} để xử lý.")
    return a


def png_bytes(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, format="PNG"); return b.getvalue()


def run(method, rgb, size, ref=None):
    t0 = time.perf_counter()
    if method.startswith("DDColor"):
        name = {"DDColor-L (paper)": "large", "DDColor-L (modelscope)": "modelscope", "DDColor-T (tiny)": "tiny"}[method]
        out, ms = get_ddcolor(name).colorize(rgb, input_size=size)
        return out, ms
    lab = rgb_to_lab(rgb / 255.0)
    if method == "CIC (Zhang 2016)":
        lab[..., 1:] = get_cic().colorize_L(lab[..., 0]); out = to_u8(lab_to_rgb(lab))
    elif method == "Welsh 2002 (ảnh mẫu)":
        out = to_u8(welsh.colorize(lab[..., 0], (ref if ref is not None else rgb) / 255.0))
    elif method == "Giả màu (Chương 3)":
        from src.models.pseudocolor import sine_transform
        out = sine_transform(np.clip(np.rint(lab[..., 0] / 100 * 255), 0, 255).astype(np.uint8))
    else:
        lab[..., 1:] = 0; out = to_u8(lab_to_rgb(lab))
    return out, (time.perf_counter() - t0) * 1000


@st.cache_data(max_entries=64, show_spinner=False)
def run_cached(method, rgb, size, ref=None):
    """Giữ kết quả theo (phương pháp, ảnh, kích thước, ảnh mẫu): mỗi lần bấm nút Streamlit chạy lại cả trang,
    nên không có bộ nhớ đệm thì mọi tab đều tô màu lại từ đầu."""
    return run(method, rgb, size, ref)


@st.cache_data(max_entries=16, show_spinner=False)
def query_maps(rgb, size):
    out, _, maps = get_ddcolor("large").colorize(rgb, return_maps=True, input_size=size)
    return out, maps


METHODS = ["DDColor-L (paper)", "DDColor-L (modelscope)", "DDColor-T (tiny)", "CIC (Zhang 2016)", "Welsh 2002 (ảnh mẫu)",
           "Giả màu (Chương 3)", "Ảnh xám (không tô)"]

st.title("Tô màu ảnh xám – DDColor (ICCV 2023)")
st.caption("Bài tập lớn Xử lý ảnh – Nhóm 15. Ảnh màu được chuyển về xám (giữ kênh L của Lab) trước khi tô; ảnh gốc dùng làm ảnh chuẩn.")

WEIGHTS = [os.path.join("weights", d, f) for d in ("ddcolor_paper", "ddcolor_paper_tiny", "ddcolor_modelscope")
           for f in ("config.json", "pytorch_model.bin")] + [os.path.join("weights", "cic", "colorization_release_v2-9b330a0b.pth")]
missing = [w for w in WEIGHTS if not os.path.exists(os.path.join(ROOT, w))]
if missing:                       # máy mới clone về: trọng số (~2 GB) không nằm trong git
    st.error("Chưa có trọng số mô hình: " + ", ".join(missing) + ". Chạy `python scripts/download_weights.py` "
             "(hoặc nháy đúp setup.bat) rồi tải lại trang.")
    st.stop()

with st.sidebar:
    src = st.radio("Nguồn ảnh", ["Ảnh màu có ảnh chuẩn", "Ảnh xám thật", "Tải ảnh lên"])
    rgb, gt, name = None, None, "anh.png"
    if src == "Ảnh màu có ảnh chuẩn":
        name = st.selectbox("Chọn ảnh", sorted(os.listdir(os.path.join(TC, "color"))))
        gt = read_image(os.path.join(TC, "color", name)); rgb = gt
    elif src == "Ảnh xám thật":
        name = st.selectbox("Chọn ảnh", sorted(os.listdir(os.path.join(TC, "gray"))))
        rgb = read_image(os.path.join(TC, "gray", name))
    else:
        up = st.file_uploader("Ảnh JPG/PNG/BMP", type=["jpg", "jpeg", "png", "bmp"])
        if up is not None:
            rgb = read_image(up); name = up.name
            if st.checkbox("Ảnh tải lên là ảnh màu (dùng làm ảnh chuẩn)", value=False):
                gt = rgb
    size = st.select_slider("Kích thước đầu vào của mạng", options=[256, 384, 512], value=512)

tabs = st.tabs(["Tô màu", "So sánh phương pháp", "Kênh Lab", "Color query", "Kết quả đánh giá"])

with tabs[0]:
    if rgb is None:
        st.info("Chọn hoặc tải một ảnh ở thanh bên trái.")
    else:
        method = st.selectbox("Phương pháp", METHODS, key="method")
        ref = None
        if method.startswith("Welsh"):
            refname = st.selectbox("Ảnh mẫu màu", sorted(os.listdir(os.path.join(TC, "color"))))
            ref = read_image(os.path.join(TC, "color", refname))
        with st.spinner("Đang tô màu..."):
            out, ms = run_cached(method, rgb, size, ref)
        cols = st.columns(3 if gt is not None else 2)
        cols[0].image(gray_of(rgb), caption="Ảnh xám đầu vào", width="stretch")
        cols[1].image(out, caption=f"{method} – {ms:.0f} ms", width="stretch")
        if gt is not None:
            cols[2].image(gt, caption="Ảnh chuẩn", width="stretch")
            c1, c2, c3 = st.columns(3)
            c1.metric("PSNR (dB)", f"{psnr(out, gt):.2f}")
            c2.metric("Độ sặc sỡ CF", f"{colorfulness(out):.1f}")
            c3.metric("CF ảnh chuẩn", f"{colorfulness(gt):.1f}")
        else:
            st.metric("Độ sặc sỡ CF", f"{colorfulness(out):.1f}")
            st.caption("Không có ảnh chuẩn nên không tính PSNR.")
        st.download_button("Tải ảnh kết quả (PNG)", png_bytes(out), file_name="tomau_" + os.path.splitext(name)[0] + ".png", mime="image/png")

with tabs[1]:
    if rgb is None:
        st.info("Chọn ảnh trước.")
    elif st.button("Chạy tất cả phương pháp", type="primary"):
        ref = read_image(os.path.join(TC, "color", "coffee.png" if name != "coffee.png" else "chelsea.png"))
        cells = st.columns(3); i = 0
        rows = []
        for m in METHODS:
            out, ms = run_cached(m, rgb, size, ref)
            cap = f"{m} – {ms:.0f} ms"
            r = {"Phương pháp": m, "Thời gian (ms)": round(ms, 1), "CF": round(colorfulness(out), 1)}
            if gt is not None:
                r["PSNR (dB)"] = round(psnr(out, gt), 2)
            rows.append(r)
            cells[i % 3].image(out, caption=cap, width="stretch"); i += 1
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with tabs[2]:
    if rgb is not None:
        method = st.selectbox("Phương pháp để xem kênh", METHODS[:4], key="labm")
        out, _ = run_cached(method, rgb, size)
        lab = rgb_to_lab(out / 255.0)
        c = st.columns(4)
        c[0].image(out, caption="Kết quả", width="stretch")
        c[1].image(to_u8(lab[..., 0] / 100), caption="L* (độ sáng, giữ từ ảnh xám)", width="stretch")
        c[2].image(to_u8((lab[..., 1] + 100) / 200), caption="a* (lục ← → đỏ)", width="stretch")
        c[3].image(to_u8((lab[..., 2] + 100) / 200), caption="b* (lam ← → vàng)", width="stretch")
        st.caption("a*, b* hiển thị với 128 là trung tính: sáng hơn nghĩa là đỏ hơn (a*) hoặc vàng hơn (b*).")

with tabs[3]:
    if rgb is not None:
        out, maps = query_maps(rgb, size)
        k = st.slider("Số query hiển thị", 3, 12, 6)
        from src.models.ddcolor_runner import diverse_queries
        order, act = diverse_queries(maps, k)
        st.caption("Mỗi bản đồ là sigmoid(tích vô hướng giữa một color query và đặc trưng ảnh) – vùng đỏ là nơi query đó hoạt động mạnh (tái hiện Hình 7 của bài báo).")
        import matplotlib.cm as cm
        cols = st.columns(4)
        cols[0].image(out, caption="Kết quả DDColor-L", width="stretch")
        for j, q in enumerate(order[:k], 1):
            cols[j % 4].image(to_u8(cm.jet(act[q])[..., :3]), caption=f"query {q}", width="stretch")

with tabs[4]:
    p = os.path.join(ROOT, "results", "coco_final.csv")
    if os.path.exists(p):
        st.subheader("COCO val2017, 5.000 ảnh 256 × 256")
        names = {"gray": "Ảnh xám", "pseudocolor": "Giả màu", "welsh": "Welsh 2002", "cic": "CIC (Zhang 2016)", "ddcolor_tiny_256": "DDColor-T",
                 "ddcolor_large_256": "DDColor-L", "ddcolor_modelscope_256": "DDColor-L (modelscope)", "ddcolor_large_512": "DDColor-L, đầu vào 512",
                 "ground_truth": "Ảnh gốc"}
        df = pd.read_csv(p)[["method", "FID", "CF", "dCF", "PSNR", "ms"]]
        df["method"] = df.method.map(lambda m: names.get(m, m))
        gt_row = df.method == "Ảnh gốc"
        for c in ["FID", "CF", "dCF", "PSNR"]:                  # ô không áp dụng (ảnh gốc, PSNR trống) hiện "–"
            df[c] = [("–" if pd.isna(v) or (g and c != "CF") else f"{v:.2f}") for v, g in zip(df[c], gt_row)]
        df["ms"] = ["–" if pd.isna(v) or v == 0 else f"{v:.1f}" for v in df["ms"]]
        st.dataframe(df.rename(columns={"method": "Phương pháp", "dCF": "ΔCF", "ms": "ms/ảnh"}), hide_index=True, width="stretch")
        st.caption("FID, ΔCF càng thấp càng tốt. PSNR bỏ qua giá trị vô cực của ảnh vốn xám.")
        for f, c in [("compare_coco.png", "So sánh trên 5 ảnh COCO chọn ngẫu nhiên"), ("failures.png", "Các loại ảnh DDColor-L làm chưa tốt")]:
            fp = os.path.join(ROOT, "figures", f)
            if os.path.exists(fp):
                st.image(fp, caption=c, width="stretch")
    else:
        st.info("Chưa có results/coco_final.csv – chạy src/evaluation/evaluate_coco.py rồi summarize.py.")

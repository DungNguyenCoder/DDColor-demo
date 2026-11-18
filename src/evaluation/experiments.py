"""
Các thí nghiệm phụ (chạy sau evaluate_coco.py):
  queries     : trực quan hóa color query (tái hiện Hình 7 của bài báo)
  robustness  : ảnh xám bị nhiễu / nén JPG / mờ / thu nhỏ -> màu thay đổi bao nhiêu
  timing      : thời gian suy luận GPU/CPU theo kích thước đầu vào
  welsh_ref   : ảnh hưởng của ảnh mẫu tới phương pháp Welsh
  python src/evaluation/experiments.py queries robustness timing welsh_ref
"""
import os, sys, io, time, json
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.models.lab import rgb_to_lab, lab_to_rgb
from src.utils.metrics import psnr, colorfulness

GT = os.path.join(ROOT, "data", "coco256", "gt")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
RES = os.path.join(ROOT, "results")
load = lambda p: np.asarray(Image.open(p).convert("RGB"))
plt.rcParams["font.family"] = "DejaVu Sans"


def gray_of(rgb):
    lab = rgb_to_lab(rgb / 255.0); lab[..., 1:] = 0
    return np.clip(np.rint(lab_to_rgb(lab) * 255), 0, 255).astype(np.uint8)


def queries(names=("000000000139.png", "000000001000.png")):
    from src.models.ddcolor_runner import DDColorRunner
    r = DDColorRunner("large", input_size=256)
    fig, axes = plt.subplots(len(names), 6, figsize=(15, 2.8 * len(names)))
    for row, n in enumerate(names):
        x = load(os.path.join(GT, n))
        out, _, maps = r.colorize(x, return_maps=True)
        from src.models.ddcolor_runner import diverse_queries
        idx, act = diverse_queries(maps, 5)                  # sigmoid(tích vô hướng query · đặc trưng), chọn 5 query khác nhau nhất
        axes[row, 0].imshow(out); axes[row, 0].set_title("DDColor", fontsize=9)
        for j, q in enumerate(idx, 1):
            axes[row, j].imshow(act[q], cmap="jet", vmin=0, vmax=1); axes[row, j].set_title(f"query {q}", fontsize=9)
        for a in axes[row]: a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "color_queries.png"), dpi=120); plt.close(fig)
    print("queries ok")


def degrade(gray_u8, kind):
    g = Image.fromarray(gray_u8)
    if kind == "noise":
        a = gray_u8.astype(float) + np.random.default_rng(0).normal(0, 12, gray_u8.shape[:2])[..., None]
        return np.clip(np.rint(a), 0, 255).astype(np.uint8)
    if kind == "jpeg":
        b = io.BytesIO(); g.save(b, format="JPEG", quality=20); return np.asarray(Image.open(b).convert("RGB"))
    if kind == "blur":
        return np.asarray(g.filter(ImageFilter.GaussianBlur(2)))
    if kind == "lowres":
        h, w = gray_u8.shape[:2]
        return np.asarray(g.resize((w // 4, h // 4), Image.BICUBIC).resize((w, h), Image.BICUBIC))
    return gray_u8


def robustness(n=300):
    from src.models.ddcolor_runner import DDColorRunner
    from src.models.cic import CICRunner
    dd = DDColorRunner("large", input_size=256); cic = CICRunner()
    files = sorted(os.listdir(GT))[:n]
    rows = []
    for f in files:
        x = load(os.path.join(GT, f)); g = gray_of(x)
        base_dd = dd.colorize(g)[0]
        lab = rgb_to_lab(g / 255.0); lab[..., 1:] = cic.colorize_L(lab[..., 0]); base_cic = np.clip(np.rint(lab_to_rgb(lab) * 255), 0, 255).astype(np.uint8)
        for kind in ["clean", "noise", "jpeg", "blur", "lowres"]:
            d = degrade(g, kind)
            o = dd.colorize(d)[0]
            lab = rgb_to_lab(d / 255.0); lab[..., 1:] = cic.colorize_L(lab[..., 0]); oc = np.clip(np.rint(lab_to_rgb(lab) * 255), 0, 255).astype(np.uint8)
            for m, out, base in [("DDColor-L", o, base_dd), ("CIC", oc, base_cic)]:
                ab_o, ab_b = rgb_to_lab(out / 255.0)[..., 1:], rgb_to_lab(base / 255.0)[..., 1:]
                rows.append({"method": m, "degradation": kind, "image": f, "cf": colorfulness(out),
                             "dab_vs_clean": float(np.linalg.norm(ab_o - ab_b, axis=-1).mean()), "psnr_vs_gt": psnr(out, x)})
    df = pd.DataFrame(rows)
    s = df.groupby(["method", "degradation"], sort=False).mean(numeric_only=True).round(2)
    s.to_csv(os.path.join(RES, "robustness.csv")); print(s)
    robustness_figure(dd)


def robustness_figure(dd=None):
    """Hình minh họa: ảnh thứ 4 của tập (biển báo STOP), hàng trên ảnh xám bị suy giảm, hàng dưới kết quả DDColor-L."""
    if dd is None:
        from src.models.ddcolor_runner import DDColorRunner
        dd = DDColorRunner("large", input_size=256)
    files = sorted(os.listdir(GT))
    x = load(os.path.join(GT, files[3])); g = gray_of(x)
    kinds = [("clean", "Ảnh sạch"), ("noise", "Nhiễu Gauss σ = 12"), ("jpeg", "JPG chất lượng 20"), ("blur", "Làm mờ σ = 2"), ("lowres", "Thu nhỏ 4 lần")]
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for j, (k, t) in enumerate(kinds):
        d = degrade(g, k); axes[0, j].imshow(d); axes[0, j].set_title(t, fontsize=10)
        axes[1, j].imshow(dd.colorize(d)[0])
        for a in axes[:, j]: a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "robustness.png"), dpi=110); plt.close(fig)


def timing():
    import torch
    from src.models.ddcolor_runner import DDColorRunner
    from src.models.cic import CICRunner
    x = load(os.path.join(ROOT, "data", "testcases", "color", "coffee.png"))
    rows = []
    for dev in ["cuda", "cpu"]:
        device = torch.device(dev)
        for name in ["tiny", "large"]:
            for size in [256, 512]:
                if dev == "cpu" and size == 512 and name == "large":
                    continue
                r = DDColorRunner(name, input_size=size, device=device)
                for _ in range(2): r.colorize(x)
                ts = sorted(r.colorize(x)[1] for _ in range(5 if dev == "cuda" else 3))
                rows.append({"model": f"DDColor-{name[0].upper()}", "device": dev.upper(), "input": size, "ms_median": round(ts[len(ts) // 2], 1)})
                print(rows[-1], flush=True)
                del r; torch.cuda.empty_cache()
        c = CICRunner(device=device); L = rgb_to_lab(x / 255.0)[..., 0]
        for _ in range(2): c.colorize_L(L)
        ts = []
        for _ in range(5):
            if dev == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter(); c.colorize_L(L)
            if dev == "cuda": torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        rows.append({"model": "CIC", "device": dev.upper(), "input": 256, "ms_median": round(sorted(ts)[2], 1)})
    from src.models import welsh
    L = rgb_to_lab(x / 255.0)[..., 0]; ref = load(os.path.join(ROOT, "data", "testcases", "color", "chelsea.png")) / 255.0
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); welsh.colorize(L, ref); ts.append((time.perf_counter() - t0) * 1000)
    rows.append({"model": "Welsh (NumPy)", "device": "CPU", "input": "600x400", "ms_median": round(sorted(ts)[2], 1)})
    pd.DataFrame(rows).to_csv(os.path.join(RES, "timing.csv"), index=False)
    print(pd.DataFrame(rows))


def welsh_ref():
    from src.models import welsh
    tgt = load(os.path.join(ROOT, "data", "testcases", "color", "chelsea.png"))
    L = rgb_to_lab(tgt / 255.0)[..., 0]
    refs = ["coffee", "rocket", "astronaut"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.5))
    axes[0, 0].imshow(gray_of(tgt)); axes[0, 0].set_title("Ảnh xám đầu vào", fontsize=10)
    axes[1, 0].imshow(tgt); axes[1, 0].set_title("Ảnh gốc", fontsize=10)
    rows = []
    for j, rname in enumerate(refs, 1):
        ref = load(os.path.join(ROOT, "data", "testcases", "color", rname + ".png"))
        out = np.clip(np.rint(welsh.colorize(L, ref / 255.0) * 255), 0, 255).astype(np.uint8)
        rows.append({"ref": rname, "psnr": round(psnr(out, tgt), 2), "cf": round(colorfulness(out), 1)})
        axes[0, j].imshow(ref); axes[0, j].set_title(f"Ảnh mẫu: {rname}", fontsize=10)
        axes[1, j].imshow(out); axes[1, j].set_title(f"PSNR {rows[-1]['psnr']} dB", fontsize=10)
    for a in axes.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "welsh_reference.png"), dpi=110); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "welsh_reference.csv"), index=False); print(rows)


if __name__ == "__main__":
    for task in sys.argv[1:]:
        globals()[task]()

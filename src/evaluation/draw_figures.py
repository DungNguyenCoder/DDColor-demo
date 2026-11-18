"""Vẽ các hình minh họa cho báo cáo: luồng xử lý, ví dụ Lab, đa nghiệm của bài toán, ảnh so sánh."""
import os, sys
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.models.lab import rgb_to_lab, lab_to_rgb
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams["font.family"] = "DejaVu Sans"


def pipeline():
    fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")

    def box(x, y, w, h, t, c):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc="white", ec="#333", lw=1))
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))

    box(0.1, 2.9, 2.2, 1.4, "Ảnh màu COCO val2017\n5.000 ảnh, 256 × 256\n+ ảnh test case", "#E8F0FE")
    box(0.1, 0.6, 2.2, 1.4, "Ảnh xám thật\n(không có ảnh chuẩn)", "#E8F0FE")
    box(2.9, 2.9, 2.0, 1.4, "RGB → Lab\ngiữ L*, bỏ a*, b*\n(ảnh xám đầu vào)", "#FFF4E5")
    box(5.5, 3.55, 2.6, 0.8, "Giả màu (Mục 1.2)", "#FFF4E5")
    box(5.5, 2.6, 2.6, 0.8, "Welsh 2002 (ảnh mẫu, NumPy)", "#FFF4E5")
    box(5.5, 1.65, 2.6, 0.8, "CIC – Zhang 2016 (CNN)", "#FFF4E5")
    box(5.5, 0.4, 2.6, 1.1, "DDColor-T / DDColor-L\n(ICCV 2023, trọng số tác giả)", "#FFE8D6")
    box(8.7, 1.6, 1.6, 1.6, "Ghép L* gốc\n+ a*, b* dự đoán\n→ RGB", "#E9F7EF")
    box(10.7, 3.1, 1.2, 1.2, "FID, CF,\nΔCF, PSNR", "#FDECEA")
    box(10.7, 1.6, 1.2, 1.2, "So sánh\nbằng mắt", "#FDECEA")
    box(10.7, 0.1, 1.2, 1.2, "Test case,\nthất bại", "#FDECEA")
    arr(2.3, 3.6, 2.9, 3.6); arr(2.3, 1.3, 3.9, 2.85)
    for y in [3.95, 3.0, 2.05, 0.95]:
        arr(4.9, 3.6, 5.5, y)
    for y in [3.0, 2.05, 0.95]:
        arr(8.1, y, 8.7, 2.4)
    ax.annotate("", xy=(10.7, 3.95), xytext=(8.1, 3.95), arrowprops=dict(arrowstyle="->", lw=1.2, ls="--"))
    ax.text(9.4, 4.05, "RGB trực tiếp", ha="center", fontsize=8)
    for y in [3.7, 2.2, 0.7]:
        arr(10.3, 2.4, 10.7, y)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "pipeline.png"), dpi=150); plt.close(fig)


def ambiguity():
    """Hai màu khác hẳn nhau nhưng cùng L* -> cùng một mức xám."""
    red = np.array([[[1.0, 0.0, 0.0]]]); Lr = rgb_to_lab(red)[0, 0, 0]
    # tìm màu lục và lam có cùng L*
    def match(axis):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2; c = np.zeros((1, 1, 3)); c[0, 0, axis] = mid
            if axis == 2:
                c[0, 0, 1] = mid * 0.55
            if rgb_to_lab(c)[0, 0, 0] < Lr: lo = mid
            else: hi = mid
        c = np.zeros((1, 1, 3)); c[0, 0, axis] = mid
        if axis == 2: c[0, 0, 1] = mid * 0.55
        return c
    cols = [red, match(1), match(2)]
    fig, axes = plt.subplots(2, 3, figsize=(8, 4.2))
    rows = []
    for j, c in enumerate(cols):
        lab = rgb_to_lab(c)[0, 0]; rgb8 = np.rint(c[0, 0] * 255).astype(int)
        g = lab.copy(); g[1:] = 0; gray = lab_to_rgb(g[None, None])[0, 0]
        axes[0, j].imshow(np.ones((20, 20, 3)) * c[0, 0]); axes[0, j].set_title(f"RGB {tuple(rgb8)}\nL*={lab[0]:.1f} a*={lab[1]:.0f} b*={lab[2]:.0f}", fontsize=8)
        axes[1, j].imshow(np.ones((20, 20, 3)) * gray); axes[1, j].set_title("ảnh xám (giữ L*)", fontsize=8)
        rows.append({"rgb": tuple(rgb8), "L": round(lab[0], 2), "a": round(lab[1], 2), "b": round(lab[2], 2)})
    for a in axes.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "ambiguity.png"), dpi=130); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "ambiguity.csv"), index=False); print(rows)


def compare(names, out_name, methods=("gray", "pseudocolor", "welsh", "cic", "ddcolor_tiny_256", "ddcolor_large_256"), labels=None):
    labels = labels or ["Ảnh xám", "Giả màu", "Welsh", "CIC", "DDColor-T", "DDColor-L", "Ảnh gốc"]
    res = os.path.join(ROOT, "results", "coco256")
    fig, axes = plt.subplots(len(names), len(methods) + 1, figsize=(2.1 * (len(methods) + 1), 2.15 * len(names)))
    axes = np.atleast_2d(axes)
    for i, n in enumerate(names):
        for j, m in enumerate(list(methods) + ["gt"]):
            p = os.path.join(ROOT, "data", "coco256", "gt", n) if m == "gt" else os.path.join(res, m, n)
            axes[i, j].imshow(Image.open(p)); axes[i, j].axis("off")
            if i == 0: axes[i, j].set_title(labels[j], fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, out_name), dpi=120); plt.close(fig)


FAILURES = [("000000009448.png", "(a)"), ("000000419882.png", "(b)"), ("000000340930.png", "(c)"), ("000000085089.png", "(d)"), ("000000238866.png", "(e)")]


def compare_coco():
    """Hình so sánh trên 5 ảnh COCO chọn ngẫu nhiên (random_state = 3) trong các ảnh có CF gốc từ 30 đến 70."""
    d = pd.read_csv(os.path.join(ROOT, "results", "coco_details.csv")); d["image"] = d.image.astype(str)
    a = d[d.method == "ddcolor_large_256"]
    names = list(a[(a.cf_gt > 30) & (a.cf_gt < 70)].sample(16, random_state=3).image[:5])
    compare(names, "compare_coco.png")


def failures(rows=FAILURES, out_name="failures.png", methods=("cic", "ddcolor_large_256")):
    """rows: list (tên ảnh COCO, nhãn hàng), chọn từ results/failure_candidates.csv. Cột: ảnh xám, các phương pháp, ảnh gốc."""
    res = os.path.join(ROOT, "results", "coco256")
    cols = ["gray"] + list(methods) + ["gt"]
    titles = {"gray": "Ảnh xám", "cic": "CIC", "ddcolor_large_256": "DDColor-L", "ddcolor_tiny_256": "DDColor-T", "gt": "Ảnh gốc"}
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.3 * len(cols) + 0.6, 2.3 * len(rows)))
    for i, (n, lab) in enumerate(rows):
        for j, m in enumerate(cols):
            p = os.path.join(ROOT, "data", "coco256", "gt", n) if m == "gt" else os.path.join(res, m, n)
            ax = axes[i, j]; ax.imshow(Image.open(p)); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)
            if i == 0: ax.set_title(titles[m], fontsize=10)
            if j == 0: ax.set_ylabel(lab, fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, out_name), dpi=120); plt.close(fig)


def testcases():
    """Chạy mọi phương pháp trên ảnh test case, lưu results/testcases/<phương pháp>/ và các lưới so sánh."""
    from src.models import welsh
    from src.models.pseudocolor import sine_transform
    from src.models.cic import CICRunner
    from src.models.ddcolor_runner import DDColorRunner
    from src.utils.metrics import psnr, colorfulness
    tc = os.path.join(ROOT, "data", "testcases")
    out_root = os.path.join(ROOT, "results", "testcases")
    cic = CICRunner(); ddt = DDColorRunner("tiny", 512); ddl = DDColorRunner("large", 512); ddm = DDColorRunner("modelscope", 512)
    ref = np.asarray(Image.open(os.path.join(tc, "color", "coffee.png")).convert("RGB")) / 255.0
    ref2 = np.asarray(Image.open(os.path.join(tc, "color", "chelsea.png")).convert("RGB")) / 255.0
    rows = []
    groups = {"color": sorted(os.listdir(os.path.join(tc, "color"))), "gray": sorted(os.listdir(os.path.join(tc, "gray")))}
    labels = ["Ảnh xám", "Giả màu", "Welsh", "CIC", "DDColor-T", "DDColor-L", "DDColor-L (modelscope)"]
    for grp, names in groups.items():
        fig, axes = plt.subplots(len(names), len(labels) + (1 if grp == "color" else 0), figsize=(2.0 * (len(labels) + 1), 2.0 * len(names)))
        for i, n in enumerate(names):
            x = np.asarray(Image.open(os.path.join(tc, grp, n)).convert("RGB"))
            lab = rgb_to_lab(x / 255.0); L = lab[..., 0]
            g = lab.copy(); g[..., 1:] = 0; gray = np.clip(np.rint(lab_to_rgb(g) * 255), 0, 255).astype(np.uint8)
            outs = [gray, sine_transform(np.clip(np.rint(L / 100 * 255), 0, 255).astype(np.uint8)),
                    np.clip(np.rint(welsh.colorize(L, ref2 if n == "coffee.png" else ref) * 255), 0, 255).astype(np.uint8)]
            c = lab.copy(); c[..., 1:] = cic.colorize_L(L); outs.append(np.clip(np.rint(lab_to_rgb(c) * 255), 0, 255).astype(np.uint8))
            outs += [ddt.colorize(x)[0], ddl.colorize(x)[0], ddm.colorize(x)[0]]
            for lab_name, o in zip(labels, outs):
                d = os.path.join(out_root, lab_name.replace(" ", "_").replace("(", "").replace(")", "")); os.makedirs(d, exist_ok=True)
                Image.fromarray(o).save(os.path.join(d, n))
                rows.append({"group": grp, "image": n, "method": lab_name, "cf": round(colorfulness(o), 2),
                             "psnr": round(psnr(o, x), 2) if grp == "color" else np.nan, "cf_gt": round(colorfulness(x), 2) if grp == "color" else np.nan})
            panels = outs + ([x] if grp == "color" else [])
            for j, o in enumerate(panels):
                axes[i, j].imshow(o); axes[i, j].axis("off")
                if i == 0: axes[i, j].set_title((labels + ["Ảnh gốc"])[j], fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(FIG, f"testcases_{grp}.png"), dpi=100); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "testcases.csv"), index=False)
    print("testcases ok")


if __name__ == "__main__":
    tasks = sys.argv[1:] or ["pipeline", "ambiguity"]
    for t in tasks:
        if t.startswith("compare:"):
            _, out, names = t.split(":", 2)
            compare(names.split(","), out)
        else:
            globals()[t]()

"""
Đánh giá các phương pháp tô màu trên COCO val2017 (256 x 256) theo các độ đo của bài DDColor: FID, CF, ΔCF, PSNR.
    python src/evaluation/evaluate_coco.py --methods gray welsh cic ddcolor_tiny ddcolor_large [--limit N]
Ảnh kết quả: results/coco256/<phương_pháp>/<id>.png; bảng: results/coco_summary.csv, results/coco_details.csv
"""
import os, sys, argparse, time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from src.models.lab import rgb_to_lab, lab_to_rgb
from src.models import welsh
from src.utils.metrics import psnr, colorfulness

GT = os.path.join(ROOT, "data", "coco256", "gt")


def load(p):
    return np.asarray(Image.open(p).convert("RGB"))


def run_method(name, files, out_dir, args):
    os.makedirs(out_dir, exist_ok=True)
    times = []
    if name == "gray":
        for f in tqdm(files, desc=name):
            lab = rgb_to_lab(load(os.path.join(GT, f)) / 255.0); lab[..., 1:] = 0
            Image.fromarray(np.clip(np.rint(lab_to_rgb(lab) * 255), 0, 255).astype(np.uint8)).save(os.path.join(out_dir, f))
    elif name == "pseudocolor":
        from src.models.pseudocolor import sine_transform
        for f in tqdm(files, desc=name):
            g = load(os.path.join(GT, f)); L = rgb_to_lab(g / 255.0)[..., 0]
            gray = np.clip(np.rint(L / 100 * 255), 0, 255).astype(np.uint8)
            t0 = time.perf_counter(); out = sine_transform(gray); times.append((time.perf_counter() - t0) * 1000)
            Image.fromarray(out).save(os.path.join(out_dir, f))
    elif name == "welsh":
        # Ảnh mẫu: ảnh khác trong kho có ảnh xám giống nhất (không dùng chính ảnh đó)
        pool = sorted(os.listdir(GT))
        labs = {}
        desc = np.stack([welsh.gray_descriptor(rgb_to_lab(load(os.path.join(GT, f)) / 255.0)[..., 0]) for f in tqdm(pool, desc="descriptor")])
        index = {f: i for i, f in enumerate(pool)}
        for f in tqdm(files, desc=name):
            gt = load(os.path.join(GT, f)) / 255.0
            L = rgb_to_lab(gt)[..., 0]
            sim = desc @ desc[index[f]]; sim[index[f]] = -9
            ref = pool[int(np.argmax(sim))]
            t0 = time.perf_counter()
            out = welsh.colorize(L, load(os.path.join(GT, ref)) / 255.0)
            times.append((time.perf_counter() - t0) * 1000)
            Image.fromarray(np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)).save(os.path.join(out_dir, f))
    elif name == "cic":
        from src.models.cic import CICRunner
        r = CICRunner()
        for f in tqdm(files, desc=name):
            gt = load(os.path.join(GT, f)) / 255.0
            lab = rgb_to_lab(gt)
            t0 = time.perf_counter(); lab[..., 1:] = r.colorize_L(lab[..., 0]); times.append((time.perf_counter() - t0) * 1000)
            Image.fromarray(np.clip(np.rint(lab_to_rgb(lab) * 255), 0, 255).astype(np.uint8)).save(os.path.join(out_dir, f))
    elif name.startswith("ddcolor"):
        from src.models.ddcolor_runner import DDColorRunner
        variant = name.split("_")[1]
        r = DDColorRunner(variant, input_size=args.input_size)
        print(name, "params", r.n_params(), "missing", len(r.missing), "unexpected", len(r.unexpected))
        r.colorize(load(os.path.join(GT, files[0])))
        for f in tqdm(files, desc=name):
            out, ms = r.colorize(load(os.path.join(GT, f))); times.append(ms)
            Image.fromarray(out).save(os.path.join(out_dir, f))
    return float(np.mean(times)) if times else 0.0


def fid(dir1, dir2):
    import torch_fidelity
    m = torch_fidelity.calculate_metrics(input1=dir1, input2=dir2, cuda=True, fid=True, verbose=False)
    return m["frechet_inception_distance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["gray", "welsh", "cic", "ddcolor_tiny", "ddcolor_large"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--input_size", type=int, default=256)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    files = sorted(os.listdir(GT))
    if args.limit:
        files = files[:args.limit]
    gt_dir = GT if not args.limit else os.path.join(ROOT, "results", f"gt_subset_{args.limit}")
    if args.limit and not os.path.isdir(gt_dir):
        os.makedirs(gt_dir)
        for f in files:
            Image.open(os.path.join(GT, f)).save(os.path.join(gt_dir, f))
    cf_gt = {f: colorfulness(load(os.path.join(GT, f))) for f in files}
    summ_path = os.path.join(ROOT, "results", f"coco_summary{args.tag}.csv")
    summary = pd.read_csv(summ_path).set_index("method").to_dict("index") if os.path.exists(summ_path) else {}
    details = []
    for m in args.methods:
        name = m + (f"_{args.input_size}" if m.startswith("ddcolor") else "")
        out_dir = os.path.join(ROOT, "results", "coco256" + args.tag, name)
        ms = run_method(m, files, out_dir, args)
        ps, cf = [], []
        for f in files:
            p = load(os.path.join(out_dir, f)); g = load(os.path.join(GT, f))
            ps.append(psnr(p, g)); cf.append(colorfulness(p))
            details.append({"method": name, "image": f, "psnr": ps[-1], "cf": cf[-1], "cf_gt": cf_gt[f]})
        fd = fid(out_dir, gt_dir)
        summary[name] = {"n": len(files), "FID": round(fd, 2), "CF": round(float(np.mean(cf)), 2),
                         "dCF": round(abs(float(np.mean(cf)) - float(np.mean(list(cf_gt.values())))), 2),
                         "PSNR": round(float(np.mean(ps)), 2), "ms_per_image": round(ms, 1)}
        print(name, summary[name], flush=True)
        pd.DataFrame.from_dict(summary, orient="index").rename_axis("method").to_csv(summ_path)
    summary["ground_truth"] = {"n": len(files), "FID": 0.0, "CF": round(float(np.mean(list(cf_gt.values()))), 2), "dCF": 0.0, "PSNR": float("nan"), "ms_per_image": 0}
    pd.DataFrame.from_dict(summary, orient="index").rename_axis("method").to_csv(summ_path)
    dp = os.path.join(ROOT, "results", f"coco_details{args.tag}.csv")
    old = pd.read_csv(dp) if os.path.exists(dp) else pd.DataFrame()
    new = pd.DataFrame(details)
    if len(old):
        old = old[~old.method.isin(new.method.unique())]
    pd.concat([old, new]).to_csv(dp, index=False)


if __name__ == "__main__":
    main()

"""
Tổng hợp kết quả COCO:
  - PSNR trung bình chỉ tính trên giá trị hữu hạn (ảnh vốn xám cho PSNR vô cực với phương pháp "ảnh xám");
  - CF, PSNR riêng trên 4.865 ảnh vốn có màu;
  - danh sách ảnh DDColor-L làm tệ nhất (PSNR thấp nhất, lệch CF lớn nhất) để phân tích thất bại.
Xuất results/coco_final.csv, results/failure_candidates.csv
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    s = pd.read_csv(os.path.join(ROOT, "results", "coco_summary.csv")).set_index("method")
    d = pd.read_csv(os.path.join(ROOT, "results", "coco_details.csv"))
    meta = pd.read_csv(os.path.join(ROOT, "data", "coco256", "meta.csv"))
    gray_set = set(meta[meta.gray_origin].image.astype(str).str.zfill(12) + ".png")
    d["gray_origin"] = d.image.isin(gray_set)
    rows = []
    for m, g in d.groupby("method", sort=False):
        fin = g[np.isfinite(g.psnr)]
        col = g[~g.gray_origin]
        rows.append({"method": m, "FID": s.loc[m, "FID"], "CF": round(g.cf.mean(), 2), "dCF": round(abs(g.cf.mean() - g.cf_gt.mean()), 2),
                     "PSNR": round(fin.psnr.mean(), 2), "n_inf": int((~np.isfinite(g.psnr)).sum()),
                     "CF_color": round(col.cf.mean(), 2), "PSNR_color": round(col[np.isfinite(col.psnr)].psnr.mean(), 2),
                     "ms": s.loc[m, "ms_per_image"]})
    rows.append({"method": "ground_truth", "FID": 0, "CF": round(d.drop_duplicates("image").cf_gt.mean(), 2), "dCF": 0, "PSNR": np.nan, "n_inf": 0,
                 "CF_color": round(d[~d.gray_origin].drop_duplicates("image").cf_gt.mean(), 2), "PSNR_color": np.nan, "ms": 0})
    out = pd.DataFrame(rows); out.to_csv(os.path.join(ROOT, "results", "coco_final.csv"), index=False); print(out.to_string(index=False))
    m = "ddcolor_large_256"
    if m in d.method.unique():
        g = d[(d.method == m) & (~d.gray_origin)].copy()
        g["dcf_img"] = g.cf - g.cf_gt
        worst = pd.concat([g.nsmallest(12, "psnr").assign(kind="PSNR thấp nhất"),
                           g.nlargest(8, "dcf_img").assign(kind="sặc sỡ hơn ảnh gốc nhiều nhất"),
                           g.nsmallest(8, "dcf_img").assign(kind="nhạt hơn ảnh gốc nhiều nhất")])
        if "cic" in d.method.unique():
            c = d[d.method == "cic"].set_index("image")
            worst["psnr_cic"] = worst.image.map(c.psnr); worst["cf_cic"] = worst.image.map(c.cf)
        worst.to_csv(os.path.join(ROOT, "results", "failure_candidates.csv"), index=False)
        print(worst[["image", "kind", "psnr", "cf", "cf_gt"]].to_string(index=False))
        # tỷ lệ ảnh mà DDColor-L có PSNR cao hơn CIC
        if "cic" in d.method.unique():
            a = d[d.method == m].set_index("image").psnr; b = d[d.method == "cic"].set_index("image").psnr
            print("DDColor-L PSNR > CIC:", int((a > b).sum()), "/", len(a))
        # ảnh gốc gần như không màu (CF < 5) nhưng không bị coi là ảnh xám
        near = d[(d.method == m) & (~d.gray_origin) & (d.cf_gt < 5)]
        n_better = int((a > b).sum()) if "cic" in d.method.unique() else -1
        pd.DataFrame([{"n_gray_origin": len(gray_set), "n_near_gray": len(near), "cf_pred_near_gray": round(near.cf.mean(), 2),
                       "n_psnr_better_than_cic": n_better}]).to_csv(
            os.path.join(ROOT, "results", "failure_stats.csv"), index=False)
        print("ảnh gần như không màu:", len(near))
    # ảnh có ly rượu (vật trong suốt) theo chú thích COCO, nếu đã tải
    ann = os.path.join(ROOT, "data", "raw", "instances_val2017.json")
    if os.path.exists(ann):
        import json
        J = json.load(open(ann, encoding="utf-8"))
        cid = [c["id"] for c in J["categories"] if c["name"] == "wine glass"][0]
        ids = {f"{a['image_id']:012d}.png" for a in J["annotations"] if a["category_id"] == cid}
        rows = []
        for mm, g in d[d.image.isin(ids)].groupby("method", sort=False):
            rows.append({"method": mm, "n": len(g), "psnr": round(g[np.isfinite(g.psnr)].psnr.mean(), 2), "cf": round(g.cf.mean(), 2), "cf_gt": round(g.cf_gt.mean(), 2)})
        pd.DataFrame(rows).to_csv(os.path.join(ROOT, "results", "wine_glass.csv"), index=False); print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

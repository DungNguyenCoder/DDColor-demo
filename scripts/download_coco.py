"""
Tải COCO val2017 (5.000 ảnh, ~778 MB) vào data/raw rồi chạy src/evaluation/prepare_coco.py.
Tải thêm chú thích đối tượng (annotations_trainval2017.zip, ~241 MB), chỉ giữ instances_val2017.json
để lọc ảnh có ly rượu khi phân tích thất bại (src/evaluation/summarize.py).
"""
import os, zipfile, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw"); os.makedirs(RAW, exist_ok=True)

dst = os.path.join(RAW, "val2017.zip")
if not os.path.exists(dst):
    urllib.request.urlretrieve("http://images.cocodataset.org/zips/val2017.zip", dst)
print("xong:", dst)

ann = os.path.join(RAW, "instances_val2017.json")
if not os.path.exists(ann):
    z = os.path.join(RAW, "annotations_trainval2017.zip")
    urllib.request.urlretrieve("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", z)
    with zipfile.ZipFile(z) as f, open(ann, "wb") as out:
        out.write(f.read("annotations/instances_val2017.json"))
    os.remove(z)
print("xong:", ann)

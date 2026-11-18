"""
Tải COCO val2017 (5.000 ảnh, ~778 MB) vào data/raw rồi chạy src/evaluation/prepare_coco.py.
Tải thêm chú thích đối tượng (annotations_trainval2017.zip, ~241 MB), chỉ giữ instances_val2017.json
để lọc ảnh có ly rượu khi phân tích thất bại (src/evaluation/summarize.py).
"""
import os, zipfile, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")


def fetch(url, dst):
    """Tải vào tệp tạm rồi đổi tên; tệp zip được kiểm tra đọc được trước khi giữ lại."""
    tmp = dst + ".part"
    urllib.request.urlretrieve(url, tmp)
    with zipfile.ZipFile(tmp) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"Tệp zip hỏng ở {bad}: {url}")
    os.replace(tmp, dst)


if __name__ == "__main__":
    os.makedirs(RAW, exist_ok=True)
    dst = os.path.join(RAW, "val2017.zip")
    if not os.path.exists(dst):
        fetch("http://images.cocodataset.org/zips/val2017.zip", dst)
    print("xong:", dst)

    ann = os.path.join(RAW, "instances_val2017.json")
    if not os.path.exists(ann):
        z = os.path.join(RAW, "annotations_trainval2017.zip")
        if not os.path.exists(z):
            fetch("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", z)
        with zipfile.ZipFile(z) as f, open(ann + ".part", "wb") as out:
            out.write(f.read("annotations/instances_val2017.json"))
        os.replace(ann + ".part", ann)
        os.remove(z)
    print("xong:", ann)

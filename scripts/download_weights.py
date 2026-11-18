"""Tải trọng số chính thức: DDColor (Hugging Face piddnad/*) và CIC (Zhang et al.), kiểm tra mã băm SHA-256."""
import os, hashlib, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# mã băm của pytorch_model.bin theo Hugging Face (trường lfs.sha256) và của tệp CIC
SHA256 = {
    "ddcolor_paper_tiny": "8a1277bc90a1bfbb6d2d83933a9a6bc821931879ca93e26e4fcec12165d41fce",
    "ddcolor_paper": "5c77db8292398a2ec1397003f0a90266c28e97b013515d5e9a32e73a4c54a67d",
    "ddcolor_modelscope": "d81711971ec59200da26d5e8a1afae8dd3778d495ea8ad7a7dadc769f403f7e7",
    "cic": "9b330a0bae53f4ded77b1e23defbf78beaa09c10ebc4c4999e8e4f4a160b93f9",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get(url, dst, digest=None):
    if os.path.exists(dst) and (digest is None or sha256(dst) == digest):
        print("đã có", dst); return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("tải", url)
    tmp = dst + ".part"                     # tải vào tệp tạm, xong mới đổi tên: bị ngắt giữa chừng thì lần sau tải lại
    urllib.request.urlretrieve(url, tmp)
    if digest is not None and sha256(tmp) != digest:
        os.remove(tmp)
        raise RuntimeError(f"Sai mã băm SHA-256: {url}")
    os.replace(tmp, dst)


if __name__ == "__main__":
    for repo in ["ddcolor_paper_tiny", "ddcolor_paper", "ddcolor_modelscope"]:
        base = f"https://huggingface.co/piddnad/{repo}/resolve/main/"
        get(base + "config.json", os.path.join(ROOT, "weights", repo, "config.json"))
        get(base + "pytorch_model.bin", os.path.join(ROOT, "weights", repo, "pytorch_model.bin"), SHA256[repo])
    get("https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
        os.path.join(ROOT, "weights", "cic", "colorization_release_v2-9b330a0b.pth"), SHA256["cic"])
    print("xong")

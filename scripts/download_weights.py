"""Tải trọng số chính thức: DDColor (Hugging Face piddnad/*) và CIC (Zhang et al.)."""
import os, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FILES = {
    "ddcolor_paper_tiny": ["pytorch_model.bin", "config.json"],
    "ddcolor_paper": ["pytorch_model.bin", "config.json"],
    "ddcolor_modelscope": ["pytorch_model.bin", "config.json"],
}


def get(url, dst):
    if os.path.exists(dst):
        print("đã có", dst); return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    print("tải", url); urllib.request.urlretrieve(url, dst)


for repo, files in FILES.items():
    for f in files:
        get(f"https://huggingface.co/piddnad/{repo}/resolve/main/{f}", os.path.join(ROOT, "weights", repo, f))
get("https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
    os.path.join(ROOT, "weights", "cic", "colorization_release_v2-9b330a0b.pth"))

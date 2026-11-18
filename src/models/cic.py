"""
CIC – "Colorful Image Colorization" (Zhang, Isola, Efros, ECCV 2016), mốc CNN có trong Bảng 1 của bài DDColor.
Kiến trúc viết lại theo mã nguồn công bố của tác giả (github.com/richzhang/colorization, giấy phép BSD-2),
nạp trọng số chính thức colorization_release_v2. Mạng dự đoán phân bố xác suất trên 313 ô màu ab rồi lấy kỳ vọng.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _block(cin, cout, n, last_stride=1, dilation=1):
    layers = []
    for i in range(n):
        s = last_stride if i == n - 1 else 1
        layers += [nn.Conv2d(cin if i == 0 else cout, cout, 3, stride=s, padding=dilation, dilation=dilation, bias=True), nn.ReLU(True)]
    return layers


class ECCVGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = nn.Sequential(*_block(1, 64, 2, last_stride=2), nn.BatchNorm2d(64))
        self.model2 = nn.Sequential(*_block(64, 128, 2, last_stride=2), nn.BatchNorm2d(128))
        self.model3 = nn.Sequential(*_block(128, 256, 3, last_stride=2), nn.BatchNorm2d(256))
        self.model4 = nn.Sequential(*_block(256, 512, 3), nn.BatchNorm2d(512))
        self.model5 = nn.Sequential(*_block(512, 512, 3, dilation=2), nn.BatchNorm2d(512))
        self.model6 = nn.Sequential(*_block(512, 512, 3, dilation=2), nn.BatchNorm2d(512))
        self.model7 = nn.Sequential(*_block(512, 512, 3), nn.BatchNorm2d(512))
        self.model8 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=True), nn.ReLU(True),
                                    nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.ReLU(True),
                                    nn.Conv2d(256, 256, 3, 1, 1, bias=True), nn.ReLU(True),
                                    nn.Conv2d(256, 313, 1, 1, 0, bias=True))
        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, 1, padding=0, bias=False)

    def forward(self, L):
        x = (L - 50.0) / 100.0
        for m in [self.model1, self.model2, self.model3, self.model4, self.model5, self.model6, self.model7, self.model8]:
            x = m(x)
        ab = self.model_out(self.softmax(x)) * 110.0
        return F.interpolate(ab, scale_factor=4, mode="bilinear", align_corners=False)


class CICRunner:
    def __init__(self, weight=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ECCVGenerator()
        weight = weight or os.path.join(ROOT, "weights", "cic", "colorization_release_v2-9b330a0b.pth")
        self.model.load_state_dict(torch.load(weight, map_location="cpu"), strict=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def colorize_L(self, L):
        """L: HxW (0..100). Mạng chạy ở 256x256, ab được phóng về kích thước gốc. Trả về ab HxWx2."""
        h, w = L.shape
        t = torch.from_numpy(L.astype(np.float32))[None, None].to(self.device)
        t256 = F.interpolate(t, size=(256, 256), mode="bicubic", align_corners=False)
        ab = self.model(t256)
        ab = F.interpolate(ab, size=(h, w), mode="bilinear", align_corners=False)
        return ab[0].permute(1, 2, 0).cpu().numpy().astype(np.float64)

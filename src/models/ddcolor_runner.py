"""
Chạy DDColor (ICCV 2023) bằng mã nguồn và trọng số chính thức (third_party/DDColor, weights/<tên>).
Chỉ suy luận, không huấn luyện lại.
"""
import os, sys, json, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DD_DIR = os.path.join(ROOT, "third_party", "DDColor")
if DD_DIR not in sys.path:
    sys.path.insert(0, DD_DIR)
from ddcolor import DDColor  # noqa: E402

MODELS = {"tiny": "ddcolor_paper_tiny", "large": "ddcolor_paper", "modelscope": "ddcolor_modelscope", "artistic": "ddcolor_artistic"}


def diverse_queries(maps, k=5):
    """
    Chọn k color query đại diện: query đầu tiên có độ biến thiên không gian lớn nhất,
    các query sau được chọn tham lam sao cho tương quan lớn nhất với các query đã chọn là nhỏ nhất.
    maps: K x H x W (tích vô hướng query · đặc trưng ảnh).
    """
    act = 1 / (1 + np.exp(-maps))
    flat = act.reshape(act.shape[0], -1)
    z = (flat - flat.mean(1, keepdims=True)) / (flat.std(1, keepdims=True) + 1e-8)
    corr = z @ z.T / z.shape[1]
    chosen = [int(np.argmax(flat.std(1)))]
    while len(chosen) < k:
        score = np.max(np.abs(corr[:, chosen]), axis=1)
        score[chosen] = np.inf
        chosen.append(int(np.argmin(score)))
    return chosen, act


class DDColorRunner:
    def __init__(self, name="large", input_size=512, device=None):
        self.name = MODELS.get(name, name)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wdir = os.path.join(ROOT, "weights", self.name)
        cfg = json.load(open(os.path.join(wdir, "config.json"), encoding="utf-8"))
        cfg["input_size"] = [input_size, input_size]
        self.input_size = input_size
        self.model = DDColor(**cfg)
        sd = torch.load(os.path.join(wdir, "pytorch_model.bin"), map_location="cpu")
        sd = sd.get("params", sd) if isinstance(sd, dict) else sd
        res = self.model.load_state_dict(sd, strict=False)
        self.missing, self.unexpected = list(res.missing_keys), list(res.unexpected_keys)
        self.model.to(self.device).eval()
        self._query_maps = None
        self.model.decoder.color_decoder.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._query_maps = out.detach()

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def colorize(self, rgb_u8, return_maps=False):
        """
        rgb_u8: HxWx3 uint8 (ảnh màu hoặc xám; chỉ dùng kênh L).
        Theo pipeline chính thức: lấy L của ảnh gốc, đưa ảnh xám (L, 0, 0) cỡ input_size vào mạng,
        phóng ab về kích thước gốc rồi ghép với L gốc.
        """
        img = rgb_u8.astype(np.float32) / 255.0
        h, w = img.shape[:2]
        orig_l = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, :1]
        small = cv2.resize(img, (self.input_size, self.input_size))
        l_small = cv2.cvtColor(small, cv2.COLOR_RGB2Lab)[:, :, :1]
        gray = cv2.cvtColor(np.concatenate([l_small, np.zeros_like(l_small), np.zeros_like(l_small)], -1), cv2.COLOR_LAB2RGB)
        t = torch.from_numpy(gray.transpose(2, 0, 1))[None].float().to(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        ab = self.model(t)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        ab = F.interpolate(ab.float(), size=(h, w))[0].cpu().numpy().transpose(1, 2, 0)
        lab = np.concatenate([orig_l, ab], -1)
        out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255.0, 0, 255).round().astype(np.uint8)
        if return_maps:
            return out, ms, self._query_maps[0].cpu().numpy()
        return out, ms

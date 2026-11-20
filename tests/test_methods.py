"""Kiểm thử phương pháp Welsh tự cài đặt, CIC và DDColor."""
import os, sys, unittest, hashlib
import numpy as np
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.lab import rgb_to_lab, lab_to_rgb
from src.models import welsh


class TestWelsh(unittest.TestCase):
    def test_box_mean_hand(self):
        x = np.arange(25, dtype=float).reshape(5, 5)
        self.assertAlmostEqual(welsh.box_mean(x, 3)[2, 2], 12.0)          # trung bình 3x3 quanh tâm
        self.assertAlmostEqual(welsh.local_std(np.full((6, 6), 7.0), 5).max(), 0.0)

    def test_keeps_luminance(self):
        rng = np.random.default_rng(0)
        L = rng.random((40, 50)) * 100
        out = welsh.colorize(L, rng.random((30, 30, 3)))
        # L được giữ nguyên trong Lab; màu ngoài gam RGB bị cắt khi đổi về RGB nên L lệch nhẹ ở vài điểm
        self.assertLess(np.abs(rgb_to_lab(out)[..., 0] - L).mean(), 1.5)

    def test_transfers_matching_colors(self):
        # Ảnh mẫu: nửa trên tối màu xanh lam, nửa dưới sáng màu vàng.
        ref = np.zeros((40, 40, 3)); ref[:20] = [0.1, 0.1, 0.6]; ref[20:] = [0.95, 0.9, 0.2]
        ref_lab = rgb_to_lab(ref)
        L = np.zeros((20, 20)); L[:10] = ref_lab[0, 0, 0]; L[10:] = ref_lab[-1, 0, 0]
        out_lab = rgb_to_lab(welsh.colorize(L, ref, n_samples=100))
        self.assertLess(out_lab[2, 2, 2], -20)      # vùng tối nhận b* âm (lam)
        self.assertGreater(out_lab[-3, 2, 2], 40)   # vùng sáng nhận b* dương (vàng)

    def test_descriptor_unit_norm(self):
        d = welsh.gray_descriptor(np.random.default_rng(1).random((64, 80)) * 100)
        self.assertAlmostEqual(float(np.linalg.norm(d)), 1.0, places=6)


class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for w in ("weights/cic/colorization_release_v2-9b330a0b.pth", "weights/ddcolor_paper_tiny/pytorch_model.bin"):
            if not os.path.exists(os.path.join(ROOT, w)):
                raise FileNotFoundError(f"Thiếu {w}: chạy python scripts/download_weights.py trước")
        from src.models.cic import CICRunner
        from src.models.ddcolor_runner import DDColorRunner
        cls.cic = CICRunner()
        cls.dd = DDColorRunner("tiny", input_size=256)

    def test_param_counts_match_paper(self):
        self.assertAlmostEqual(sum(p.numel() for p in self.cic.model.parameters()) / 1e6, 32.2, places=1)
        self.assertAlmostEqual(self.dd.n_params() / 1e6, 55.0, places=1)

    def test_all_weights_loaded(self):
        self.assertEqual((self.dd.missing, self.dd.unexpected), ([], []))

    def test_weight_hash(self):
        p = os.path.join(ROOT, "weights", "ddcolor_paper_tiny", "pytorch_model.bin")
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        self.assertEqual(h, "8a1277bc90a1bfbb6d2d83933a9a6bc821931879ca93e26e4fcec12165d41fce")  # SHA-256 trên Hugging Face

    def test_ddcolor_shape_and_luminance(self):
        rng = np.random.default_rng(2)
        for h, w in [(37, 53), (256, 256), (401, 299)]:
            x = (rng.random((h, w, 3)) * 255).astype(np.uint8)
            out, _ = self.dd.colorize(x)
            self.assertEqual(out.shape, (h, w, 3))
            dl = np.abs(rgb_to_lab(out / 255.0)[..., 0] - rgb_to_lab(x / 255.0)[..., 0])
            self.assertLess(dl.mean(), 0.6)

    def test_gray_and_color_input_same_output(self):
        x = np.asarray(Image.open(os.path.join(ROOT, "data", "testcases", "color", "coffee.png")).convert("RGB"))
        lab = rgb_to_lab(x / 255.0); lab[..., 1:] = 0
        g = np.clip(np.rint(lab_to_rgb(lab) * 255), 0, 255).astype(np.uint8)
        a, _ = self.dd.colorize(x); b, _ = self.dd.colorize(g)
        self.assertLess(np.abs(a.astype(int) - b.astype(int)).mean(), 1.5)   # mạng chỉ dùng kênh L

    def test_query_maps(self):
        _, _, maps = self.dd.colorize(np.zeros((64, 64, 3), np.uint8), return_maps=True)
        self.assertEqual(maps.shape[0], 100)                                   # K = 100 color query

    def test_input_size_per_call(self):
        # mô hình dựng ở 512 nhưng chạy với input_size=256 phải cho kết quả như mô hình dựng ở 256
        from src.models.ddcolor_runner import DDColorRunner
        x = (np.random.default_rng(5).random((96, 128, 3)) * 255).astype(np.uint8)
        other = DDColorRunner("tiny", input_size=512, device=self.dd.device)
        a, _ = self.dd.colorize(x)
        b, _ = other.colorize(x, input_size=256)
        self.assertLessEqual(np.abs(a.astype(int) - b.astype(int)).max(), 1)

    def test_cic_output_range(self):
        ab = self.cic.colorize_L(np.full((50, 70), 60.0))
        self.assertEqual(ab.shape, (50, 70, 2)); self.assertLess(np.abs(ab).max(), 110)


if __name__ == "__main__":
    unittest.main()

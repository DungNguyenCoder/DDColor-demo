"""Kiểm thử chuyển đổi Lab tự cài đặt và các độ đo."""
import os, sys, unittest
import numpy as np
import cv2
from skimage import color as skcolor

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.lab import rgb_to_lab, lab_to_rgb, to_gray_lab
from src.utils.metrics import psnr, colorfulness


class TestLab(unittest.TestCase):
    def test_known_colors(self):
        # Giá trị tham chiếu: trắng (100, 0, 0), đen (0, 0, 0), đỏ thuần ≈ (53,24; 80,09; 67,20)
        lab = rgb_to_lab(np.array([[[1, 1, 1], [0, 0, 0], [1, 0, 0]]], float))[0]
        np.testing.assert_allclose(lab[0], [100, 0, 0], atol=0.01)
        np.testing.assert_allclose(lab[1], [0, 0, 0], atol=0.01)
        np.testing.assert_allclose(lab[2], [53.24, 80.09, 67.20], atol=0.05)

    def test_gray_has_zero_ab(self):
        lab = rgb_to_lab(np.array([[[0.3, 0.3, 0.3], [0.8, 0.8, 0.8]]]))
        np.testing.assert_allclose(lab[..., 1:], 0, atol=1e-6)

    def test_matches_skimage(self):
        x = np.random.default_rng(0).random((64, 64, 3))
        np.testing.assert_allclose(rgb_to_lab(x), skcolor.rgb2lab(x), atol=0.02)

    def test_close_to_opencv(self):
        # OpenCV dùng công thức xấp xỉ: lệch skimage và bản tự cài tới khoảng 0,38 đơn vị ab
        x = np.random.default_rng(1).random((64, 64, 3)).astype(np.float32)
        self.assertLess(np.abs(rgb_to_lab(x) - cv2.cvtColor(x, cv2.COLOR_RGB2Lab)).max(), 0.5)

    def test_round_trip(self):
        x = np.random.default_rng(2).random((64, 64, 3))
        self.assertLess(np.abs(lab_to_rgb(rgb_to_lab(x)) - x).max(), 1e-6)

    def test_gray_input_keeps_L(self):
        x = np.random.default_rng(3).random((32, 32, 3))
        g = to_gray_lab(x)
        np.testing.assert_allclose(rgb_to_lab(g)[..., 0], rgb_to_lab(x)[..., 0], atol=0.05)
        self.assertLess(np.abs(g[..., 0] - g[..., 1]).max(), 1e-6)


class TestMetrics(unittest.TestCase):
    def test_colorfulness_gray_is_zero(self):
        self.assertAlmostEqual(colorfulness(np.full((8, 8, 3), 120, np.uint8)), 0.0)

    def test_colorfulness_hand_example(self):
        # Ảnh đỏ thuần đều: rg = 255, yb = 127,5; σ = 0 nên CF = 0,3 · √(255² + 127,5²) ≈ 85,53
        self.assertAlmostEqual(colorfulness(np.tile(np.array([255, 0, 0], np.uint8), (4, 4, 1))), 85.53, places=2)

    def test_psnr_hand_example(self):
        a = np.full((10, 10, 3), 100, np.uint8)
        self.assertAlmostEqual(psnr(a + 10, a), 28.13, places=2)




class TestImageIO(unittest.TestCase):
    """Hàm đọc ảnh dùng chung của giao diện (src/utils/image_io.py)."""

    def test_alpha_on_white_and_resize(self):
        import io
        from PIL import Image
        from src.utils.image_io import read_image
        im = Image.new("RGBA", (3000, 1000), (255, 0, 0, 0))            # hoàn toàn trong suốt -> nền trắng
        b = io.BytesIO(); im.save(b, format="PNG"); b.seek(0)
        a, orig = read_image(b)
        self.assertEqual(orig, (3000, 1000)); self.assertEqual(a.shape, (683, 2048, 3))
        self.assertEqual(int(a.min()), 255)

    def test_16bit_keeps_brightness(self):
        import io
        from PIL import Image
        from src.utils.image_io import read_image
        g = (np.arange(256, dtype=np.uint16).reshape(16, 16) * 257)
        b = io.BytesIO(); Image.fromarray(g).save(b, format="PNG"); b.seek(0)
        a, _ = read_image(b)
        np.testing.assert_array_equal(a[..., 0], np.arange(256).reshape(16, 16))


if __name__ == "__main__":
    unittest.main()

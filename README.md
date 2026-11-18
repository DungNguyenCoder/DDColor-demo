# DDColor-demo – Tô màu ảnh xám với DDColor (ICCV 2023)

Bài tập lớn học phần **Xử lý ảnh (INT13146)** – Học viện Công nghệ Bưu chính Viễn thông, nhóm 15.

Đề tài: *[ICCV'23] DDColor: Towards Photo-Realistic Image Colorization via Dual Decoders*
(Kang và cộng sự, ICCV 2023, tr. 328–338 – [arXiv 2212.11613](https://arxiv.org/abs/2212.11613), mã nguồn gốc [piddnad/DDColor](https://github.com/piddnad/DDColor)).

Dự án giải thích và kiểm chứng DDColor, so sánh với:
- **Giả màu** (Chương 3 của học phần) – tự cài đặt bằng NumPy;
- **Welsh 2002** (tô màu theo ảnh mẫu trong không gian Lab) – tự cài đặt bằng NumPy;
- **CIC – Zhang 2016** (mạng CNN, có trong Bảng 1 của bài DDColor) – trọng số chính thức.

## Cấu trúc

| Thư mục | Nội dung |
|---|---|
| `src/models/lab.py` | sRGB ↔ CIELAB tự cài đặt |
| `src/models/pseudocolor.py`, `src/models/welsh.py` | Giả màu, Welsh 2002 (NumPy) |
| `src/models/cic.py`, `src/models/ddcolor_runner.py` | CIC và DDColor (suy luận) |
| `src/utils/metrics.py` | PSNR, độ sặc sỡ CF (Hasler & Süsstrunk 2003) |
| `src/evaluation/` | Chuẩn bị dữ liệu, đánh giá COCO (FID/CF/ΔCF/PSNR), thí nghiệm phụ, kiểm thử chức năng, vẽ hình |
| `src/app/app.py` | Demo Streamlit |
| `tests/` | Kiểm thử tự động (unittest) |
| `third_party/DDColor/` | Mã nguồn gốc của tác giả (Apache-2.0), giữ nguyên |
| `data/testcases/` | Ảnh test case (ảnh mẫu scikit-image) |
| `results/`, `figures/` | Bảng số liệu và hình |
| `report/` | Script dựng báo cáo Word |

## Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/download_weights.py      # DDColor tiny/large/modelscope (~2 GB) + CIC (~130 MB)
```

## Chạy

```bash
# Demo (hoặc nháy đúp run_app.bat: tự mở trình duyệt), mở http://localhost:8501
python -m streamlit run src/app/app.py

# Kiểm thử tự động (23 trường hợp)
python -m unittest discover -s tests

# Đánh giá trên COCO val2017 (5.000 ảnh) và chú thích đối tượng
python scripts/download_coco.py
python src/evaluation/prepare_coco.py
python src/evaluation/evaluate_coco.py --methods gray pseudocolor welsh cic ddcolor_tiny ddcolor_large ddcolor_modelscope
python src/evaluation/evaluate_coco.py --methods ddcolor_large --input_size 512
python src/evaluation/summarize.py

# Thí nghiệm phụ, kiểm thử chức năng, hình
python src/evaluation/prepare_testcases.py
python src/evaluation/experiments.py queries robustness timing welsh_ref
python src/evaluation/draw_figures.py pipeline ambiguity testcases compare_coco failures
python src/evaluation/run_functional_tests.py   # 16 test case chức năng, gồm chạy giao diện

# Báo cáo Word (đường dẫn ra mặc định trong report/build_report.py). Trên Windows có Microsoft Word,
# script tự tính sẵn mục lục, danh mục bảng/hình (report/finalize_word.ps1) và xuất kèm bản PDF xem trước.
python report/build_report.py
```

## Kết quả chính (5.000 ảnh COCO val2017, 256 × 256)

Xem `results/coco_final.csv`. FID thấp hơn là tốt hơn; ΔCF là chênh lệch độ sặc sỡ trung bình so với ảnh gốc.

| Phương pháp | FID | ΔCF | PSNR (dB) |
|---|---|---|---|
| CIC (Zhang 2016) | 30,30 | 4,16 | 22,38 |
| DDColor-T | 7,19 | 2,58 | 23,09 |
| DDColor-L | 6,47 | 2,54 | 23,45 |

## Ghi chú

- Chỉ suy luận bằng trọng số chính thức của tác giả; không huấn luyện lại DDColor.
- `.streamlit/config.toml` tắt bước hỏi email ở lần chạy Streamlit đầu tiên; `scripts/download_weights.py` kiểm tra mã băm SHA-256 của trọng số.
- Dữ liệu COCO và trọng số không đưa lên git (xem `.gitignore`); tải lại bằng `scripts/`.
- Nhóm dùng công cụ AI hỗ trợ viết mã; mọi kết quả đã được chạy lại và kiểm chứng bằng `tests/` và các script đánh giá.

# Music Mood RecSys

Hệ gợi ý nhạc theo **tâm trạng (mood)** dựa trên phản hồi ngầm (implicit feedback), dùng thuật toán **ALS** (Alternating Least Squares).

## 1. Mục tiêu
- Ghép dữ liệu **Music Info** (có tag Last.fm) với **Spotify Tracks** (đặc trưng âm thanh).
- Gán nhãn tâm trạng sơ bộ cho bài hát từ tag Last.fm.
- Huấn luyện ALS trên lịch sử nghe (playcount) để gợi ý bài hát.
- Đánh giá bằng giao thức thống nhất (xem `docs/evaluation_protocol.md`).

## 2. Cấu trúc thư mục
```
music-mood-recsys/
├── data/
│   ├── raw/                 # Dữ liệu gốc (KHÔNG commit)
│   └── processed/           # tracks.parquet, sample_interactions.parquet, track_mood_labels.parquet
├── notebooks/               # 01_merge_data, 02_tag_analysis, 03_als_baseline, ...
├── src/
│   ├── data.py              # Đọc/ghép dữ liệu (Thy)
│   ├── mood.py              # Gán nhãn mood từ tag (Như)
│   ├── split.py             # Chia train/test theo giao thức (Hân)
│   ├── model.py             # Huấn luyện & gợi ý ALS
│   ├── baselines.py         # Popularity, Random
│   ├── synthetic.py         # Dữ liệu giả để chạy thử khi chưa có dữ liệu thật
│   └── evaluate.py          # Các metric xếp hạng
├── configs/
│   └── mood_mapping.json    # Ánh xạ tag -> nhóm mood (Như)
├── docs/
│   ├── als_summary.md       # Tóm tắt ALS
│   └── evaluation_protocol.md  # Giao thức đánh giá
├── tests/
├── CONTRACTS.md             # Hợp đồng giữa các thành viên (schema, hàm)
├── requirements.txt
└── README.md
```

## 3. Cài đặt
```bash
git clone <repo-url>
cd music-mood-recsys
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Dữ liệu
Đặt file gốc vào `data/raw/` (xem hướng dẫn tải trong notebook `01_merge_data`).
File đầu ra dùng chung nằm ở `data/processed/`, schema chi tiết ở [CONTRACTS.md](CONTRACTS.md).

## 5. Quy trình chạy
1. `notebooks/01_merge_data.ipynb` → `tracks.parquet`, `sample_interactions.parquet`
2. `notebooks/02_tag_analysis.ipynb` → `mood_mapping.json`, `track_mood_labels.parquet`
3. `src/split.py` → chia train/test theo giao thức
4. `notebooks/03_als_baseline.ipynb` → baseline, ALS, chỉnh siêu tham số, bảng kết quả
5. `src/evaluate.py` → Recall@K, NDCG@K, ...

Chạy thử khi chưa có dữ liệu thật: notebook `03` tự dùng dữ liệu giả (`src/synthetic.py`); kết quả chỉ để kiểm tra code.

Kiểm thử: `python -m pytest -q`

## 6. Phân công
| Thành viên | Phần việc |
|---|---|
| Thy | Tải & ghép dữ liệu, `tracks.parquet`, `sample_interactions.parquet`, bảng thống kê |
| Hân | Khung repo, tóm tắt ALS, `CONTRACTS.md`, giao thức đánh giá |
| Như | Phân tích tag Last.fm, `mood_mapping.json`, `track_mood_labels.parquet` |

## 7. Quy ước làm việc với Git
- Nhánh chính: `main` (chỉ merge qua Pull Request).
- Nhánh cá nhân: `thy/<tên-việc>`, `han/<tên-việc>`, `nhu/<tên-việc>`.
- Không commit dữ liệu lớn; chỉ commit file mẫu nhỏ (< 10 MB).
- Đổi schema trong `CONTRACTS.md` phải báo cả nhóm trước khi merge.

## 8. Tài liệu tham khảo
- Hu, Koren, Volinsky (2008). *Collaborative Filtering for Implicit Feedback Datasets*. ICDM.
- Thư viện `implicit`: https://benfred.github.io/implicit/

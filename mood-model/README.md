# Module Tâm trạng — Hệ gợi ý nhạc theo gu và tâm trạng

Phần việc của TV3. Học hồ sơ tâm trạng cho từng bài từ tag Last.fm và đặc trưng âm thanh Spotify, cung cấp hàm chấm độ hợp tâm trạng cho mô hình kết hợp.

## Bắt đầu

1. Đặt dữ liệu vào `data/raw/`:
   - `Music_Info.csv` (bắt buộc)
   - `User_Listening_History.csv` (cần cho task T6)
   - Nếu TV1 đã xuất `tracks.parquet`, `train.parquet`, `test.parquet`, `item_map.parquet`, `user_map.parquet` thì đặt vào `data/processed/`.
2. Cài thư viện: `pip install -r requirements.txt`
3. Mở Claude Code trong thư mục này và nói: **"Đọc CLAUDE.md rồi bắt đầu task T0"**.

## Tài liệu

| File | Nội dung |
|---|---|
| `CLAUDE.md` | Chỉ dẫn cho Claude Code |
| `docs/project_context.md` | Workflow nhóm, vai trò module |
| `docs/eda_summary.md` | Số liệu EDA liên quan |
| `docs/mood_spec.md` | Đặc tả các cải tiến M1–M8 |
| `docs/data_contracts.md` | File đầu ra và hàm cho TV2 |
| `docs/decisions.md` | Nhật ký quyết định |
| `TASKS.md` | Danh sách task T0–T8 |
| `configs/mood_config.yaml` | Cấu hình |
| `reports/RESULTS.md` | Kết quả |

## Cách TV2 dùng module này

Chạy lần lượt để sinh đủ file (mỗi bước đều có `--sample` để thử nhanh):

```bash
python -m src.mood.io               # kiểm tra dữ liệu
python -m src.mood.tag_audit        # T1: tag_audit.csv, suspect_tracks.csv
python -m src.mood.labeling         # T2: track_mood_labels.parquet
python -m src.mood.train_classifier # T3: models/mood_classifier.joblib
python -m src.mood.predict          # T4: track_mood.parquet  <-- file TV2 cần
python -m src.mood.mood_space       # T4: mood_centroids.json <-- vị trí nhóm trên thanh kéo
```

Sau đó:

```python
from src.mood import api

# 1. Nút chọn nhanh -> điểm trên thanh kéo (tâm học được từ dữ liệu, không phải đặt tay)
valence, energy = api.quick_mood("vui")          # (0.653, 0.643)

# 2. Người dùng tự kéo thanh -> đang nghiêng về nhóm nào
api.slider_to_mood(0.3, 0.9)
# {'cang_thang': 0.55, 'nang_dong': 0.25, 'buon': 0.12, ...}

# 3. Chấm độ hợp tâm trạng cho danh sách bài ứng viên (cùng thứ tự với track_ids)
mood = api.mood_scores(track_ids, valence, energy)   # ndarray [0, 1], 20.000 bài ~25 ms

# 4. Kết hợp với điểm hợp gu của ALS
score = alpha * taste_score + (1 - alpha) * mood
```

## Đã ghép thật với taste-model (Phần II)

`src/mood/taste.py` nạp nguyên `taste-model/src/` bằng importlib (tên gói `taste_src`,
không trùng gói `src` ở đây) nên dùng lại đúng `ALS`, `Popularity` và `combine_scores`
của TV2 — không sửa, không cài lại.

Từ 2026-11-02 mọi đánh giá chạy trên **bộ chia train/test chính thức của TV1**
(`project/data/`, xem `data_preprocessing_report.md`). Thêm `--split local` để tái lập
các số cũ của Phần II.

```bash
python -m src.mood.splits                                           # kiểm tra bộ chia chính thức
python -m src.mood.taste --split official                           # ALS trên toàn bộ tập train
python -m src.mood.evaluate_four_sets --split official --n-per-set 3 --n-users 60000
python -m src.mood.labeling_embed --catalogue official              # A1: nhãn tag bằng embedding
python -m src.mood.evaluate_mood --split official --n-users 20000   # A3: bài ví dụ so với thanh kéo
```

Kết quả chính trên dữ liệu chính thức (chi tiết trong `reports/RESULTS.md` Phần III):

| Câu hỏi | Trả lời bằng số |
|---|---|
| Tâm trạng có thêm gì ngoài gu? | **Có.** Ghép `taste_weight = 0,5` đạt NDCG@3 = 0,6106 ở tập "đúng gu + đúng tâm trạng", hơn chỉ-gu **+24,0%** (18.001 cặp user-nhóm). Mô hình gu một mình gần như không phân biệt được tâm trạng (chênh +0,0217) |
| `taste_weight` nên bằng bao nhiêu? | **0,5** — tốt hơn cả 0,3 và 0,7 ở tập quan trọng nhất |
| Trọng số tin cậy nào cho ALS? | **`engagement_score` của TV1 tốt hơn `playcount`**: 0,6505 so với 0,6106 (+30,4% so với +24,0%). Đề xuất TV2 đưa vào grid-search |
| Thay bảng trọng số tag bằng embedding? | **Không** — macro-F1 tụt 0,5656 → 0,5176; khớp vị trí âm thanh 84,9% → 52,3% |
| Thay thanh kéo bằng chọn 3 bài ví dụ? | **Không** — Recall@20 0,0630 so với 0,0711, kém hơn **−9,0%** so với tọa độ tự đặt |
| Cá nhân hóa mục tiêu lợi bao nhiêu? | **+2,7%** Recall@20 so với tọa độ tự đặt (không phải +12,6% như Phần I — xem mục 18.2, nguyên nhân chính là catalogue 10-core bỏ mất phần đuôi dài) |

Hồ sơ tâm trạng của từng bài (nếu cần dùng trực tiếp):

```python
profiles = api.load_mood_profiles()   # index = track_id
profiles.loc["TRIOREW128F424EAF0", ["mood_pred", "mood_confidence", "mood_source"]]
```

Lưu ý: `mood_scores` trả 0 và ghi warning cho bài không có trong hồ sơ. Thang điểm thực tế nằm
trong khoảng ~0,02–0,65 chứ không trải hết [0, 1] (vì xác suất nhóm của điểm thanh kéo luôn chia
cho 5 nhóm), nên nếu cần cân bằng với điểm hợp gu thì chuẩn hóa hai vế trước khi trộn.

## Mã nhóm tâm trạng

`vui`, `buon`, `thu_gian`, `nang_dong`, `cang_thang` — dùng thống nhất ở mọi file đầu ra.

## Các cải tiến so với bản gốc

1. Nhãn mềm từ tag cảm xúc của người nghe, có xử lý mâu thuẫn.
2. Bộ phân loại từ toàn bộ đặc trưng âm thanh, suy ra tâm trạng cho bài không có tag.
3. Vị trí các nhóm trên thanh kéo học từ dữ liệu.
4. Điểm hợp tâm trạng kết hợp xác suất nhóm và độ gần trên thanh kéo.
5. Cá nhân hóa mục tiêu tâm trạng theo lịch sử nghe của từng người, đánh giá trên dữ liệu thật.

## Lưu ý về số liệu

Phần I và Phần II của `reports/RESULTS.md` chạy trên bộ chia tự làm (che 30%, kho 50.683 bài).
Phần III chạy lại trên bộ chia chính thức của nhóm. Khi báo cáo ra ngoài, **dùng số Phần III**.
Các file Phần III mang hậu tố `_official` nên không ghi đè số cũ.

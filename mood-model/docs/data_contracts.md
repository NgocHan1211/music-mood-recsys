# Hợp đồng dữ liệu của module tâm trạng

Không đổi tên file, tên cột, chữ ký hàm khi chưa hỏi người dùng (TV2, TV1 phụ thuộc vào đây).

## Đầu vào

| File | Nguồn | Cột dùng |
|---|---|---|
| `data/processed/tracks.parquet` (ưu tiên) hoặc `data/raw/Music_Info.csv` | TV1 / Kaggle | `track_id, name, artist, tags, valence, energy, danceability, acousticness, instrumentalness, speechiness, liveness, loudness, tempo, mode, key, duration_ms` |
| `data/raw/User_Listening_History.csv` (chỉ cho M6, tùy chọn) | Kaggle | `user_id, track_id, playcount` |
| `data/processed/train.parquet`, `test.parquet` (nếu TV1 đã có) | TV1 | `uidx, iidx, playcount` (+ `item_map.parquet`, `user_map.parquet` để đổi về ID gốc) |

## Đầu ra

### `data/processed/tag_audit.csv`
`tag, n_tracks, category` (category ∈ genre, decade, country, emotion, other)

### `data/processed/track_mood_labels.parquet` (nhãn từ tag)
| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `track_id` | str | |
| `y_vui, y_buon, y_thu_gian, y_nang_dong, y_cang_thang` | float | nhãn mềm, tổng = 1 (NaN nếu không có tag cảm xúc) |
| `mood_label` | str/None | nhãn cứng nếu đủ tự tin |
| `label_status` | str | `labeled`, `conflict`, `no_emotion_tag`, `suspect` |

### `data/processed/track_mood.parquet` (**TV2 dùng**)
| Cột | Kiểu | Ý nghĩa |
|---|---|---|
| `track_id` | str | khóa nối với dữ liệu nhóm |
| `p_vui, p_buon, p_thu_gian, p_nang_dong, p_cang_thang` | float | xác suất đã hiệu chỉnh, tổng = 1 |
| `mood_pred` | str | nhóm có xác suất cao nhất |
| `mood_confidence` | float | xác suất của `mood_pred` |
| `mood_source` | str | `tag` (có nhãn người nghe) hoặc `model` (suy ra từ âm thanh) |
| `valence, energy` | float | giá trị gốc, để vẽ và tính khoảng cách |

Mọi bài trong `tracks` phải có đúng 1 dòng.

### `data/processed/mood_centroids.json`
```json
{
  "groups": ["vui", "buon", "thu_gian", "nang_dong", "cang_thang"],
  "centroids": {"vui": {"mean": [0.72, 0.68], "cov": [[0.02, 0.0], [0.0, 0.03]], "n": 1234}},
  "manual_baseline": {"vui": [0.80, 0.75]}
}
```
(Số trong ví dụ là giả định, file thật phải sinh từ code.)

### `models/mood_classifier.joblib`
Pipeline sklearn (tiền xử lý + mô hình đã hiệu chỉnh) và danh sách cột đặc trưng.

## Hàm công khai — `src/mood/api.py`

```python
GROUPS = ["vui", "buon", "thu_gian", "nang_dong", "cang_thang"]

def load_mood_profiles() -> pd.DataFrame:
    """Đọc track_mood.parquet, index = track_id."""

def quick_mood(code: str) -> tuple[float, float]:
    """Nút chọn nhanh -> tọa độ (valence, energy) = tâm học được của nhóm."""

def slider_to_mood(valence: float, energy: float) -> dict[str, float]:
    """Điểm thanh kéo -> xác suất thuộc từng nhóm (tổng = 1)."""

def mood_scores(track_ids: Sequence[str], valence: float, energy: float,
                lam: float | None = None) -> np.ndarray:
    """Độ hợp tâm trạng [0, 1] cho từng bài, cùng thứ tự với track_ids."""

def user_mood_target(user_track_ids: Sequence[str], code: str,
                     weights: Sequence[float] | None = None) -> tuple[float, float]:
    """Tâm trạng mục tiêu cá nhân hóa (M6) cho user và nhóm `code`."""

def mood_journey(start: tuple[float, float], end: tuple[float, float],
                 n_steps: int) -> list[tuple[float, float]]:
    """(Tùy chọn, M7) Danh sách điểm mục tiêu đi từ start tới end."""
```

Yêu cầu: `mood_scores` xử lý được 20.000 bài trong < 1 giây; bài không có trong profile trả về 0 và ghi warning.

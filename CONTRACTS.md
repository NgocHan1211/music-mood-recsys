# CONTRACTS: Hợp đồng giữa các thành viên

> Tài liệu này là **nguồn sự thật duy nhất** về định dạng file và hàm dùng chung.
> Trạng thái: **ĐỀ XUẤT (v0.1)**. Thy và Như cần đối chiếu với dữ liệu thực và xác nhận/chỉnh sửa các mục đánh dấu `[CẦN XÁC NHẬN]` trước khi chốt v1.0.
> Quy tắc: muốn đổi schema thì sửa file này trong Pull Request và báo cả nhóm.

## 0. Quy ước chung
- Định dạng lưu: **Parquet** (pyarrow), nén `snappy`, nằm trong `data/processed/`.
- Tên cột: `snake_case`, chữ thường.
- Khóa chính bài hát: **`track_id`** (kiểu `string`), duy nhất trong toàn dự án.
- Khóa chính người dùng: **`user_id`** (kiểu `string`).
- Giá trị thiếu: dùng `null`/`NaN`, không dùng chuỗi rỗng hay `-1`.
- Mã hóa văn bản: UTF-8. Thang thời gian: UTC.
- Seed ngẫu nhiên mặc định: `42`.

## 1. Dữ liệu đầu vào / đầu ra

### 1.1 `data/processed/tracks.parquet`  (Thy)
Mỗi dòng = **một bài hát** đã ghép Music Info với Spotify Tracks.

| Cột | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `track_id` | string | ✅ | Khóa chính (ID trong Music Info) |
| `name` | string | ✅ | Tên bài hát |
| `artist` | string | ✅ | Nghệ sĩ |
| `spotify_id` | string | ⬜ | ID Spotify nếu ghép được, ngược lại null |
| `tags` | string | ⬜ | Tag Last.fm, phân cách bằng dấu phẩy |
| `genre` | string | ⬜ | Thể loại |
| `year` | int | ⬜ | Năm phát hành |
| `duration_ms` | int | ⬜ | Thời lượng |
| `danceability`, `energy`, `valence`, `acousticness`, `instrumentalness`, `liveness`, `speechiness` | float | ⬜ | Đặc trưng âm thanh Spotify (0–1) |
| `loudness`, `tempo` | float | ⬜ | Đặc trưng âm thanh Spotify |
| `matched` | bool | ✅ | True nếu ghép được với Spotify Tracks |
| `match_method` | string | ✅ | `spotify_id` / `name_artist_exact` / `name_artist_fuzzy` / `none` |

Ràng buộc: `track_id` duy nhất, không null. Tất cả bài có tương tác phải có mặt trong file này. `[CẦN XÁC NHẬN]` tên cột thực tế của hai bộ dữ liệu gốc.

### 1.2 `data/processed/sample_interactions.parquet`  (Thy)
Mỗi dòng = **một cặp (user, bài hát)** với số lần nghe.

| Cột | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `user_id` | string | ✅ | Khóa |
| `track_id` | string | ✅ | Khóa, phải tồn tại trong `tracks.parquet` |
| `playcount` | int | ✅ | Số lần nghe, **> 0** |
| `timestamp` | datetime64[ns, UTC] | ⬜ | Chỉ có nếu dữ liệu gốc có. `[CẦN XÁC NHẬN]` |

Ràng buộc: cặp (`user_id`, `track_id`) **duy nhất** (đã gộp playcount nếu trùng). Mẫu nhỏ này được commit lên repo (< 10 MB); bản đầy đủ không commit.

### 1.3 `data/processed/track_mood_labels.parquet`  (Như)
Mỗi dòng = **một bài hát** với nhãn mood sơ bộ.

| Cột | Kiểu | Bắt buộc | Mô tả |
|---|---|---|---|
| `track_id` | string | ✅ | Khóa, tồn tại trong `tracks.parquet` |
| `mood` | string | ⬜ | Mood chính (một trong các khóa của `mood_mapping.json`), null nếu không gán được |
| `mood_score` | float | ⬜ | Độ tin cậy 0–1 của mood chính |
| `mood_all` | string | ⬜ | Các mood khớp, dạng JSON `{"happy":0.6,"calm":0.4}` |
| `n_mood_tags` | int | ✅ | Số tag cảm xúc khớp |

### 1.4 `configs/mood_mapping.json`  (Như)
```json
{
  "version": "0.1",
  "moods": {
    "happy":   ["happy", "feel good", "upbeat", "cheerful"],
    "sad":     ["sad", "melancholy", "heartbreak"],
    "calm":    ["chill", "relaxing", "mellow"],
    "energetic": ["energetic", "party", "workout"]
  }
}
```
Tag được so khớp sau khi chuyển chữ thường và bỏ khoảng trắng thừa. Danh sách mood cuối cùng do Như chốt; Hân/Thy chỉ đọc khóa từ file này, **không hard-code**.

## 2. Hàm dùng chung trong `src/`

Mọi hàm có type hints và docstring. Chữ ký dưới đây là **cam kết**; thêm tham số tùy chọn được, đổi tham số bắt buộc thì phải báo.

### `src/data.py` (Thy)
```python
def load_tracks(path: str = "data/processed/tracks.parquet") -> pd.DataFrame: ...
def load_interactions(path: str = "data/processed/sample_interactions.parquet") -> pd.DataFrame: ...
def build_tracks(music_info_path: str, spotify_path: str) -> pd.DataFrame: ...
def match_stats(tracks: pd.DataFrame) -> pd.DataFrame: ...   # tỷ lệ khớp theo match_method
```

### `src/mood.py` (Như)
```python
def load_mood_mapping(path: str = "configs/mood_mapping.json") -> dict[str, list[str]]: ...
def label_tracks(tracks: pd.DataFrame, mapping: dict[str, list[str]]) -> pd.DataFrame: ...
# trả về DataFrame đúng schema mục 1.3
```

### `src/split.py` (Hân)
```python
def split_interactions(
    interactions: pd.DataFrame,
    strategy: str = "random_holdout",   # "random_holdout" | "leave_one_out" | "temporal"
    test_ratio: float = 0.2,
    min_user_interactions: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]: ...
# trả về (train, test), cùng schema với interactions; mọi user trong test đều có trong train
```

### `src/model.py` (dùng chung)
```python
def build_matrix(
    interactions: pd.DataFrame, transform: str = "log1p", alpha: float = 40.0
) -> tuple[scipy.sparse.csr_matrix, dict, dict]: ...
# trả về (ma trận user x item, user_id->index, track_id->index)

def train_als(matrix, factors=64, regularization=0.01, alpha=1.0, iterations=20, seed=42): ...
# alpha đã được nhân vào ma trận ở build_matrix; để alpha=1.0 ở đây để không nhân hai lần
def recommend(model, matrix, user_index: int, n: int = 10) -> list[tuple[int, float]]: ...
# luôn loại các bài đã có trong train
```

### `src/evaluate.py` (Hân)
```python
def evaluate(
    recommendations: dict[str, list[str]],   # user_id -> danh sách track_id theo thứ tự xếp hạng
    test: pd.DataFrame,
    ks: tuple[int, ...] = (10, 20),
) -> dict[str, float]: ...
# khóa kết quả: "recall@10", "ndcg@10", "precision@10", "map@10", "hitrate@10", "coverage@10"
```

## 3. Quy ước dữ liệu & mô hình
- **Biến đổi playcount** cho ALS: mặc định `log1p`, `confidence = 1 + alpha * log1p(playcount)`. `[CẦN THẢO LUẬN]`
- **Lọc dữ liệu** trước khi chia: giữ user có ≥ `min_user_interactions` tương tác; bài có ≥ 2 user nghe. Giá trị lọc phải được ghi lại trong notebook.
- **Chỉ mục:** `user_index` và `track_index` sinh từ dữ liệu **train**; user/bài chỉ có trong test bị bỏ qua khi đánh giá (cold-start đánh giá riêng).
- **Tái lập:** mọi bước ngẫu nhiên nhận `seed` (mặc định 42).

## 4. Quy trình thay đổi hợp đồng
1. Mở Pull Request sửa `CONTRACTS.md`, tag Thy và Như.
2. Chỉ merge khi cả hai đã approve.
3. Tăng `version` ở đầu file và ghi vào mục lịch sử.

## 5. Lịch sử
| Phiên bản | Ngày | Người | Thay đổi |
|---|---|---|---|
| 0.1 | 2026-10-14 | Hân | Bản đề xuất đầu tiên |

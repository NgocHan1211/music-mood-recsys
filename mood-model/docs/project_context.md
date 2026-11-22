# Bối cảnh dự án

## Đề tài

Hệ thống gợi ý bài hát **vừa hợp gu** (học từ lịch sử nghe thật) **vừa đúng tâm trạng** (người dùng chọn bằng thanh kéo hoặc ô chọn nhanh).

## Dữ liệu

Bộ **Million Song Dataset + Spotify + Last.fm** (Kaggle), gồm 2 file:

| File | Số dòng | Nội dung |
|---|---|---|
| `Music_Info.csv` | 50.683 | Mỗi dòng 1 bài: `track_id, name, artist, spotify_preview_url, spotify_id, tags, genre, year, duration_ms, danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo, time_signature` |
| `User_Listening_History.csv` | 9.711.301 | `track_id, user_id, playcount` |

Module tâm trạng **chỉ cần `Music_Info.csv`** (hoặc `tracks.parquet` do TV1 xuất).

## Workflow chung của nhóm

1. **Ghép, làm sạch, lọc 10/10** (TV1): user ≥ 10 bài, bài ≥ 10 người nghe; cắt playcount ở 60.
2. **Chia user train/test** (TV1): 20% user làm test; mỗi user test, 70% bài làm đầu vào, 30% bị che làm đáp án.
3. **Điểm mức độ chủ động** (TV1): trọng số 0–1 cho mỗi cặp (user, bài).
4. **Mô hình gu** (TV2): ALS với độ tin cậy = 1 + α × điểm chủ động; fold-in cho user test.
5. **Mô hình tâm trạng** (TV3 — module này): học tâm trạng của bài từ tag Last.fm + đặc trưng âm thanh.
6. **Chấm điểm kết hợp** (TV2): `score = α × hợp_gu + (1 − α) × hợp_tâm_trạng`. Phần `hợp_tâm_trạng` lấy từ hàm `mood_scores()` của module này.
7. **Đánh giá** (TV1): Recall/NDCG @10/@20 trên bài bị che; đánh giá có điều kiện tâm trạng dùng nhãn tâm trạng của module này.

## Vai trò của module tâm trạng

Module phải cung cấp cho phần còn lại của hệ thống:

1. **Hồ sơ tâm trạng của mọi bài** (kể cả bài không có tag): xác suất thuộc 5 nhóm.
2. **Vị trí các nhóm trên mặt phẳng valence–energy**, học từ dữ liệu (để đặt vị trí thanh kéo và nút chọn nhanh).
3. **Hàm chấm điểm độ hợp tâm trạng** của một danh sách bài với tâm trạng mục tiêu.
4. **Nhãn tâm trạng của bài** để TV1 làm đánh giá có điều kiện tâm trạng.

## Góp ý của giảng viên mà module này cần trả lời

- Xử lý đầu vào tâm trạng phải "phức tạp hơn / đổi mới hơn", **không chỉ chọn bài theo nhãn có sẵn**.
  → Vì vậy: nhãn học từ cảm nhận người nghe (tag), bộ phân loại suy ra tâm trạng cho bài không có tag, vị trí nhóm học từ dữ liệu, cá nhân hóa mục tiêu tâm trạng theo từng người dùng.
- Đánh giá phải dùng **dữ liệu thật của người dùng** → TV1 dùng lịch sử nghe thật; module này cung cấp nhãn và hàm để đánh giá.

## 5 nhóm tâm trạng

| Mã | Tên | Mô tả |
|---|---|---|
| `vui` | Vui | tích cực, tươi sáng |
| `buon` | Buồn | tiêu cực, trầm |
| `thu_gian` | Thư giãn | nhẹ nhàng, êm |
| `nang_dong` | Năng động | sôi động, nhiều năng lượng |
| `cang_thang` | Căng thẳng | mạnh, gắt, tối |

Số nhóm có thể giảm nếu dữ liệu không đủ (xem `docs/mood_spec.md` mục M2).

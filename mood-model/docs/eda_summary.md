# Tóm tắt EDA liên quan tới module tâm trạng

Nguồn: báo cáo EDA của nhóm (notebook `analysis.ipynb`). Chỉ giữ các số liệu ảnh hưởng tới module này.

## Bảng bài hát (`Music_Info.csv`)

- 50.683 bài. 20.224 bài (39,9%) không có ai nghe → vẫn dùng được để **huấn luyện mô hình tâm trạng** (chỉ cần tag + audio).
- 11 đặc trưng âm thanh từ Spotify: danceability, energy, key, loudness, mode, speechiness, acousticness, instrumentalness, liveness, valence, tempo.

## Tag Last.fm (`tags`) — nguồn nhãn chính

- Thiếu ở 1.127 bài (2,2%).
- Chỉ **100 tag khác nhau**, trung bình 3,74 tag/bài, mọi tag xuất hiện ở hơn 5 bài.
- Tag gồm 4 loại: thể loại (rock, indie, metal, jazz…), thập niên (60s–00s), quốc gia (british, french…), **cảm xúc** (beautiful, mellow, chill, love…).
- **Rủi ro chính:** tag cảm xúc ít và thiên về nhóm nhẹ nhàng. Có thể không đủ bài cho nhóm `buon` và `cang_thang`. Phải đếm trước khi làm (task T1).

## `genre`

- Thiếu 55,9%, cùng nghệ sĩ bị gán nhiều genre (69,2% nghệ sĩ) → **không dùng làm nhãn**, chỉ dùng để mô tả hoặc làm đặc trưng phụ cho phân tích.

## Lỗi dữ liệu cần xử lý (nếu chưa có `tracks.parquet` của TV1)

| Lỗi | Số lượng | Xử lý |
|---|---|---|
| `tempo = 0` | 10 bài | gán NaN, điền trung vị |
| `time_signature = 0` | 10 bài | gán NaN |
| năm < 1950 | 13 bài | gán NaN |
| `spotify_id` trùng | 9 | đánh dấu `spotify_id_dup = True` |
| thời lượng quá ngắn/dài | 118 bài | đánh dấu, loại khỏi tập huấn luyện bộ phân loại |

## Nghi vấn ghép nhầm phiên bản (bản cover)

Vài bài phổ biến có tên nghệ sĩ là bản cover, ví dụ "Bring Me To Life – Katherine Jenkins", "Party In The U.S.A. – The Barden Bellas". Nếu audio features lấy từ bản cover mà tag thuộc bản gốc, cặp (audio → tag) bị nhiễu. Cần rà soát (task T1).

## Thiên lệch phổ biến

1% bài chiếm 34,8% lượt nghe. Khi chấm điểm tâm trạng **không được** cộng thêm độ phổ biến; việc đó thuộc phần kết hợp của TV2.

# Chạy mô hình gu cải tiến trên dữ liệu của Thy

Script này dùng **đúng hai file do Thy đã chuẩn bị**:

- `data/processed/train.csv`
- `data/processed/test.csv`

`test.csv` bị giữ kín trong lúc chọn mô hình. Script lấy 15% tương tác theo từng user từ `train.csv` làm validation; sau đó chọn cấu hình bằng `NDCG@10` trên validation, fit lại trên toàn bộ train và đánh giá đúng một lần trên test.

## 1. Chuẩn bị thư mục dữ liệu từ Drive

Tải thư mục data từ Drive về máy. Trong thư mục project, cấu trúc nên là:

```text
han_music_starter/
  data/
    processed/
      train.csv
      test.csv
    tracks.parquet              # hoặc Music Info.csv, nếu muốn rerank theo artist
```

Hai file train/test cần có ít nhất `user_id`, `track_id`, `playcount`. Theo báo cáo của Thy, chúng có thêm `engagement_score` và `engagement_level`.

Không mở rồi lưu lại CSV bằng Excel, vì Excel có thể đổi ID hoặc làm hỏng số 0 ở đầu ID.

## 2. Cài môi trường

Python 3.10 của bạn dùng được. Mở PowerShell tại thư mục `han_music_starter`:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Nếu `py -3.10` không có, thay bằng đường dẫn Python 3.10 của máy.

## 3. Smoke run trước khi chạy toàn bộ dữ liệu

Lệnh này chỉ lấy 2.000 user từ train, đánh giá 500 user. Với CSV 5,6 triệu dòng, chương trình sẽ đọc theo từng cụm và chỉ giữ các dòng của nhóm user này trong RAM. Nó kiểm tra schema, `implicit`, engagement weighting và output trước khi chạy toàn bộ.

```powershell
.\.venv\Scripts\python.exe -m src.run_real_taste `
  --train data\processed\train.csv `
  --test data\processed\test.csv `
  --max-train-users 2000 `
  --eval-users 500 `
  --factors 16 `
  --iterations 5 `
  --out runs\taste_smoke_02
```

Nếu output có `selected_config.json`, `metrics.csv` và không báo lỗi thì mới chạy toàn bộ.

## 4. Chạy mô hình gu đầy đủ

Lệnh này so sánh ba cách đặt confidence cho implicit ALS:

- `binary`: mọi lượt nghe có cùng trọng số.
- `log`: `log(1 + playcount)`; baseline implicit feedback phổ biến.
- `engagement`: `log(1 + playcount) × (0.5 + engagement_score)`; giảm trọng số interaction yếu và tăng interaction có bằng chứng gắn bó mạnh.

Engagement score **chỉ dùng từ train.csv**. Cột engagement trong test chỉ phục vụ báo cáo metrics theo từng nhóm, không đi vào mô hình.

```powershell
.\.venv\Scripts\python.exe -m src.run_real_taste `
  --train data\processed\train.csv `
  --test data\processed\test.csv `
  --eval-users 10000 `
  --factors 64 `
  --iterations 15 `
  --weightings binary,log,engagement `
  --out runs\taste_full_01
```

`--eval-users 10000` là một cohort đánh giá cố định theo seed 42. Không nói đây là đánh giá toàn bộ user. Khi máy đủ mạnh, dùng `--eval-users 0` để đánh giá tất cả user đủ điều kiện.

## 5. Tùy chọn: giới hạn bài cùng nghệ sĩ

Nếu có `tracks.parquet`/`Music Info.csv` có `track_id` và cột nghệ sĩ, thêm rerank đa dạng:

```powershell
.\.venv\Scripts\python.exe -m src.run_real_taste `
  --train data\processed\train.csv `
  --test data\processed\test.csv `
  --tracks data\tracks.parquet `
  --artist-col artist `
  --max-per-artist 2 `
  --eval-users 10000 `
  --factors 64 `
  --iterations 15 `
  --out runs\taste_artistcap_01
```

Nếu tên cột là `artist_name`, thay `--artist-col artist_name`. So sánh rerank với baseline trên **cùng test cohort** và cùng seed.

## 6. Đọc kết quả

Trong `runs/taste_full_01/`:

- `metrics.csv`: bảng chính để báo cáo; chọn dòng `stage=test`, model `als_selected`.
- `selected_config.json`: weighting được chọn bằng validation, trước khi mở test.
- `manifest.json`: đường dẫn, SHA256 của input, seed, số user/bài, cách tách validation và policy test.
- `test_als_selected_per_user.csv`: metric theo người dùng.
- `test_als_selected_recommendations.csv`: top-50 recommendation của cohort đánh giá.
- `als_model.npz`: ALS sau khi fit toàn bộ train, dùng cho fold-in/ứng dụng.

Các cột `favorite_NDCG@10`, `active_NDCG@10`, ... (nếu có) là breakdown theo nhãn engagement của **test**. Chúng chỉ dùng để quan sát mô hình có ưu tiên đúng các bài gắn bó mạnh hơn không; không được dùng để chọn model.

## 7. Cách trình bày đúng

1. Báo cáo cả Popularity và `als_selected`.
2. Nêu rõ weighting nào được chọn trên validation.
3. Nêu rõ test đã được Thy lọc theo 10-core, item test có trong train, nên **không** đo item cold-start.
4. Nếu chạy `--max-train-users` hoặc `--eval-users`, ghi rõ đây là tập con cố định seed 42.
5. Không dùng test để thử lần lượt nhiều tham số rồi chọn số cao nhất.

## 8. Các hướng cải tiến tiếp theo

| Hướng | Làm được ngay? | Cách đánh giá |
|---|---:|---|
| Engagement-weighted ALS | Có | So binary/log/engagement bằng validation, khóa cấu hình rồi test một lần. |
| Artist-cap rerank | Có nếu có file tracks | So NDCG/Recall và diversity/artist coverage cùng cohort. |
| Adaptive alpha gu–mood | Chờ code mood | Validation theo từng nhóm user; không dùng test để học alpha. |
| Seed-track fold-in cho user mới | Có | Giữ lại 3–5 bài seed từ train, đo recommendation với profile fold-in. |
| Long-tail-aware rerank | Có | Báo cáo Recall/NDCG riêng cho popular và long-tail item. |
| LLM/query tự nhiên | Sau baseline | Dùng cho input/explanation; không thay ALS baseline nếu chưa có query–user–item data. |

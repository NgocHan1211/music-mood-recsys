# CLAUDE.md — Module Tâm trạng (Mood Model)

File này là chỉ dẫn chính cho Claude Code. Đọc hết file này trước khi làm bất cứ việc gì.

## 1. Bối cảnh

- Đồ án môn học, nhóm 3 người: **Hệ thống gợi ý âm nhạc theo gu và tâm trạng**.
- Repo/folder này chỉ chứa **module tâm trạng**, do TV3 phụ trách (người đang làm việc với bạn).
- Module gu (ALS, fold-in) do TV2 làm; dữ liệu và đánh giá do TV1 làm. Module tâm trạng phải xuất ra file và hàm **đúng hợp đồng** trong `docs/data_contracts.md` để TV2 dùng được.

Đọc thêm, theo thứ tự:
1. `docs/project_context.md` — toàn bộ workflow nhóm và vị trí của module này.
2. `docs/eda_summary.md` — các số liệu EDA đã có, ảnh hưởng tới quyết định.
3. `docs/mood_spec.md` — đặc tả chi tiết các cải tiến cần làm.
4. `docs/data_contracts.md` — schema file đầu vào / đầu ra và chữ ký hàm.
5. `TASKS.md` — danh sách task theo thứ tự, kèm tiêu chí hoàn thành.
6. `configs/mood_config.yaml` — cấu hình nhóm tâm trạng, tag ứng viên, ngưỡng.

## 2. Cách làm việc

- Làm **lần lượt từng task trong `TASKS.md`**, không nhảy cóc. Mỗi task:
  1. Nêu kế hoạch ngắn (3–6 dòng) trước khi code.
  2. Viết code trong `src/mood/`, chạy thật trên dữ liệu.
  3. Ghi kết quả (số liệu, hình, quyết định) vào `reports/RESULTS.md` dưới tiêu đề của task.
  4. Đánh dấu `[x]` trong `TASKS.md`.
  5. Dừng lại, tóm tắt kết quả cho người dùng, chờ xác nhận rồi mới sang task tiếp.
- **Chạy trên bản mẫu trước** (`--sample`, ~5.000 bài) rồi mới chạy toàn bộ.
- Mọi quyết định thay đổi so với đặc tả (đổi số nhóm, gộp nhóm, đổi ngưỡng) ghi vào `docs/decisions.md` kèm lý do và số liệu.

## 3. Quy tắc bắt buộc

- **Không bịa số liệu.** Mọi con số trong báo cáo phải đến từ code đã chạy.
- **Không bịa tag.** Chỉ dùng tag thật sự có trong cột `tags` của dữ liệu (bộ này chỉ có ~100 tag khác nhau). Tag ứng viên trong config chỉ là gợi ý để đối chiếu.
- **Không dùng `tags` làm đặc trưng đầu vào cho bộ phân loại** (tag là nguồn sinh nhãn → rò rỉ). Đầu vào chỉ là đặc trưng âm thanh.
- **Chia fold theo nghệ sĩ** (GroupKFold theo `artist`) khi đánh giá bộ phân loại, tránh cùng nghệ sĩ nằm ở cả train và test.
- `random_state = 42` ở mọi chỗ có ngẫu nhiên.
- **Không đổi tên file, tên cột, chữ ký hàm** trong `docs/data_contracts.md`. Nếu thật sự cần đổi, hỏi người dùng trước.
- **Dừng và hỏi người dùng** khi: một nhóm tâm trạng có ít hơn `min_tracks_per_group` bài; tag có nghĩa mơ hồ (danh sách `ambiguous` trong config); kết quả bất thường (macro-F1 < 0.3, hoặc một nhóm gần như không bao giờ được dự đoán).
- Không commit dữ liệu (`data/` nằm trong `.gitignore`).

## 4. Môi trường và quy ước code

- Python 3.10+. Thư viện: xem `requirements.txt`. Nếu thiếu `lightgbm`, dùng `HistGradientBoostingClassifier` của scikit-learn thay thế.
- Cấu trúc:
  ```
  data/raw/Music_Info.csv          # người dùng tự đặt vào (không commit)
  data/processed/                  # file đầu ra theo data_contracts.md
  src/mood/                        # toàn bộ code module
  tests/                           # pytest cho các hàm chính
  reports/RESULTS.md               # kết quả từng task
  reports/figures/                 # hình (png) dùng cho báo cáo
  notebooks/                       # chỉ để trình bày, không chứa logic chính
  ```
- Mỗi bước chạy được bằng lệnh: `python -m src.mood.<ten_buoc> [--sample]`.
- Tên biến, hàm: tiếng Anh. Comment và docstring: tiếng Việt, ngắn gọn.
- Mã nhóm tâm trạng dùng thống nhất: `vui`, `buon`, `thu_gian`, `nang_dong`, `cang_thang`.
- Hàm công khai cho TV2 nằm trong `src/mood/api.py`, có docstring và test.

## 5. Đầu vào dữ liệu

- Ưu tiên `data/processed/tracks.parquet` (TV1 xuất, đã làm sạch). Nếu chưa có, đọc `data/raw/Music_Info.csv` và tự làm sạch tối thiểu theo `docs/eda_summary.md` mục "Lỗi dữ liệu".
- Cột `tags` là chuỗi phân tách bằng dấu phẩy, ví dụ `"rock, alternative, indie"`. Chuẩn hóa: chữ thường, bỏ khoảng trắng thừa.

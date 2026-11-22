# TASKS — Module tâm trạng

Làm lần lượt. Sau mỗi task: ghi kết quả vào `reports/RESULTS.md`, đánh dấu [x], dừng và báo người dùng.

## T0. Khởi tạo
- [x] Tạo môi trường, cài `requirements.txt`, kiểm tra có dữ liệu trong `data/raw/` hoặc `data/processed/`.
- [x] `src/mood/io.py`: hàm `load_tracks(sample: bool)` (đọc parquet nếu có, không thì CSV + làm sạch tối thiểu theo `docs/eda_summary.md`), `parse_tags()`.
- [x] Test: `tests/test_io.py` (tags được tách và chuẩn hóa đúng; không trùng `track_id`).

**Xong khi:** `python -m src.mood.io --sample` in ra số bài, số cột, 5 dòng đầu.

## T1. Kiểm kê tag và rà dữ liệu (M1)
- [x] `src/mood/tag_audit.py`: đếm tần suất 100 tag, gán category (tự động theo danh sách trong config, phần còn lại để `other`), xuất `tag_audit.csv`.
- [x] Liệt kê tag cảm xúc **thật có**, so với `candidate_tags`; in ra tag ứng viên không tồn tại.
- [x] Rà 50 bài phổ biến, xuất `suspect_tracks.csv` (cột `track_id, name, artist, reason`).
- [x] Hình: top 30 tag theo tần suất, tô màu theo category.

**Xong khi:** có bảng "tag cảm xúc → số bài". **DỪNG, hỏi người dùng** xác nhận ánh xạ tag → nhóm trước khi sang T2.

## T2. Nhãn yếu từ tag (M2)
- [x] `src/mood/labeling.py`: nhãn mềm, xử lý mâu thuẫn, nhãn cứng → `track_mood_labels.parquet`.
- [x] Thống kê: số bài mỗi nhóm, số bài `conflict`, `no_emotion_tag`.
- [x] Nếu nhóm < `min_tracks_per_group`: **DỪNG, hỏi** gộp nhóm hay dùng nhãn luật.
- [x] Hình: boxplot valence, energy theo nhóm; kiểm định Kruskal–Wallis.
- [x] Test: tổng nhãn mềm = 1; bài không có tag cảm xúc có NaN.

**Xong khi:** mỗi nhóm cuối cùng có ≥ `min_tracks_per_group` bài có nhãn.

## T3. Bộ phân loại (M3)
- [x] `src/mood/features.py`: dựng ma trận đặc trưng âm thanh (không có tags/genre/artist).
- [x] `src/mood/train_classifier.py`: 4 mô hình `baseline_ve`, `logreg_ve`, `logreg_full`, `gbm_full`; GroupKFold 5 fold theo artist.
- [x] Bảng so sánh: macro-F1, accuracy, log-loss, F1 từng nhóm; ma trận nhầm lẫn mô hình tốt nhất.
- [x] Hiệu chỉnh xác suất; reliability diagram, ECE trước/sau.
- [x] Permutation importance.
- [x] Huấn luyện mô hình cuối, lưu `models/mood_classifier.joblib`.

**Xong khi:** có bảng so sánh 4 mô hình. Nếu macro-F1 tốt nhất < 0.3: **DỪNG, báo người dùng**.

## T4. Hồ sơ tâm trạng mọi bài + không gian tâm trạng (M3, M4)
- [x] `src/mood/predict.py`: dự đoán cho mọi bài → `track_mood.parquet` (bài có nhãn tag: trộn nhãn tag và dự đoán, `mood_source = tag`).
- [x] `src/mood/mood_space.py`: tâm + hiệp phương sai mỗi nhóm → `mood_centroids.json`.
- [x] Hình: scatter valence–energy + ellipse mỗi nhóm + điểm tự đặt cũ.
- [x] Kiểm tra: mọi bài có đúng 1 dòng; tổng xác suất = 1.

**Xong khi:** hai file đầu ra đúng hợp đồng.

## T5. API cho TV2 (M5)
- [x] `src/mood/api.py`: đủ các hàm trong `docs/data_contracts.md` (trừ `user_mood_target`, `mood_journey` làm ở T6, T7 — tạm `raise NotImplementedError`).
- [x] `tests/test_api.py`: kiểu trả về, khoảng giá trị [0,1], tốc độ 20.000 bài < 1s, quick_mood trả về đúng tâm.
- [x] Ví dụ sử dụng trong `notebooks/demo_api.ipynb` hoặc `README.md`.

**Xong khi:** `pytest` qua hết.

## T6. Cá nhân hóa + đánh giá trên lịch sử nghe thật (M6)
- [x] Cài `user_mood_target` với shrinkage.
- [x] `src/mood/evaluate_mood.py`: dùng chia user của TV1 (nếu có) hoặc tự chia 20% user test (seed 42, user ≥ 10 bài, che 30%).
- [x] Với mỗi (user test, nhóm m có ≥ 3 bài bị che thuộc m): xếp hạng các ứng viên theo `mood_score`, đo Recall@20, NDCG@20 trên bài bị che thuộc m. So sánh 3 cách đặt mục tiêu: tọa độ tự đặt, tâm học được (M4), tâm cá nhân (M6).
- [x] Ứng viên: toàn bộ bài user chưa nghe trong phần thấy được (hoặc 1.000 bài ngẫu nhiên + bài bị che nếu quá chậm — ghi rõ).

**Xong khi:** có bảng so sánh 3 cách đặt mục tiêu.

## T7 (tùy chọn). Hành trình cảm xúc (M7)
- [x] Cài `mood_journey`, ví dụ minh họa buồn → vui với 8 bước, hình đường đi trên mặt phẳng valence–energy.

## T8. Tổng hợp báo cáo (M8)
- [x] `reports/RESULTS.md`: phần "Tổng kết" trả lời 4 câu hỏi của M8, kèm bảng và hình.
- [x] Liệt kê hạn chế: nhãn yếu, ít tag cảm xúc, nghi bản cover, nhóm nào là nhãn luật.

---

## Phần II — Ba cải tiến từ tài liệu tham khảo (chốt trong họp 2026-10-30)

Nguồn: HDBN (ACM TOIS 2025) và Read Between the Tracks (NLP4MusA 2026).

> **⚠️ Số liệu đã cập nhật 22/11/2026.** Phần II chạy trên bộ chia tự làm (che 30%) và kho
> 50.683 bài. Toàn bộ đã chạy lại ở Phần III trên **bộ chia chính thức của TV1** (catalogue
> 22.557 bài). Các dòng "Xong khi" dưới đây đã đổi sang **số chính thức**; số của Phần II để
> trong ngoặc chỉ để truy vết. **Báo cáo và slide chỉ dùng số chính thức.**

## A2. Khung đánh giá bốn tập — mô hình tâm trạng thêm gì so với mô hình gu
- [x] Nối sang `taste-model` của TV2 (`src/mood/taste.py`), huấn luyện ALS trên cohort 50.000 user, loại 145.227 tương tác bị che.
- [x] Bộ chia tái lập được theo `user_id` (`hidden_for_user`) để ALS và A2 dùng chung một bộ chia.
- [x] `src/mood/evaluate_four_sets.py`: bốn tập ứng viên, 7 bộ xếp hạng, NDCG@N từng tập.
- [x] Chạy ở N = 3 / 5 / 10, kiểm tra `random` ≈ 0,25 ở cả bốn tập.

**Xong khi:** có bảng NDCG@N bốn tập → **ghép 0,5 đạt 0,6106 ở set1 (N=3, 18.001 cặp), hơn chỉ-gu
+24,0%; mức phân biệt tâm trạng tăng từ +0,0217 lên +0,3594.** Bền theo cỡ tập: +24,0% (N=3) /
+23,4% (N=5) / +24,9% (N=10).
*(Phần II, bộ chia cũ: 0,6146 / +23,9% ở N=5; +23,0% / +23,9% / +25,7% cho N=3/5/10; phân biệt
+0,027 → +0,376. Kết luận không đổi.)*

## A1. Gán nhãn tag bằng embedding + chuẩn hóa z-score
- [x] `src/mood/labeling_embed.py`: nhúng tên tag và tên nhóm, z-score theo nhóm, giữ tag có max z ≥ 1.
- [x] So 4 cách gán nhãn, chấm bằng vị trí valence/energy thật của tag và bằng macro-F1 của bộ phân loại T3.
- [x] Thử 4 mô hình nhúng (mô hình của bài gốc bị khóa trên HuggingFace).

**Xong khi:** có bảng so sánh → **bảng tay thắng (84,9% khớp vị trí âm thanh / macro-F1 0,5656, so
với 52,3% / 0,5176 của embedding argmax và 73,8% / 0,5541 của bản ghép). Giữ bảng tay.**
*(Phần II, kho 50.683 bài: 83,0% / 0,5711 so với 50,0% / 0,5327 và 72,3% / 0,5562. Thứ hạng và
khoảng cách y nguyên.)*

## A3. "Nói ngầm" so với "nói tỏ"
- [x] Thêm mục tiêu `seed3` vào `evaluate_mood.py`, giữ nguyên luồng ngẫu nhiên để T6 tái lập đúng.
- [x] Quét 3 / 5 / 10 bài ví dụ, báo tỷ lệ phải lui về tâm chung.

**Xong khi:** có dòng thứ tư trong bảng T6 → **`seed3` (3 bài ví dụ) kém rõ: −9,0% Recall@20, đã
loại hẳn, không đưa vào `api.py`. `personal` vẫn tốt nhất nhưng chỉ +2,7%, không phải +12,6%.
Phát hiện "nói ngầm thắng nói tỏ" của bài gốc không chuyển sang hệ này.**
*(Phần II, bộ chia cũ và kho 50.683 bài: 3 bài +0,1%; 5 bài +10,4%; `personal` +12,6%. Phân rã
nguyên nhân thay đổi ở mục Phần III.)*

## Phần III — Chạy lại trên bộ dữ liệu chính thức của nhóm (2026-11-02)

Nguồn: bộ chia train/test của TV1 (`project/data/`, xem `data_preprocessing_report.md`).

- [x] `src/mood/splits.py`: đọc `train.csv` / `test.csv`, cache parquet, `user_views`, `als_frame`; kiểm tra bộ chia (0 cặp trùng, 0 cold-item).
- [x] Huấn luyện lại ALS trên toàn bộ tập train chính thức (289.572 user × 22.557 bài, 5.627.321 tương tác, rò rỉ 0).
- [x] Chạy lại **A2** ở N = 3 / 5 / 10 trên bộ chia chính thức.
- [x] Thử **`engagement_score`** (cột mới của TV1) làm trọng số tin cậy cho ALS.
- [x] Chạy lại **A1** trên catalogue chính thức 22.557 bài.
- [x] Chạy lại **A3** trên bộ chia chính thức, cộng hai phép đối chứng tách nguyên nhân (cỡ mẫu, kho nhạc).
- [x] 95 test qua ở thời điểm Phần III (thêm 6 test cho tầng đọc bộ chia). **Hiện tại: 119/119 test qua (`python -m pytest`, chạy lại 22/11/2026) sau khi thêm test cho L1. Báo cáo cuối dùng số 119.**

**Xong khi:** biết rõ kết luận nào đổi →
**A2 không đổi** (+24,0% / +23,4% / +24,9%); **`engagement_score` tốt hơn `playcount` (+30,4%)**;
**A1 không đổi** (bảng tay 84,9% / macro-F1 0,5656 vẫn thắng);
**A3 đổi**: `seed3` nay kém rõ (−9,0%, loại hẳn) và `personal` chỉ còn **+2,7%** thay vì +12,6%,
nguyên nhân chính là catalogue 10-core bỏ mất phần đuôi dài (−5,4 trên 9,9 điểm).

### Việc cần nhóm chốt
- **TV2**: đưa `engagement_score` vào grid-search mô hình gu; xác nhận cấu hình ALS cuối cùng.
- **TV1 / cả nhóm**: mọi số cá nhân hóa trong báo cáo cuối phải dùng **+2,7%**, không dùng +12,6%.

---

## Phần IV — Bảng G1: đánh giá trên toàn catalogue (2026-11-22)

Nguồn: mô hình gu thật của TV2 (`music-mood-recsys` commit `357b4c2` + `9ddd4b5`, push 22/11).

- [x] Clone repo nhóm, đối chiếu `taste-model/` đang có với bản TV2 vừa push.
- [x] Xác minh dữ liệu: `train.csv`/`test.csv` khớp **SHA256 từng bit** với manifest của TV2.
- [x] `src/mood/evaluate_catalogue.py`: nạp code TV2 từ hai nguồn, lớp `TasteMoodHybrid`, quét α trên validation.
- [x] Tái lập baseline của TV2 trên máy này — `popularity` sai số 0, `als_binary` sai số 1e-6.
- [x] Dựng dòng `ALS + mood`: α=0,9 chọn trên validation, test chạy đúng một lần.
- [x] Báo cáo thêm α=0,5 (điểm vận hành của A2), gắn cờ `reported_only`.
- [x] Ghi `reports/RESULTS.md` mục 21–26 và `docs/decisions.md` (6 dòng ngày 2026-11-22).

**Xong khi:** bảng G1 có đủ ba dòng →
**ALS thắng Popularity 4,77× Recall@10 / 5,36× NDCG@10 / 6,38× MAP@10**;
**ALS+mood ở α=0,9 ngang ALS thuần** (−0,92% NDCG@10, chênh < 1,3% ở cả 6 chỉ số);
**ở α=0,5 mất −54,5% NDCG@10 nhưng vẫn hơn Popularity 2,44×**.
Đường cong α dốc → **không có một α tốt nhất**, đây là bằng chứng số cho thiết kế thanh kéo α.

### Việc cần nhóm chốt sau Phần IV
- ~~**TV2**: push thêm `data.py` + `evaluate.py`; merge nhánh `han/taste-als-report`~~ →
  **✅ xong 22/11** (commit `26516b5` + PR #2, `main` nay ở `d359738`). Cần `git pull` về máy.
- ~~**TV3**: chạy lại **A2** bằng cấu hình gu chính thức~~ → **✅ xong 22/11, kết luận KHÔNG ĐỔI**:
  +24,3% (cohort đầy đủ, 18.001 cặp) và +25,2% (cohort 10k của G1). Báo cáo dùng **+24,3%**;
  giữ +24,0% (`log`/15) làm phép kiểm tra độ bền. Chi tiết: `reports/RESULTS.md` mục 27.
- **Cả nhóm**: bổ sung bootstrap CI theo user; chênh −0,92% chưa đủ để kết luận chắc "ngang nhau".

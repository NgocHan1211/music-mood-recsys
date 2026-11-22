# Đặc tả module tâm trạng và các cải tiến

## Phiên bản gốc (baseline cần cải tiến)

Bản ban đầu chỉ làm: người dùng chọn tâm trạng → điểm cố định (v*, e*) tự đặt tay → lọc bài theo khoảng cách Euclid trên (valence, energy). Hạn chế:

1. Tọa độ các nhóm tự đặt, không có căn cứ.
2. Chỉ dùng 2 đặc trưng; bỏ qua acousticness, mode, tempo… vốn cũng mang cảm xúc.
3. Không kiểm chứng valence/energy của Spotify có khớp cảm nhận người nghe không.
4. Mọi người dùng cùng một "vui" như nhau.

Bản baseline này **vẫn phải cài** (`baseline_ve`) để so sánh với các cải tiến.

---

## M1. Kiểm kê tag và rà dữ liệu

- Liệt kê toàn bộ tag, tần suất, phân loại thủ công thành: thể loại / thập niên / quốc gia / cảm xúc / khác.
- Đối chiếu tag cảm xúc thật có với `candidate_tags` trong config.
- Rà 50 bài phổ biến nhất (theo số người nghe nếu có `User_Listening_History.csv`, nếu không thì theo số tag) → đánh dấu bài nghi ghép nhầm bản cover vào `data/processed/suspect_tracks.csv`; loại khỏi tập huấn luyện.

## M2. Nhãn yếu từ tag (weak labeling) — cải tiến 1

Thay vì "bài có tag X thì thuộc nhóm Y", tạo **nhãn mềm**:

- Mỗi tag cảm xúc có trọng số cho từng nhóm (trong `mood_config.yaml`, ví dụ `mellow: {thu_gian: 1.0, buon: 0.3}`).
- Vector nhãn của bài = tổng trọng số các tag của bài, chuẩn hóa thành phân phối xác suất.
- Bài có tag thuộc các nhóm mâu thuẫn (ví dụ `happy` + `sad`) và không nhóm nào trội > `conflict_margin` → đánh dấu `conflict`, loại khỏi huấn luyện.
- Nhãn cứng `mood_label` = nhóm có xác suất cao nhất, chỉ khi xác suất ≥ `min_label_confidence`.
- **Nhóm thiếu tag** (< `min_tracks_per_group` bài): DỪNG, hỏi người dùng chọn một trong hai:
  - (a) gộp nhóm (ví dụ `cang_thang` vào `nang_dong`);
  - (b) định nghĩa nhóm bằng luật âm thanh (ví dụ valence < 0.35 và energy > 0.7) — ghi rõ trong báo cáo là "nhãn luật", không phải nhãn người nghe.

## M3. Bộ phân loại tâm trạng từ âm thanh — cải tiến 2

- **Mục đích:** gán hồ sơ tâm trạng cho mọi bài, kể cả bài không có tag cảm xúc.
- **Đầu vào:** đặc trưng âm thanh đã chuẩn hóa: valence, energy, danceability, acousticness, instrumentalness, speechiness, liveness, loudness, tempo, mode, key (one-hot hoặc sin/cos). **Không** dùng tags, genre, artist.
- **Mô hình so sánh:**
  1. `baseline_ve`: gán nhóm theo khoảng cách tới tọa độ tự đặt (chỉ valence, energy).
  2. `logreg_ve`: Logistic Regression chỉ dùng valence, energy.
  3. `logreg_full`: Logistic Regression toàn bộ đặc trưng.
  4. `gbm_full`: LightGBM (hoặc HistGradientBoosting) toàn bộ đặc trưng, `class_weight` cân bằng.
- **Đánh giá:** GroupKFold 5 fold theo `artist`. Chỉ số: macro-F1, accuracy, F1 từng nhóm, ma trận nhầm lẫn, log-loss.
- **Hiệu chỉnh xác suất:** dùng `CalibratedClassifierCV` (isotonic hoặc sigmoid) cho mô hình tốt nhất; báo cáo reliability diagram và ECE trước/sau.
- **Giải thích:** feature importance (permutation importance) → trả lời "ngoài valence/energy, đặc trưng nào mang cảm xúc".
- Mô hình cuối huấn luyện trên toàn bộ bài có nhãn, dự đoán cho mọi bài → `track_mood.parquet`.

## M4. Không gian tâm trạng học từ dữ liệu — cải tiến 3

- Với mỗi nhóm, tính **tâm** (trung bình có trọng số theo xác suất nhãn) và **hiệp phương sai** của (valence, energy) trên các bài có nhãn.
- Mỗi nhóm được biểu diễn là một phân phối Gaussian 2 chiều → `mood_centroids.json`.
- Dùng cho:
  - Nút chọn nhanh: chọn `vui` → thanh kéo nhảy tới tâm học được của `vui`.
  - Thanh kéo: điểm (v, e) bất kỳ → xác suất thuộc từng nhóm (Gaussian, chuẩn hóa) → hiển thị "đang nghiêng về Thư giãn 70%".
- Vẽ hình: scatter (valence, energy) mẫu bài tô màu theo nhóm + ellipse 1σ mỗi nhóm + điểm tự đặt cũ để so sánh.

## M5. Hàm chấm độ hợp tâm trạng — cải tiến 4

Với tâm trạng mục tiêu (từ thanh kéo hoặc nút chọn), mỗi bài s có:

- `p_target(s)` = Σ_m w_m · P(m | s), trong đó w_m là xác suất thuộc nhóm m của điểm thanh kéo (M4), P(m|s) từ bộ phân loại (M3).
- `prox(s)` = exp(−d²/2σ²), d là khoảng cách Mahalanobis (hoặc Euclid) từ (valence, energy) của bài tới điểm thanh kéo.
- `mood_score(s) = λ · p_target(s) + (1 − λ) · prox(s)`, mặc định λ = 0.6 (trong config).

Như vậy thanh kéo vẫn tinh chỉnh liên tục được, nhưng điểm không chỉ phụ thuộc 2 con số.

## M6. Cá nhân hóa mục tiêu tâm trạng — cải tiến 5

- Ý tưởng: "vui" của người nghe rock khác "vui" của người nghe acoustic.
- Với user và nhóm m: lấy các bài user đã nghe (có trọng số điểm chủ động nếu có, không thì 1) có P(m|s) cao; tính tâm cá nhân μ_u,m.
- Co về tâm chung theo số bài: μ' = (n · μ_u,m + k · μ_m) / (n + k), k = `shrinkage_k`.
- Hàm `user_mood_target(...)` trả về điểm mục tiêu đã cá nhân hóa; TV2 dùng thay cho tâm chung.
- Đánh giá: dùng `User_Listening_History.csv`, chia user như workflow chung; so sánh mood-conditioned Recall@20 khi dùng tâm chung vs tâm cá nhân (chỉ cần đánh giá phần tâm trạng: xếp hạng các bài bị che theo `mood_score`, không cần ALS).

## M7 (tùy chọn). Hành trình cảm xúc

- Người dùng chọn tâm trạng hiện tại A và mong muốn B → chia đoạn A→B thành N điểm → mỗi điểm chọn bài có `mood_score` cao nhất chưa được chọn.
- Chỉ làm nếu M1–M6 xong. Cung cấp hàm `mood_journey(...)` trả danh sách điểm mục tiêu.

## M8. Phân tích kiểm chứng (đưa vào báo cáo)

1. Valence/energy có tách được các nhóm không? (boxplot + kiểm định Kruskal–Wallis).
2. Thêm đặc trưng khác cải thiện bao nhiêu so với chỉ valence/energy? (so `logreg_ve` vs `logreg_full`).
3. Bộ phân loại so với baseline tọa độ tự đặt?
4. Cá nhân hóa (M6) có cải thiện đánh giá trên lịch sử nghe thật không?

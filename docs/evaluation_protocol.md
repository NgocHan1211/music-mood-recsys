# Giao thức đánh giá (Evaluation Protocol)

Tài liệu mô tả **các cách tạo tập test từ dữ liệu thực tế**, so sánh ưu nhược điểm, và đưa ra **khuyến nghị** cho dự án. Đối tượng: mô hình ALS gợi ý bài hát từ `playcount` (implicit feedback).

## 1. Điều kiện khởi đầu cần xác nhận

Cách chia dữ liệu phụ thuộc vào việc dữ liệu có **mốc thời gian** hay không.

- Bộ dữ liệu lịch sử nghe thường gặp trong dạng này (user, bài hát, playcount) **thường không có timestamp**. Nếu đúng như vậy thì **không thể chia theo thời gian** và cột `timestamp` trong `CONTRACTS.md` sẽ là null.
- Việc cần làm ngay: Thy kiểm tra `sample_interactions.parquet` có thông tin thời gian không, rồi nhóm chọn phương án ở mục 6.

Mọi phương án dưới đây đều theo các nguyên tắc chung:
1. **Chỉ tính trên tương tác dương** (playcount > 0); ô = 0 không phải "không thích".
2. **Chia theo từng user** để mọi user trong test đều có lịch sử trong train.
3. **Không để rò rỉ (leakage):** không dùng thông tin test khi lọc, chuẩn hóa hoặc huấn luyện.
4. **Cố định seed** và lưu lại danh sách chia để tái lập.

## 2. Tiền xử lý trước khi chia

| Bước | Lý do |
|---|---|
| Gộp tương tác trùng (user, bài) bằng cách cộng playcount | Mỗi cặp một dòng |
| Chỉ giữ bài có trong `tracks.parquet` | Đảm bảo có metadata/mood |
| Lọc user có ≥ 5 tương tác (`min_user_interactions`) | Cần đủ dữ liệu để vừa train vừa test |
| Lọc bài có ≥ 2 user nghe (tùy chọn) | Bài chỉ 1 người nghe không học được gì từ cộng tác |
| Lọc **trước** khi chia, dựa trên toàn bộ dữ liệu | Để user/bài không "biến mất" sau khi chia; chấp nhận được vì chỉ lọc theo số lượng, không theo nội dung test |

Ghi rõ ngưỡng lọc và số user/bài/tương tác còn lại sau lọc trong notebook.

## 3. Các cách tạo tập test

### 3.1 Giữ ngẫu nhiên theo từng user (Random holdout per user)
**Cách làm:** với mỗi user, lấy ngẫu nhiên 20% tương tác làm test (tối thiểu 1), phần còn lại là train.
- ✅ Đơn giản, dùng được khi không có timestamp, mỗi user có nhiều mục test nên metric ổn định.
- ❌ Không phản ánh thứ tự thời gian thực; có thể "dự đoán quá khứ bằng tương lai".
- Dùng khi: không có timestamp (**phương án chính nếu thiếu thời gian**).

### 3.2 Leave-one-out (LOO)
**Cách làm:** mỗi user giữ đúng 1 tương tác làm test (ngẫu nhiên, hoặc gần nhất nếu có thời gian), phần còn lại train.
- ✅ Train tối đa dữ liệu, phổ biến trong bài báo nên dễ so sánh.
- ❌ Chỉ 1 mục test mỗi user, metric nhiễu (HitRate/NDCG phụ thuộc vào đúng 1 bài); bản ngẫu nhiên không đại diện cho ngữ cảnh thực.
- Dùng khi: dữ liệu nhỏ, hoặc để so sánh với tài liệu khác.

### 3.3 Chia theo thời gian toàn cục (Global temporal split)
**Cách làm:** chọn mốc thời gian T; mọi tương tác trước T là train, sau T là test (có thể thêm tập validation giữa hai mốc).
- ✅ **Sát thực tế nhất** (hệ thống luôn dự đoán tương lai từ quá khứ), tránh rò rỉ thời gian.
- ❌ Cần timestamp; có user/bài mới chỉ xuất hiện sau T (cold-start) làm giảm số user đánh giá được.
- Dùng khi: **có timestamp** (phương án chính nếu có).

### 3.4 Chia theo thời gian từng user (Per-user temporal split)
**Cách làm:** với mỗi user, sắp xếp theo thời gian, lấy 20% tương tác gần nhất (hoặc 1 bài cuối) làm test.
- ✅ Mọi user đều có test; đơn giản; vẫn tôn trọng thứ tự trong từng user.
- ❌ Vẫn có thể rò rỉ gián tiếp giữa user (xu hướng toàn cục đã "biết" tương lai của user khác).
- Dùng khi: có timestamp nhưng cần giữ lại nhiều user.

### 3.5 Chia theo user (User-based / cold-start user)
**Cách làm:** chọn ngẫu nhiên một nhóm user làm test; toàn bộ tương tác của họ không có trong train. Khi đánh giá, cho mô hình xem một phần tương tác của user đó (ví dụ 80% để tạo vector, 20% để test) hoặc dùng mô hình hybrid.
- ✅ Đo khả năng xử lý user mới.
- ❌ ALS thuần không có vector cho user ngoài train, phải dùng "fold-in" (`recalculate_user=True` trong `implicit`) hoặc mood/nội dung.
- Dùng khi: đánh giá kịch bản cold-start (**đánh giá bổ sung**).

### 3.6 Chia theo bài hát (Item cold-start)
**Cách làm:** giữ riêng một nhóm bài hát, loại khỏi train; test xem mô hình có gợi ý được bài mới không.
- ✅ Phù hợp mục tiêu "gợi ý theo mood" vì bài mới có thể gán mood từ tag.
- ❌ ALS thuần không làm được; cần phương pháp dựa nội dung.
- Dùng khi: đánh giá phần hybrid/mood (**đánh giá bổ sung**).

### 3.7 K-fold cross-validation
**Cách làm:** chia mỗi user thành K phần, lần lượt dùng 1 phần làm test.
- ✅ Ước lượng ổn định, có độ lệch chuẩn.
- ❌ Tốn K lần huấn luyện, thường không cần cho dữ liệu lớn.
- Dùng khi: cần báo cáo khoảng tin cậy.

### Bảng so sánh

| Phương án | Cần timestamp | Thực tế | Độ ổn định metric | Ghi chú |
|---|---|---|---|---|
| Random holdout per user | Không | Trung bình | Cao | Phương án chính nếu thiếu thời gian |
| Leave-one-out | Không (tốt hơn nếu có) | Thấp–TB | Thấp | Dễ so sánh với bài báo |
| Global temporal | **Có** | **Cao nhất** | Trung bình | Phương án chính nếu có thời gian |
| Per-user temporal | **Có** | Cao | Cao | Giữ được nhiều user |
| Cold-start user | Không | Cao (kịch bản mới) | Trung bình | Bổ sung |
| Cold-start item | Không | Cao (kịch bản mới) | Trung bình | Bổ sung, cần hybrid |
| K-fold | Không | Trung bình | Rất cao | Tốn thời gian |

## 4. Tập validation và test

Tránh chọn siêu tham số trên tập test.
- Từ **train** của mục 3, tách tiếp một tập **validation** theo **cùng cách chia** (ví dụ lấy thêm 10% tương tác mỗi user).
- Chọn `factors`, `regularization`, `alpha`, `iterations` trên validation.
- Sau khi chốt, **huấn luyện lại trên train + validation** và đánh giá **một lần** trên test.

## 5. Cách xếp hạng và lấy mẫu negative

- **Full ranking (khuyến nghị):** chấm điểm **toàn bộ bài hát**, bỏ các bài đã có trong train của user, lấy top-K, so với test. Không thiên lệch, tốn tính toán nhưng ALS nhanh.
- **Sampled negatives (ví dụ 99 hoặc 100 bài ngẫu nhiên):** nhanh hơn, nhưng nhiều nghiên cứu (Krichene & Rendle, 2020) chỉ ra metric lấy mẫu **không nhất quán** với xếp hạng đầy đủ, dễ sai khi so sánh mô hình. Chỉ dùng nếu bắt buộc, và nêu rõ.
- **Không** coi bài chưa nghe là "negative thật"; trong full ranking, bài chưa nghe chỉ là ứng viên chưa biết.
- Khi huấn luyện ALS, negative đã được xử lý ngầm qua confidence `c = 1` cho các ô 0, không cần lấy mẫu.

## 6. Metric và K

| Metric | Ý nghĩa | Ghi chú |
|---|---|---|
| **Recall@K** | Tỷ lệ bài test nằm trong top-K | Metric chính |
| **NDCG@K** | Tính thêm vị trí xếp hạng | Metric chính |
| Precision@K | Tỷ lệ gợi ý trúng | Bị giới hạn khi ít mục test mỗi user |
| MAP@K | Trung bình độ chính xác theo vị trí | Bổ sung |
| HitRate@K | Có ít nhất 1 trúng không | Dùng cho leave-one-out |
| Coverage@K | Tỷ lệ bài hát được gợi ý ít nhất 1 lần | Đo đa dạng |
| Popularity bias | Độ phổ biến trung bình của bài gợi ý | Phát hiện thiên lệch |

- **K đề xuất:** 10 và 20 (báo cáo cả hai); K = 50 nếu muốn xem xa hơn.
- Tính metric **theo từng user rồi lấy trung bình** (macro-average); ghi rõ số user đánh giá.
- Nên có **baseline**: (1) gợi ý bài phổ biến nhất (popularity), (2) ngẫu nhiên, để biết ALS có thực sự tốt hơn.
- Nếu test cho playcount: có thể dùng playcount làm **trọng số relevance** trong NDCG, nhưng phải ghi rõ để khỏi nhầm với NDCG nhị phân.

## 7. Các lỗi rò rỉ dữ liệu cần tránh

1. Dùng test để chọn siêu tham số hoặc ngưỡng lọc.
2. Chia ngẫu nhiên khi có timestamp rồi báo cáo như dự đoán tương lai.
3. Không loại bài đã nghe trong train khỏi danh sách gợi ý (làm kết quả cao giả).
4. Tính thống kê (độ phổ biến, chuẩn hóa) trên toàn bộ dữ liệu rồi mới chia.
5. Nhãn mood dựa trên tag Last.fm được tạo từ chính hành vi nghe của user trong test (kiểm tra nguồn tag).
6. Còn user/bài chỉ có trong test: loại khỏi đánh giá chính và báo cáo riêng.

## 8. Khuyến nghị cho dự án

| Ưu tiên | Khuyến nghị |
|---|---|
| **Chính (nếu có timestamp)** | Global temporal split (3.3), validation lấy từ cuối phần train |
| **Chính (nếu KHÔNG có timestamp)** | Random holdout per user, 20% test, `min_user_interactions = 5`, seed 42 (3.1) |
| **Phụ** | Leave-one-out (3.2) để so sánh với tài liệu tham khảo |
| **Bổ sung** | Cold-start item (3.6) để đánh giá phần mood/hybrid |
| **Xếp hạng** | Full ranking, loại bài đã nghe trong train |
| **Metric** | Recall@10/20, NDCG@10/20 (chính); Precision, MAP, Coverage (phụ) |
| **Baseline** | Popularity và Random |

**Lý do:** random holdout per user là lựa chọn an toàn nhất khi thiếu thời gian vì mỗi user có nhiều mục test nên số liệu ổn định; temporal split là chuẩn vàng nếu có dữ liệu. Full ranking tránh thiên lệch của negative sampling.

## 9. Triển khai trong repo
- `src/split.py::split_interactions(strategy=...)` cài đặt 3 chiến lược: `random_holdout`, `leave_one_out`, `temporal`.
- `src/evaluate.py::evaluate(...)` tính Recall/NDCG/Precision/MAP/HitRate@K.
- Lưu chỉ mục chia ra `data/processed/split_*.parquet` để tái lập, không commit nếu lớn.

## Tài liệu tham khảo
- Hu, Koren, Volinsky (2008). *Collaborative Filtering for Implicit Feedback Datasets*.
- Krichene & Rendle (2020). *On Sampled Metrics for Item Recommendation*. KDD.
- Meng et al. (2020). *Exploring Data Splitting Strategies for the Evaluation of Recommendation Models*. RecSys.
- Sun et al. (2020). *Are We Evaluating Rigorously? Benchmarking Recommendation for Reproducible Evaluation and Fair Comparison*. RecSys.

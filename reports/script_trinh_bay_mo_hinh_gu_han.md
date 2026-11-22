# Script trình bày — Phần mô hình gu của Hân

> Thời lượng gợi ý: 6–8 phút. Nói tự nhiên, không cần đọc y nguyên từng chữ.

## 1. Mở đầu — phần này giải quyết gì?

“Phần của mình là mô hình **gu cá nhân hóa**. Câu hỏi mình muốn trả lời khá đơn giản: nếu chỉ biết lịch sử nghe nhạc của một người, liệu mình có gợi ý đúng gu của họ tốt hơn việc đưa những bài đang phổ biến cho tất cả mọi người hay không?

Đây là nền cho hệ cuối cùng của nhóm. Sau này, điểm gu này sẽ được ghép với điểm mood của bạn Như. Nhưng trước khi ghép mood, tụi mình cần chứng minh riêng rằng phần gu đã có ích trên dữ liệu nghe thật.”

## 2. Dữ liệu đã dùng

“Dữ liệu đầu vào là `train.csv` và `test.csv` do bạn Thy tiền xử lý từ lịch sử nghe thực tế.

Trước khi lấy mẫu chạy model, tập sau lọc 10-core có khoảng **7,03 triệu tương tác**, **289.572 người dùng** và **22.557 bài hát**. Điều kiện 10-core nghĩa là user và bài hát đều phải xuất hiện đủ nhiều để mô hình có tín hiệu học.

Ở thí nghiệm này mình lấy mẫu cố định **10.000 user** từ train và đánh giá **2.000 user**. Seed là 42, nên cùng code và cùng file input thì chạy lại sẽ ra đúng cohort user như cũ. Mình lấy mẫu vì chạy toàn bộ 289 nghìn user trên máy cá nhân sẽ nặng, còn 10 nghìn user là đủ để làm thí nghiệm baseline có kiểm chứng.”

Nếu bị hỏi “vậy có phải test toàn bộ data không?” thì trả lời:

“Chưa phải toàn bộ user gốc. Đây là đánh giá trên một tập con cố định 10 nghìn user train và 2 nghìn user test. Vì vậy trong báo cáo mình gọi đúng là *10k-user experiment*, không khẳng định đây là kết quả cuối cùng trên toàn bộ 289 nghìn user.”

## 3. Chống leakage: train, validation, test được dùng thế nào?

“Điểm mình muốn nhấn mạnh nhất là **test không được dùng để chọn model**.

Mình lấy `train.csv`, rồi ẩn thêm 15% lịch sử của từng user để tạo validation. Trên validation, mình so sánh các biến thể model và chọn theo NDCG@10. Sau khi chọn xong, mình train lại trên toàn bộ train sample và chỉ đánh giá một lần trên `test.csv`.

Nên số test cuối mới là số mình dùng để kết luận; còn validation chỉ dùng để ra quyết định chọn cấu hình.”

Nếu bị hỏi “engagement ở test có chui vào model không?” thì trả lời:

“Không. `engagement_score` chỉ dùng từ **train** khi thử biến thể engagement. Nhãn engagement của test chỉ dùng để báo cáo theo nhóm sau khi xếp hạng, không được đưa vào feature hay confidence của model.”

## 4. Các baseline và mô hình thử

“Mình có một baseline là **Popularity**. Nó đề xuất những bài được nhiều user nghe nhất; nó không biết em là ai, nên mọi user gần như nhận danh sách giống nhau. Nếu ALS không hơn baseline này thì phần cá nhân hóa chưa có ý nghĩa.

Sau đó mình dùng **ALS cho implicit feedback**. Dữ liệu nghe nhạc không phải rating 1–5 sao, nên mình chỉ biết user có tương tác và mức độ tương tác. ALS học một vector ẩn cho user và một vector ẩn cho bài; user có lịch sử nghe gần nhau sẽ có vector gần nhau, từ đó hệ xếp hạng bài chưa nghe.

Mình thử ba cách tạo confidence:

1. **ALS binary**: user đã nghe bài thì coi là một tín hiệu dương như nhau.
2. **ALS log-playcount**: nghe nhiều hơn thì confidence lớn hơn, nhưng dùng log để không để một bài nghe cực nhiều lấn át hết các bài khác.
3. **ALS engagement**: kết hợp log-playcount với `engagement_score` do Thy tạo. Công thức là `log(1 + playcount) × (0,5 + engagement_score)`. Ý tưởng là bài nghe chủ động/gắn bó sẽ có trọng số cao hơn bài nghe lướt.”

## 5. Kết quả validation và quyết định chọn model

“Trên validation, Popularity có NDCG@10 là **0,0199**. Ba ALS đều cao hơn rất nhiều:

- ALS binary: **0,0862**
- ALS log: **0,0835**
- ALS engagement: **0,0847**

Theo rule đã định trước, mình chọn NDCG@10 validation cao nhất. Vì vậy cấu hình được chọn là **ALS binary**, với 32 latent factors, 8 iterations, regularization 0,1 và alpha 10.

Một điểm quan trọng là binary chỉ hơn engagement một chút, chứ không phải hơn rất xa. Nên engagement vẫn đáng giữ làm ablation và phân tích thêm, nhưng mình không tự chọn engagement chỉ vì nó nghe có vẻ ‘thông minh’ hơn.”

## 6. Kết quả test: mô hình gu có thực sự hiệu quả không?

“Sau khi chọn binary bằng validation, mình đánh giá trên test độc lập.

Popularity có Recall@10 là **0,0307**, NDCG@10 là **0,0243** và MAP@10 là **0,0124**.

ALS binary được chọn có Recall@10 là **0,1464**, NDCG@10 là **0,1302** và MAP@10 là **0,0788**.

Nói theo tỷ lệ dễ hình dung:

- Recall@10 tăng khoảng **4,77 lần**.
- NDCG@10 tăng khoảng **5,36 lần**.
- MAP@10 tăng khoảng **6,38 lần**.

Nên kết luận ở đây khá rõ: học gu từ lịch sử nghe cá nhân tốt hơn rất nhiều so với chỉ đưa bài phổ biến. Đây là bằng chứng baseline mà nhóm cần trước khi ghép mood.”

Nếu bị hỏi “NDCG là gì?” thì trả lời ngắn:

“Recall đo xem mình tìm lại được bao nhiêu bài user thật sự nghe; NDCG còn thưởng thêm khi bài đúng nằm ở vị trí đầu danh sách. MAP đo chất lượng thứ hạng của các bài đúng. Nên cả ba cùng tăng là tín hiệu tốt, không phải chỉ một metric may mắn tăng.”

## 7. Phân tích engagement — một kết quả không đơn giản nhưng có giá trị

“Kết quả tổng thể nói rằng ALS binary thắng, nên mình không thể nói engagement luôn tốt hơn. Nhưng khi nhìn riêng nhóm bài `favorite` trên validation, engagement có NDCG@10 **0,0811**, còn binary là **0,0581**.

Diễn giải hợp lý là: engagement có vẻ hữu ích khi bài toán muốn giữ đúng những bài user thật sự gắn bó/nghe kỹ. Còn nếu mục tiêu là dự đoán mọi kiểu hành vi, bao gồm nghe lướt và khám phá, thì binary tổng quát hơn.

Đây không phải thất bại của engagement. Nó tạo ra một hướng cải tiến có câu hỏi nghiên cứu rõ ràng: liệu ta có nên thay đổi confidence hoặc trọng số theo kiểu hành vi nghe của từng user hay không?”

## 8. Giới hạn cần nói thẳng

“Phần này mới là mô hình gu, chưa phải hệ gu + mood hoàn chỉnh. Thứ hai, đây là thí nghiệm 10 nghìn user, chưa phải chạy full toàn bộ tập. Thứ ba, metric offline đo khả năng dự đoán lịch sử nghe bị ẩn, chưa chứng minh hoàn toàn trải nghiệm người dùng ngoài đời.

Mình chủ động nêu giới hạn để không overclaim. Tuy nhiên, với scope baseline của tuần này, model đã có train/test thật, protocol rõ ràng, output tái lập và chênh lệch rõ với baseline.”

## 9. Phương hướng ghép với mood

“Bước tiếp theo là ghép điểm gu này với nhánh mood của Như. Công thức dự kiến là:

`score cuối = α × score_gu + (1 − α) × score_mood`.

Người dùng có thể tự kéo α để ưu tiên gu hay mood. Ngoài metric gộp, nhóm sẽ đánh giá tách hai thứ: bài đó có đúng gu không, và có đúng mood mục tiêu không.

Ngoài ra, phát hiện engagement gợi ý một hướng sáng tạo hơn: với user nghe ổn định hoặc nhiều favorite, có thể tăng trọng số engagement; với user thích khám phá thì giữ ALS binary. Đây là hướng sau baseline, chưa đưa vào kết luận hiện tại.”

## 10. Câu chốt

“Tóm lại, phần mô hình gu đã hoàn thành mục tiêu cơ bản: ALS cá nhân hóa vượt Popularity rõ rệt trên dữ liệu nghe thực tế. Mô hình được chọn là ALS binary; engagement chưa thắng toàn cục nhưng cho tín hiệu tốt ở nhóm favorite và mở ra hướng adaptive model. Kết quả này sẽ là nhánh `score_gu` để nhóm ghép với mood ở bước tiếp theo.”

---

## File nên mở khi trình bày

1. `reports/bao_cao_mo_hinh_gu_han.html` — trang báo cáo/biểu đồ chính.
2. `runs/taste_10k_01/metrics.csv` — số liệu gốc để đối chiếu nếu bị hỏi.
3. `runs/taste_10k_01/selected_config.json` — bằng chứng cấu hình selected là ALS binary.
4. `runs/taste_10k_01/manifest.json` — seed, nguồn data và quy tắc đánh giá tái lập.

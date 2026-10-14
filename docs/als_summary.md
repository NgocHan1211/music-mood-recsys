# Tóm tắt ALS cho phản hồi ngầm (Implicit Feedback)

*Nguồn chính: Hu, Koren & Volinsky (2008), "Collaborative Filtering for Implicit Feedback Datasets", ICDM; tài liệu thư viện `implicit`.*

## 1. Bài toán và vì sao cần ALS

Trong dự án này, dữ liệu tương tác là **số lần nghe (playcount)** của người dùng với bài hát. Đây là **phản hồi ngầm (implicit feedback)**, khác với đánh giá sao (explicit feedback):

| | Explicit (sao 1–5) | Implicit (playcount, click) |
|---|---|---|
| Có tín hiệu "không thích" | Có (sao thấp) | **Không**: chưa nghe ≠ không thích |
| Giá trị lớn nghĩa là | Thích nhiều | **Tin cậy hơn** rằng người dùng thích |
| Dữ liệu nhiễu | Ít | Nhiều (nghe nhầm, nhạc nền, chia sẻ tài khoản) |
| Ma trận | Chỉ có các ô đã chấm | **Toàn bộ ô đều có ý nghĩa** (kể cả ô = 0) |

Hệ quả: các phương pháp phân rã ma trận cho explicit (chỉ tối ưu trên các ô đã quan sát) không phù hợp. Hu et al. đề xuất mô hình xử lý **cả ô = 0** nhưng với **độ tin cậy thấp hơn**, và dùng **ALS** để huấn luyện hiệu quả.

## 2. Mô hình

Gọi `r_ui` là playcount của user `u` với bài `i`.

**Preference (sở thích nhị phân):**
```
p_ui = 1  nếu r_ui > 0
p_ui = 0  nếu r_ui = 0
```

**Confidence (độ tin cậy):**
```
c_ui = 1 + α · r_ui
```
- Ô chưa nghe (r = 0) có `c = 1`: có tin cậy nhưng thấp.
- Nghe càng nhiều, `c` càng lớn, ta càng tin user thích bài đó.
- `α` điều chỉnh mức tăng. Với playcount lệch nặng (long-tail), thường dùng biến đổi log: `c_ui = 1 + α · log(1 + r_ui / ε)`.

**Phân rã ma trận:** mỗi user `u` có vector ẩn `x_u ∈ R^f`, mỗi bài `i` có vector ẩn `y_i ∈ R^f`. Dự đoán sở thích: `p̂_ui = x_uᵀ y_i`.

**Hàm mất mát:**
```
min  Σ_{u,i} c_ui (p_ui − x_uᵀ y_i)²  +  λ ( Σ_u ‖x_u‖² + Σ_i ‖y_i‖² )
```
- Tổng chạy trên **tất cả** cặp (u, i), không chỉ cặp đã quan sát.
- Sai số ở ô có `c` lớn bị phạt nặng hơn, nên mô hình ưu tiên khớp với bài nghe nhiều.
- `λ` là hệ số điều chuẩn, chống overfit.

## 3. Thuật toán ALS

Hàm mất mát không lồi theo đồng thời `X` và `Y`, nhưng **lồi (bậc hai) nếu cố định một bên**. ALS luân phiên:

1. **Cố định Y, giải X.** Với mỗi user:
   ```
   x_u = (Yᵀ Cᵘ Y + λI)⁻¹ Yᵀ Cᵘ p(u)
   ```
   trong đó `Cᵘ` là ma trận đường chéo chứa `c_ui` của user `u`, `p(u)` là vector sở thích của `u`.
2. **Cố định X, giải Y.** Với mỗi bài:
   ```
   y_i = (Xᵀ Cⁱ X + λI)⁻¹ Xᵀ Cⁱ p(i)
   ```
3. Lặp lại cho đến khi hội tụ hoặc đủ số vòng.

**Mẹo tăng tốc quan trọng.** Viết `Yᵀ Cᵘ Y = YᵀY + Yᵀ (Cᵘ − I) Y`. Phần `YᵀY` tính **một lần** cho mọi user; phần `(Cᵘ − I)` chỉ khác 0 ở các bài user đã nghe (rất thưa). Nhờ vậy độ phức tạp mỗi vòng lặp chỉ khoảng `O(f²·N + f³·(m+n))`, tuyến tính theo số tương tác `N` (m user, n bài), đủ nhanh cho dữ liệu lớn. Mỗi user/bài giải độc lập nên **song song hóa dễ dàng**.

**Gợi ý:** với user `u`, tính điểm `x_uᵀ y_i` cho mọi bài, **loại các bài user đã nghe trong tập train**, lấy top-K.

## 4. Siêu tham số

| Tham số | Ý nghĩa | Khoảng thử gợi ý |
|---|---|---|
| `factors` (f) | Số chiều vector ẩn | 32, 64, 128, 256 |
| `regularization` (λ) | Điều chuẩn L2 | 0.001 – 1.0 (thang log) |
| `alpha` (α) | Hệ số tăng confidence | 1, 10, 40, 100 |
| `iterations` | Số vòng ALS | 15 – 50 |

Lưu ý: `alpha` và `regularization` **tương tác với nhau** và phụ thuộc cách chuẩn hóa playcount (log hay thô), nên cần tìm kiếm cùng lúc trên tập validation, không chỉnh riêng lẻ.

## 5. Dùng thư viện `implicit`

```python
import implicit
from scipy.sparse import csr_matrix

# user_items: ma trận CSR (n_users x n_items), giá trị = playcount (đã biến đổi)
model = implicit.als.AlternatingLeastSquares(
    factors=64, regularization=0.01, alpha=40.0, iterations=20, random_state=42
)
model.fit(user_items)

ids, scores = model.recommend(
    userid=0, user_items=user_items[0], N=10, filter_already_liked_items=True
)
```
Điểm cần kiểm tra theo **phiên bản** bạn cài (đọc changelog):
- Từ các bản gần đây, `fit()` nhận ma trận **user × item**; các bản cũ nhận **item × user**. Sai chiều sẽ cho kết quả vô nghĩa mà không báo lỗi.
- `implicit` dùng conjugate gradient để xấp xỉ bước giải, nhanh hơn giải đóng; có phiên bản GPU.
- Nếu dùng nhiều luồng BLAS, nên đặt `OPENBLAS_NUM_THREADS=1` để tránh chậm.

## 6. Ưu và nhược điểm

**Ưu điểm**
- Xử lý đúng bản chất implicit feedback; mở rộng tốt với dữ liệu thưa, lớn.
- Ít siêu tham số, baseline mạnh và dễ tái lập.
- Vector ẩn của bài hát có thể dùng cho tìm bài tương tự.

**Nhược điểm**
- **Cold-start:** user/bài hát mới chưa có tương tác thì không có vector.
- Không dùng trực tiếp nội dung (tag, đặc trưng âm thanh), nên cần kết hợp với nhãn mood (hybrid) cho bài toán gợi ý theo tâm trạng.
- Độ nhạy cao với cách biến đổi playcount và `alpha`.
- Thiên lệch phổ biến (popularity bias): dễ gợi ý bài nổi tiếng.

## 7. Liên hệ với dự án

- Ma trận user–bài từ `sample_interactions.parquet` (cột `playcount`) là đầu vào của ALS.
- Nhãn mood (`track_mood_labels.parquet`) có thể dùng để **lọc/xếp lại** kết quả ALS theo mood mong muốn (bước hậu xử lý), hoặc làm đặc trưng cho cold-start.
- Cần chốt cách biến đổi playcount (thô, log, cắt ngưỡng) và ghi vào `CONTRACTS.md` để mọi người dùng chung.

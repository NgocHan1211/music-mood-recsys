# Hướng dẫn gán nhãn tâm trạng — bộ nhãn do người nghe

**Đọc hết trang này trước khi mở file gán nhãn.** Mất khoảng 8 phút.

---

## 1. Việc này để làm gì

Toàn bộ nhãn tâm trạng của đồ án hiện là **nhãn yếu**: suy từ tag Last.fm qua một bảng trọng số
nhóm tự đặt. Không ai trong dữ liệu từng nói "bài này vui".

Hệ quả: khi nhóm so T2 (bảng tay) với embedding (A1) hay LLM (L1), cả ba đều được **chấm bằng
chính hệ nhãn tag đó**. Đó là lập luận vòng tròn — thứ hạng có thể chỉ phản ánh "cách nào giống
bảng tag nhất", không phải "cách nào đúng nhất".

**180 bài các bạn gán nhãn sẽ là mốc độc lập đầu tiên của đồ án.** Nó cho phép trả lời:

- Nhãn từ tag có khớp cảm nhận người nghe không, và khớp bao nhiêu phần trăm?
- Ở nhóm `conflict` (tag mâu thuẫn) và `no_emotion_tag` (không có tag cảm xúc), mô hình đoán có
  đúng không?
- Giữa ba cách gán nhãn (bảng tay / embedding / LLM), cách nào gần người thật nhất?

Không có bộ này, mục "Hạn chế" của báo cáo phải viết *"chưa khảo sát người thật"*. Có rồi thì
nhóm có một con số.

---

## 2. Ba quy tắc không được phá

**1. Nghe trước đã.** Phát ít nhất 20 giây rồi mới quyết. Đừng đọc tên bài rồi đoán.

**2. Không tra cứu.** Không Google, không xem tag Last.fm, không mở Spotify đọc genre, không hỏi
ChatGPT. Nếu biết sẵn bài đó thì cứ dùng cảm nhận của mình, nhưng **đừng đi tìm thêm thông tin**.

**3. Không bàn với nhau khi chưa xong.** Mỗi người gán độc lập hết 180 bài. Chỉ họp sau khi cả ba
đã nộp file. Nếu bàn trước thì mức đồng thuận đo được sẽ là số giả.

> **Tuyệt đối không mở `annotation/KEY_KHONG_MO_TRUOC_KHI_GAN.csv`.** File đó chứa đáp án của máy
> (tag, nhãn T2, dự đoán mô hình). Mở trước là bộ nhãn mất giá trị, và không có cách nào sửa ngoài
> việc lấy mẫu lại từ đầu.

---

## 3. Năm nhóm — định nghĩa

Phán đoán theo **hai trục**: nhạc nghe **tích cực hay tiêu cực**, và **năng lượng cao hay thấp**.

```
                      NĂNG LƯỢNG CAO
                            │
     cang_thang         nang_dong
     (dữ, tối, gắt)     (cường độ cao —
                         KHÔNG cần tích cực)
                            │
  ──────────────────────────┼────────────────────────
                            │
          buon                        vui
      (u sầu, nặng)              (sáng, tích cực —
                                  đây là BẮT BUỘC)
               thu_gian     │
            (êm, dễ chịu)
                            │
                      NĂNG LƯỢNG THẤP

     TIÊU CỰC  ◄──────────────────────────►  TÍCH CỰC
```

Vị trí trong sơ đồ này là **centroid thật đã học được từ dữ liệu**, không phải bốn góc chia đều:

| nhóm | valence | energy |
|---|---|---|
| cang_thang | 0,295 | 0,876 |
| nang_dong | 0,490 | 0,795 |
| vui | 0,653 | 0,643 |
| buon | 0,310 | 0,589 |
| thu_gian | 0,402 | 0,478 |

Hai điều đọc được ngay từ bảng: `nang_dong` nằm ở **giữa** trục valence (0,490) nên nó *không phải*
nhóm tích cực — nó là nhóm **cường độ cao**. Và `buon` có energy **cao hơn** `thu_gian`.

| Mã | Nghe như thế nào | Dấu hiệu |
|---|---|---|
| `vui` | tươi sáng, phấn chấn, dễ mỉm cười | hợp âm sáng, giai điệu bắt tai, nhịp vừa đến nhanh. **Bắt buộc có sắc thái tích cực** |
| `buon` | u sầu, nặng lòng, mất mát, cô đơn | hợp âm tối, giai điệu đi xuống, giọng hát buồn |
| `thu_gian` | êm, nhẹ, dễ chịu, muốn ngồi yên | nhịp chậm–vừa, âm lượng đều, mộc, không gắt |
| `nang_dong` | bị đẩy đi, cường độ cao, thúc ép | nhịp mạnh dứt khoát, beat rõ, dồn dập, âm lượng lớn. **Không cần tích cực** — trung tính hoặc hơi tối vẫn được, miễn chưa tới mức gắt/đe dọa |
| `cang_thang` | dữ dội, gắt, nặng, bất an, hung hãn | méo tiếng, trống dồn, gào, nghe mệt nếu để lâu |
| `khong_ro` | thật sự không xếp được vào đâu | dùng khi bí, nhưng đừng dùng quá 10% số bài |

### Hai điều dễ sai nhất

**`buon` không có nghĩa là "nhạc nhẹ".** Một bài gào thét, trống nặng, guitar méo mà sắc thái là
đau khổ/mất mát thì vẫn là `buon`. Buồn nói về **sắc thái tình cảm**, không nói về độ to. Nếu cứ
mặc định buồn = chậm và yên thì nhãn sẽ lệch hệ thống.

**Nghe nhạc, đừng đọc chủ đề lời.** Một bài có lời về chia tay nhưng nhạc tươi sáng, nhịp nhảy thì
gán theo **phần nghe được**, không theo nội dung lời. Đoạn 30 giây thường không đủ nghe hết lời,
nên cứ phán đoán bằng âm thanh.

### Cách tách bốn cặp hay lẫn

| Lẫn giữa | Hỏi mình câu này |
|---|---|
| `buon` ↔ `thu_gian` | Nghe xong thấy **nặng lòng** (`buon`) hay thấy **dễ chịu, bình yên** (`thu_gian`)? |
| `buon` ↔ `cang_thang` | Cảm giác là **đau/chùng xuống** (`buon`) hay **bức bối/hung hãn** (`cang_thang`)? |
| `nang_dong` ↔ `cang_thang` | Mạnh mà **phấn khích, muốn vận động** (`nang_dong`) hay mạnh mà **tối, gắt, mệt** (`cang_thang`)? |
| `vui` ↔ `nang_dong` | Hỏi **hai câu, theo thứ tự**: (1) *có sáng/tích cực rõ không?* Không → loại `vui`; nếu mạnh mà không tối thì là `nang_dong`. Có → (2) *cường độ có cao đến mức thúc ép không?* Vừa phải → `vui`; rất cao → `nang_dong`. **Đừng hỏi "có muốn nhảy không"** — nhạc vui cũng nhảy được, câu đó không tách được gì (đo thật: `vui` có danceability 0,681, `nang_dong` 0,635) |

Vẫn do dự giữa đúng hai nhóm? Chọn nhóm nào **mạnh hơn một chút** làm `mood_chinh`, nhóm kia ghi
vào `mood_phu`, rồi để `do_tu_tin = 1`. Đừng dừng lại suy nghĩ quá 1 phút cho một bài — nhãn do dự
cũng là dữ liệu, cột `do_tu_tin` sẽ ghi lại điều đó.

---

## 4. Gán nhãn ở đâu, file nào

### Cách nên dùng: trang `annotation/gan_nhan.html`

**Bấm đôi vào `annotation/gan_nhan.html`** là xong — nó mở bằng trình duyệt, không cần cài gì,
không cần tài khoản. Mỗi bài hiện ra kèm trình phát 30 giây ngay tại chỗ; nghe rồi bấm một trong
sáu nhóm là tự sang bài sau.

Vì sao không dùng bảng tính: gán 180 bài trong Sheets nghĩa là bấm link → chờ tab mới → nghe →
quay lại bảng → gõ chữ, lặp 180 lần. Phần lớn thời gian và phần lớn lỗi chính tả nằm ở đó.

Ba điểm cần biết khi dùng trang:

1. **Gõ tên mình vào ô "tên bạn"** ở góc trên trước khi bắt đầu — nó quyết định tên file xuất ra
   và quyết định nhãn của bạn được lưu riêng, không lẫn với người khác nếu dùng chung máy.
2. **Nhãn tự lưu trong trình duyệt sau mỗi lần bấm**, nhưng đó chỉ là bộ nhớ tạm: xoá dữ liệu
   trang, hay dùng cửa sổ ẩn danh, là mất sạch. **Cứ khoảng 30 bài bấm "Lưu file CSV" một lần.**
   Nút "Nạp lại CSV cũ" cho phép làm tiếp từ file đã lưu, kể cả trên máy khác.
3. Xong thì bấm **"Lưu file CSV"**, file tải về tên `nhan_<tên>.csv`. Đặt nó vào `annotation/`,
   ghi đè file trống cùng tên.

Năm nhóm trong trang được đặt **đúng toạ độ centroid đã học** trên mặt phẳng valence × energy, nên
vị trí nút phản ánh khoảng cách thật giữa các nhóm — và nhìn vào là thấy ngay `buon` nằm *cao hơn*
`thu_gian`.

Bàn phím cho nhanh: `1`–`5` chọn nhóm, `6` không rõ, `Shift`+số là nhóm phụ, `space` phát/dừng,
`r` nghe lại, `q w e` độ tự tin, `x` đánh dấu không nghe được, `↵` bài sau, `←` `→` chuyển bài.

### Hoặc làm bằng bảng tính

File của bạn: **`annotation/nhan_<tên>.csv`** — `nhan_nhu.csv`, `nhan_han.csv`, `nhan_thy.csv`.
Chỉ sửa file mang tên mình. Mở bằng **Google Sheets** (File → Import → chọn CSV) thì cột link bấm
được ngay; xong thì File → Download → CSV, đặt lại đúng tên.

Dù làm bằng cách nào thì file nộp cũng phải đúng các cột dưới đây — trang HTML xuất ra đúng
định dạng này. Mỗi hàng là một bài; năm cột đầu đã có sẵn, **bạn điền năm cột sau**:

| Cột | Điền gì |
|---|---|
| `link_nghe_30s` | bấm vào để phát đoạn 30 giây *(có sẵn)* |
| `link_spotify` | mở bản đầy đủ, dùng khi link 30 giây lỗi *(có sẵn)* |
| `ten_bai`, `nghe_si` | chỉ để tìm lại bài; **nghe trước rồi hãy nhìn** *(có sẵn)* |
| **`mood_chinh`** | **bắt buộc** — một trong: `vui` `buon` `thu_gian` `nang_dong` `cang_thang` `khong_ro` |
| **`mood_phu`** | nhóm thứ hai nếu bài ở giữa hai nhóm; không có thì để trống |
| **`do_tu_tin`** | `1` = đoán, `2` = khá chắc, `3` = rất chắc |
| **`nghe_duoc`** | `1` = phát được, `0` = cả hai link đều lỗi |
| `ghi_chu` | tuỳ ý, ví dụ *"nhạc không lời"*, *"preview là đoạn intro"* |

Viết mã nhóm **đúng chính tả, không dấu, chữ thường** (`thu_gian` chứ không phải `thư giãn`) — script
chấm điểm so khớp theo đúng chuỗi này.

**Link lỗi thì sao:** ghi `nghe_duoc = 0`, thử `link_spotify`. Nếu vẫn không được thì để
`mood_chinh` trống — đừng đoán từ tên bài. Script sẽ tự loại các bài đó.

---

## 5. Quy trình

1. **Mỗi người** gán hết 180 bài. Khoảng **2–2,5 giờ**, chia 2–3 buổi được, nhưng đừng đổi tiêu
   chí giữa các buổi — bí thì đọc lại mục 3.
2. Nộp file vào `annotation/` (tên đúng `nhan_<tên>.csv`). Đủ cả ba file mới chạy bước sau.
   Dựng lại trang gán nhãn khi cần: `python -m src.mood.make_label_tool`.
3. Chạy `python -m src.mood.score_agreement`. Script sẽ in:
   - **Fleiss' kappa** — mức đồng thuận giữa ba người
   - **nhãn chốt** = đa số phiếu (2/3 trở lên); 3 người 3 ý thì đánh dấu `tranh_chap`
   - so sánh nhãn người với: nhãn tag T2, dự đoán mô hình, nhãn LLM

   > **Lưu ý về hai dòng sẽ bị thiếu dữ liệu.** Nhãn tag T2 chỉ tồn tại ở 60 bài tầng
   > `labeled` (hai tầng kia chính là chỗ tag bó tay), nên dòng đó sẽ có n ≈ 40–60, không phải
   > 180 — đúng thiết kế, không phải lỗi. Còn **nhãn LLM gần như rỗng** (n ≈ 2–15) vì L1 mới
   > chạy trên 1.000 + 1.597 bài khác, trùng với mẫu 180 này rất ít. Muốn so LLM cho đủ thì
   > phải chạy `labeling_llm` trên đúng 180 bài này — tốn khoảng 15–20 nghìn token, nằm trong
   > hạn mức miễn phí Groq một ngày. **Chạy trước khi gán nhãn cũng được, miễn đừng ai xem kết
   > quả** cho tới khi gán xong.
4. **Họp rà các bài `tranh_chap`.** Chỉ chỉnh **hướng dẫn này** cho rõ hơn, **không chỉnh bảng
   trọng số tag** theo bộ nhãn này — làm vậy là biến nó thành tập huấn luyện và mất vai trò mốc
   độc lập.

### Đọc Fleiss' kappa

| κ | Nghĩa | Nên làm gì |
|---|---|---|
| < 0,20 | gần như không đồng thuận | định nghĩa nhóm có vấn đề, phải viết lại mục 3 rồi gán lại |
| 0,21–0,40 | yếu | nêu thẳng trong báo cáo là hạn chế |
| 0,41–0,60 | trung bình | **mức thực tế hay gặp cho nhãn cảm xúc** — dùng được |
| > 0,60 | tốt | dùng tự tin |

Kappa thấp **không phải thất bại**: nó đo được **giới hạn trên** của bài toán. Nếu ba người chỉ
đồng thuận κ = 0,45 thì không mô hình nào nên được kỳ vọng vượt mức đó, và đó là một phát hiện
đáng viết vào báo cáo.

---

## 6. Những chỗ bộ nhãn này vẫn còn yếu — ghi vào báo cáo

1. **Ba người gán là chính tác giả đồ án**, không phải người ngoài → có thiên lệch kỳ vọng.
2. **Chỉ nghe 30 giây**, và đoạn preview không chắc là đoạn tiêu biểu của bài.
3. **Vẫn thấy tên bài và nghệ sĩ**, nên không loại hết được kiến thức có sẵn.
4. **180 bài** là mẫu nhỏ; chia theo 15 ô thì mỗi ô chỉ 12 bài, đủ để thấy xu hướng chứ chưa đủ
   kết luận chắc cho từng ô.
5. Nhãn là **cảm nhận về bài hát**, không phải cảm xúc người nghe lúc đó — vẫn không trả lời được
   câu "nghe xong có vui hơn không".
6. **`do_tu_tin` mặc định là `2`** trong trang gán nhãn, để mỗi bài chỉ tốn một cú bấm. Nghĩa là
   cột đó lệch về `2` ở những bài người gán không chủ động sửa, nên **đừng dùng nó như thang đo
   định lượng** — chỉ dùng để lọc ra các bài được đánh dấu `1` (đoán) khi phân tích các ca lệch.

---

## 7. Mẫu được chọn thế nào

180 bài lấy **phân tầng**: 12 bài × 3 trạng thái nhãn × 5 nhóm dự đoán, chỉ trong catalogue chính
thức 22.557 bài, thứ tự đã trộn (`seed = 42`).

| Trạng thái | Nghĩa | Vì sao phải có |
|---|---|---|
| `labeled` | tag cho ra nhãn rõ | kiểm nhãn T2 có khớp người nghe không |
| `conflict` | tag mâu thuẫn, T2 bỏ | T2 bỏ đúng hay bỏ oan? |
| `no_emotion_tag` | không có tag cảm xúc | chỗ duy nhất chỉ có mô hình đoán — yếu nhất |

Lấy đều mỗi ô **không** phản ánh phân bố thật (thật thì lệch 7,3:1). Cố ý vậy: để mỗi nhóm có đủ
bài mà xét. Khi báo cáo tỷ lệ khớp toàn cục thì phải nói rõ là **trên mẫu cân bằng**, không phải
trên catalogue.

Dựng lại mẫu:

```bash
python -m src.mood.build_label_set --n-per-cell 12 --labellers nhu,han,thy
```

Lệnh này **không ghi đè** `nhan_*.csv` đã có.

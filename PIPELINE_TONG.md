# PIPELINE TỔNG — DS300 Nhóm 2
## Hệ gợi ý âm nhạc theo gu và tâm trạng mục tiêu

Cập nhật 22/11/2026 · File này dùng để **rà soát**: mỗi chặng ghi rõ đầu vào, lệnh chạy, file
sản phẩm, người chủ trì, trạng thái và **phần còn thiếu**. Không chứa số liệu mới; mọi số dẫn về
file gốc.

Thành viên: **TV1 = Thy** (dữ liệu) · **TV2 = Hân** (mô hình gu) · **TV3 = Như** (mô hình tâm trạng).

Ký hiệu trạng thái: `[x]` xong có bằng chứng · `[~]` làm dở · `[ ]` chưa có.

---

## 0. Sơ đồ toàn tuyến

```
                        ┌─────────────────── NGUỒN THÔ ───────────────────┐
                        │ Music_Info.csv              50.683 bài          │
                        │ User Listening History.csv  9,71 M tương tác    │
                        └────────────────────┬────────────────────────────┘
                                             │
   S1  EDA + làm sạch (TV1)  ────────────────┤
       EDA_report.md · data_preprocessing_report.md
                                             │
   S2  BỘ CHIA CHÍNH THỨC (TV1)  ────────────┤  10-core lặp → 7,03 M · 289.572 user · 22.557 bài
       data/train.csv · data/test.csv           + engagement_score
                                             │
            ┌────────────────────────────────┴────────────────────────────────┐
            │                                                                 │
   ════ NHÁNH GU (TV2) ════                              ════ NHÁNH TÂM TRẠNG (TV3) ════
            │                                                                 │
   S3  Popularity / ALS / fold-in                    S4  T1 kiểm kê 100 tag
       taste-model/src/models.py                         → chỉ 5 tag là cảm xúc
            │                                                                 │
            │                                         S5  T2 nhãn yếu từ tag
            │                                             track_mood_labels.parquet
            │                                                                 │
            │                                         S6  T3 LightGBM / 12 đặc trưng
            │                                             models/mood_classifier.joblib
            │                                                                 │
            │                                         S7  T4 hồ sơ 50.683 bài
            │                                             track_mood.parquet
            │                                             mood_centroids.json
            │                                                                 │
            │                                         S8  T5 API chấm điểm
            │                                             src/mood/api.py
            │                                                                 │
            └────────────────────────┬────────────────────────────────────────┘
                                     │
   S9   GHÉP ĐIỂM                    │   score = α·taste + (1−α)·mood
        src/mood/taste.py  ←── nạp nguyên taste-model/src bằng importlib
                                     │
            ┌────────────────────────┼────────────────────────┐
            │                        │                        │
   S10 Đánh giá A2             S11 Đánh giá T6/A3      S12 Thí nghiệm nhãn
       bốn tập gu×mood             mục tiêu tâm trạng       A1 embedding · L1 LLM
       four_sets_*.csv             mood_eval_*.csv          tag_weights_embed.csv
                                                            llm_compare.csv
                                     │
   S13 BẢNG KẾT QUẢ CHÍNH THỨC  ← ❗ CHƯA CÓ cho nhánh gu (Popularity/ALS/ALS+mood)
                                     │
   S14 Giao diện demo  ← ❗ CHƯA CÓ
                                     │
   S15 Nhãn độc lập + khảo sát mù  ← ❗ CHƯA CÓ
                                     │
   S16 Báo cáo + slide  (bản 05/11 và 09/11 đã có)
```

---

## 1. Bảng trạng thái nhanh

| # | Chặng | Chủ trì | Trạng thái | Chặn việc gì |
|---|---|---|---|---|
| S1 | EDA + làm sạch | TV1 | `[x]` | — |
| S2 | Bộ chia chính thức 10-core | TV1 | `[x]` | — |
| S3 | Popularity / ALS / fold-in | TV2 | `[x]` **có số thật 22/11** | — |
| S4 | T1 kiểm kê tag | TV3 | `[x]` | — |
| S5 | T2 nhãn yếu | TV3 | `[x]` | — |
| S6 | T3 bộ phân loại | TV3 | `[x]` | — |
| S7 | T4 hồ sơ + tâm nhóm | TV3 | `[x]` | — |
| S8 | T5 API | TV3 | `[x]` | — |
| S9 | Ghép điểm | TV3 | `[x]` | — |
| S10 | A2 bốn tập | TV3 | `[x]` | — |
| S11 | T6 cá nhân hóa + A3 | TV3 | `[x]` | — |
| S12 | A1 embedding / L1 LLM | TV3 | `[x]` A1, `[~]` **L1 dở** | S16 |
| S13 | **Bảng G1 — kết quả trên catalogue** | TV2 + TV3 | `[x]` **đủ 3 dòng 22/11** | — |
| S14 | **Giao diện demo** | cả nhóm | `[ ]` | **bảo vệ/seminar** |
| S15 | Nhãn độc lập + khảo sát mù | cả nhóm | `[ ]` | phần "hạn chế" |
| S16 | Báo cáo + slide | cả nhóm | `[~]` chờ S13 | — |

---

## 2. Chi tiết từng chặng

### S1 · EDA và làm sạch — TV1 · `[x]`

| | |
|---|---|
| Vào | `Music Info.csv`, `mood-model/data/raw/User Listening History.csv` |
| Ra | `EDA_report.md`, `data_preprocessing_report.md` |
| Đã xử lý | `tempo=0` (10 bài), `time_signature=0` (10), `year<1950` (13), `spotify_id` trùng (18 dòng), thời lượng bất thường (118) |
| Còn thiếu | — |

### S2 · Bộ chia chính thức — TV1 · `[x]`

| | |
|---|---|
| Ra | `data/train.csv` (5.627.321), `data/test.csv` (1.400.194) |
| Cột mới | `engagement_score` (0–1), `engagement_level` (4 mức) |
| Kiểm tra | 0 cặp (user,bài) trùng train/test · 0 cold-item |
| Lệnh kiểm tra lại | `cd mood-model && python -m src.mood.splits` |
| Còn thiếu | `[ ]` **chưa tách tập validation** — giao thức đòi chọn cấu hình trên validation rồi mới mở test **một lần**. Hiện cả nhóm chọn tham số trực tiếp trên test. |

### S3 · Mô hình gu — TV2 · `[x]` code / `[ ]` số thật

| | |
|---|---|
| Code | `taste-model/src/models.py` (Popularity, ALS weighted implicit, fold-in), `evaluate.py`, `hybrid.py`, `run.py` |
| Kiểm thử | 15 test qua (`VERIFICATION.md`, 28/10) |
| Đã chạy | **chỉ demo dữ liệu giả** 120 user × 240 bài |
| Còn thiếu | `[ ]` chạy trên `data/train.csv` thật · `[ ]` grid-search cấu hình ALS trên validation · `[ ]` điền bảng `taste-model/docs/HANDOFF_CHECKLIST.md` (hiện ghi "Chưa chạy thật") |

> ⚠️ **Lưu ý quan trọng:** ALS trên dữ liệu thật hiện **chỉ tồn tại bên trong mood-model**
> (`src/mood/taste.py`, dùng cho A2). Đó là ALS cấu hình mặc định `factors=32`, 15 vòng, **không
> grid-search**, dựng để làm đối chứng cho A2 — **không phải** kết quả chính thức của nhánh gu.
> Báo cáo cuối vẫn thiếu bảng Popularity / ALS / ALS+mood.

### S4 · T1 kiểm kê tag — TV3 · `[x]`

| | |
|---|---|
| Lệnh | `python -m src.mood.tag_audit` |
| Ra | `data/processed/tag_audit.csv`, `suspect_tracks.csv`, `outputs/top_100_tags.csv` |
| Kết luận | chỉ 5/100 tag là tag cảm xúc; 25 tag trong thiết kế ban đầu không tồn tại |

### S5 · T2 nhãn yếu từ tag — TV3 · `[x]`

| | |
|---|---|
| Lệnh | `python -m src.mood.labeling` |
| Ra | `data/processed/track_mood_labels.parquet` |
| Kết quả | 26.752 bài `labeled` (52,78%) · 5 nhóm · lệch 7,3:1 |
| Phụ thuộc | bảng trọng số 53 tag trong `configs/mood_config.yaml` |

### S6 · T3 bộ phân loại — TV3 · `[x]`

| | |
|---|---|
| Lệnh | `python -m src.mood.train_classifier` |
| Ra | `models/mood_classifier.joblib`, `model_comparison.csv`, `permutation_importance.csv`, `oof_proba.parquet` |
| Chống rò rỉ | tag **không** làm đặc trưng · GroupKFold(5) theo `artist` |
| Kết quả | `gbm_full` macro-F1 **0,571** vs baseline 0,363 |
| Còn thiếu | `[ ]` khoảng tin cậy (bootstrap) cho macro-F1 |

### S7 · T4 hồ sơ tâm trạng — TV3 · `[x]`

| | |
|---|---|
| Lệnh | `python -m src.mood.predict` rồi `python -m src.mood.mood_space` |
| Ra | **`track_mood.parquet`** (file TV2 cần), **`mood_centroids.json`** |
| Phủ | 50.683 bài · tag 52,8% / mô hình 47,2% |

### S8 · T5 API — TV3 · `[x]`

| | |
|---|---|
| File | `src/mood/api.py` |
| Hàm | `mood_scores`, `quick_mood`, `slider_to_mood`, `user_mood_target`, `group_probabilities` |
| Tốc độ | 20.000 bài = 23 ms (hợp đồng < 1 s) |

### S9 · Ghép điểm — TV3 · `[x]`

| | |
|---|---|
| File | `src/mood/taste.py` — nạp `taste-model/src` bằng `importlib` dưới tên `taste_src` |
| Nghĩa | dùng lại đúng `ALS`, `Popularity`, `combine_scores` của TV2, **không sửa dòng nào** |
| Lệnh | `python -m src.mood.taste --split official` |
| Còn thiếu | `[ ]` chốt α cuối bằng validation (hiện α là tham số điều khiển, báo cáo quét 0,3/0,5/0,7 trên test) |

### S10 · A2 bốn tập — TV3 · `[x]`

| | |
|---|---|
| Lệnh | `python -m src.mood.evaluate_four_sets --split official --n-per-set 3 --n-users 60000` |
| Ra | `four_sets_eval_n{3,5,10}_official*.csv`, `four_sets_summary_*.csv`, `reports/figures/four_sets_*.png` |
| Kết quả | ghép 0,5 → set1 **+24,0%** so với chỉ-gu; phân biệt tâm trạng +0,0217 → +0,3594 |
| Biến thể | `_engage` (dùng `engagement_score`) → **+30,4%** |
| Còn thiếu | `[ ]` bootstrap CI theo user · `[ ]` ablation từng tín hiệu của `engagement_score` |

### S11 · T6 cá nhân hóa + A3 — TV3 · `[x]`

| | |
|---|---|
| Lệnh | `python -m src.mood.evaluate_mood --split official --n-users 20000` |
| Ra | `mood_eval_official*.csv`, `mood_eval_summary_*.csv` |
| Kết quả | `personal` **+2,7%** Recall@20 (không phải +12,6%) · `seed3` **−9,0%**, đã loại |
| T7 | `python -m src.mood.journey` → `mood_journey_example.csv`, `figures/mood_journey.png` |

### S12 · A1 embedding `[x]` / L1 LLM `[~]` — TV3

| | A1 | L1 |
|---|---|---|
| Lệnh | `python -m src.mood.labeling_embed --catalogue official` | `python -m src.mood.labeling_llm ...` |
| Ra | `tag_weights_embed.csv`, `tag_group_similarity.csv` | `llm_labels_*.csv`, `llm_compare.csv`, `llm_by_popularity.csv` |
| Kết luận | bảng tay thắng (84,9% vs 52,3%) → **giữ bảng tay** | hiệu chuẩn xong; mở rộng **1.597/9.888 bài**, macro-F1 giảm 0,5656 → 0,5236 |
| Còn thiếu | — | `[ ]` chạy hết hoặc **chốt dừng ở mẫu đại diện**; đang bị chặn bởi quota Groq 200k token/ngày (cần ~712k token) |

---

## 3. Các lỗ hổng (G1 đã lấp 22/11)

### ✅ G1 — ĐÃ XONG 22/11/2026

Test, 2.000 user, catalogue 17.691 bài. Chi tiết: `mood-model/reports/RESULTS.md` mục 21–26.

| Mô hình | Recall@10 | NDCG@10 | MAP@10 | Recall@50 | NDCG@50 | MAP@50 |
|---|---|---|---|---|---|---|
| Popularity | 0,0307 | 0,0243 | 0,0124 | 0,0846 | 0,0418 | 0,0155 |
| **ALS binary** | **0,1464** | **0,1302** | **0,0788** | **0,3007** | **0,1812** | **0,0908** |
| ALS + mood, α=0,9 *(chọn trên validation)* | 0,1453 | 0,1290 | 0,0778 | 0,2992 | 0,1802 | 0,0898 |
| ALS + mood, α=0,5 *(điểm vận hành A2, chỉ báo cáo)* | 0,0662 | 0,0593 | 0,0319 | 0,1222 | 0,0779 | 0,0351 |

- ALS hơn Popularity **4,77× / 5,36× / 6,38×** → module gu có giá trị rõ ràng.
- α=0,9 **ngang** ALS thuần (−0,92% NDCG@10, chênh < 1,3% ở cả 6 chỉ số).
- α=0,5 mất **−54,5%** so với chỉ-gu, nhưng vẫn **hơn Popularity 2,44×**.
- **A2 và G1 lệch nhau về α tốt nhất (0,5 vs 0,9)** — hai mục tiêu khác nhau, và đó là lập luận
  số cho thiết kế thanh kéo α.

Artifact: `mood-model/reports/results/g1_taste_mood_10k/` · Code: `src/mood/evaluate_catalogue.py`

### 🟡 G2 — Validation: đã có cho nhánh gu, còn thiếu cho nhánh mood

| | Trạng thái |
|---|---|
| Cấu hình ALS (`factors`, vòng, confidence) | ✅ TV2 chọn trên validation NDCG@10, test một lần |
| α của G1 | ✅ chọn trên validation (8 giá trị), test một lần |
| `λ`, `σ`, `k` (shrinkage) của nhánh mood | ❌ **vẫn đặt tay trong `mood_config.yaml`, chưa validate** |
| α của A2 | ❌ quét 0,3/0,5/0,7 **trực tiếp trên test** |

Hai dòng cuối vẫn là chỗ dễ bị bắt khi bảo vệ.

### ✅ G5 — Code gu trong repo nhóm: ĐÃ SỬA 22/11/2026

Ban đầu `music-mood-recsys/src/run_real_taste.py` không import được (`src/data.py` và
`src/evaluate.py` trong repo là bản scaffold cũ, khác API). **TV2 đã sửa cùng ngày**:

| Commit | Nội dung |
|---|---|
| `26516b5` | `fix: include data and evaluation modules` — đẩy `data.py` + `evaluate.py` đúng bản |
| `d359738` | merge PR #2 → `reports/results/taste_10k_01/` vào `main` |

Hai file đó **giống hệt từng byte** bản đã dùng dựng G1 → **số liệu G1 không đổi**.

`main` trên GitHub nay ở `d359738`, máy TV3 còn ở `5a206c7` → **cần `git pull`**.
`evaluate_catalogue.py` đã sửa để chọn nguồn **theo từng file** và tự nhận ra bản scaffold cũ,
nên chạy đúng dù đã pull hay chưa.

### ❗ G3 — Chưa có giao diện demo

Seminar hứa thanh kéo tâm trạng + hai chế độ + α. Hiện chỉ có hàm Python. Một demo tối thiểu
(Streamlit/Gradio, 1 file, gọi `api.mood_scores` + `taste.py`) đủ để trình diễn hành trình cảm
xúc T7 — là thứ dễ gây ấn tượng nhất mà nhóm đang có.

### ❗ G4 — Git đang đặt sai chỗ, **không được `git add` lúc này**

```
git rev-parse --show-toplevel  →  /Users/nhule        ← cả thư mục home là repo
git log                        →  0 commit
```

Repo bao trùm toàn bộ home: `.ssh/`, `.claude.json`, `.zsh_history`, `Library/`, mọi project khác.
Cộng thêm dữ liệu lớn ngay trong project:

| File | Dung lượng | Đang bị ignore? |
|---|---|---|
| `data/train.csv` | **887 MB** | ❌ không |
| `data/test.csv` | **212 MB** | ❌ không |
| `Music Info.csv` | 14 MB | ❌ không |
| `mood-model/data/`, `mood-model/.env.local` | — | ✅ có (`mood-model/.gitignore`) |

`git add .` từ đây sẽ stage > 1,1 GB dữ liệu **và khóa SSH**. Cách sửa:

1. Khởi tạo repo **tại đúng thư mục project**, không dùng repo home:
   ```bash
   cd /Users/nhule/Desktop/UIT-DS-SJ/DS300/project
   git init
   ```
2. Viết `.gitignore` tại project trước khi add:
   ```
   data/
   *.csv
   *.parquet
   *.npz
   *.joblib
   __pycache__/
   *.pyc
   .DS_Store
   .env.local
   .venv/
   tmp/
   outputs/.matplotlib/
   ```
3. Dữ liệu và model không vào git — chia sẻ qua Drive, trong README ghi đường dẫn và cách tải.
4. Repo ở home: hỏi lại trước khi xóa `/Users/nhule/.git`, vì nó nằm ngoài phạm vi đồ án.

---

## 4. Việc cần làm — theo ưu tiên

### P0 · Phải có trước khi nộp

| # | Việc | Ai | Ghi chú |
|---|---|---|---|
| ~~1~~ | ~~Tách validation~~ | TV2 | ✅ **xong** — 15% mỗi user, chỉ từ `train.csv` |
| ~~2~~ | ~~Popularity + ALS trên dữ liệu thật~~ | TV2 | ✅ **xong 22/11** |
| ~~3~~ | ~~ALS + mood~~ | TV3 | ✅ **xong 22/11** — G1 đủ 3 dòng |
| ~~1~~ | ~~Chạy lại A2 bằng cấu hình gu chính thức~~ | TV3 | ✅ **xong 22/11 — kết luận không đổi**, +24,0% → **+24,3%** |
| ~~2~~ | ~~TV2 push `data.py` + `evaluate.py`; merge `han/taste-als-report`~~ | TV2 | ✅ **xong 22/11** — xem G5; TV3 cần `git pull` |
| 3 | Chốt **một bộ số duy nhất** cho báo cáo (xem mục 5) | cả nhóm | tránh mâu thuẫn giữa 4 tài liệu |
| 4 | **Demo tối thiểu**: thanh kéo valence–energy + α + nút "hành trình cảm xúc" | TV3 | 1 file Streamlit, không cần đẹp |
| 5 | **Sửa git** — xem cảnh báo G4, đừng commit trước khi sửa | TV3 | repo hiện trỏ vào cả thư mục home |
| 6 | Chốt L1: chạy hết hoặc viết rõ "dừng ở mẫu N bài, kết luận sơ bộ âm" | TV3 | đừng để trạng thái "đang chạy" trong báo cáo cuối |
| 7 | Validate `λ`, `σ`, `k` của nhánh mood (hiện đặt tay) | TV3 | nửa còn lại của G2 |

### P1 · Nâng chất lượng, làm được thì rất đáng

| # | Việc | Ai |
|---|---|---|
| 8 | **Bootstrap CI theo user** cho A2, T6, và macro-F1 T3 (giữ các mood của cùng user trong cùng lần lấy mẫu) | TV3 |
| 9 | Ablation `engagement_score`: từng tín hiệu + đối chứng thang confidence | TV2 |
| 10 | **Tập nhãn độc lập** ~300–500 bài, trải đủ 5 nhóm + bài conflict + bài không tag, 2–3 người gán độc lập, đo Fleiss' kappa | cả nhóm |
| 11 | Dùng tập ở #10 chấm lại T2 vs embedding vs LLM — lần này công bằng vì nhãn không sinh từ tag | TV3 |
| 12 | Thống nhất K: nhánh gu đang dùng @10/@50, nhánh mood @10/@20 | cả nhóm |

### P2 · Hướng phát triển, nêu trong báo cáo chứ không cần làm

- Khảo sát nghe mù: chấm riêng hợp-gu / hợp-mood / chất lượng chuyển bài, đảo thứ tự danh sách.
- Học α riêng từng người thay vì để người dùng kéo.
- Hiểu câu tiếng Việt → (mood, α), theo hướng *Just Ask for Music*.
- Đo trước/sau khi nghe nếu muốn nói về "điều tiết cảm xúc" (kèm quy trình đồng ý tham gia).

---

## 5. Mâu thuẫn số liệu cần chốt

Bốn tài liệu đang giữ số ở các thời điểm khác nhau. **Phải chốt dùng cột "số đúng"** và sửa lại
`TASKS.md` + `decisions.md`, nếu không hội đồng sẽ thấy ba con số cho cùng một phép đo.

**`TASKS.md` và `docs/decisions.md` đã được sửa ngày 22/11/2026.** Bảng dưới là bộ số chuẩn; cột
cuối ghi chỗ **còn lại** cần rà.

| Phép đo | Số cũ (Phần II) | **Số chuẩn (chính thức)** | Còn phải rà ở |
|---|---|---|---|
| T6 `personal` vs tọa độ tay | +12,6% | **+2,7%** | slide cũ |
| A2 set1 ghép 0,5 (N=3) | 0,6146 / +23,9% (N=5) · 0,6106 / +24,0% (`log`/15) | **0,6120 / +24,3%** (`binary`/8, chính thức) | `TASKS.md` Phần II–III nói +24,0% |
| A2 phân biệt tâm trạng | +0,027 → +0,376 | **+0,0208 → +0,3605** | — |
| A1 bảng tay khớp âm thanh | 83,0% | **84,9%** | — |
| A1 bảng tay macro-F1 | 0,5711 | **0,5656** | — |
| A3 `seed3` | +0,1% (ngang) | **−9,0% (loại hẳn)** | — |
| Số test qua | 95 (Phần III) · 117 (05/11) | **119/119** (chạy lại 22/11) | `BAO_CAO_NOI_DUNG_DA_LAM.md` ghi 117 · docx 09/11 |
| Phủ nhóm sau T2 | dự kiến vui 7.555 / buon 8.041 | **thật: vui 2.446 / buon 1.413** | — |

Cách sửa đã dùng, để giữ nhất quán nếu sau này sửa thêm:
- `TASKS.md` là file theo dõi task → **thay số trong dòng "Xong khi"** bằng số chính thức, số
  Phần II để trong ngoặc *(Phần II, bộ chia cũ: …)*.
- `docs/decisions.md` là **nhật ký có ngày** → **không ghi đè**; dòng bị thay thế mang dấu ⚠️ trỏ
  tới dòng mới. Năm dòng đã được gắn dấu: T1 (phủ dự kiến), T6 (cỡ mẫu), A2 (cấu hình ALS còn mở),
  A1 (83,0%/0,5711), A3 (điều kiện mở của `seed3`).

Ngoài ra đừng so trực tiếp hai thước "khớp vị trí âm thanh": **84,9%** tính theo **tag**, còn
**51,7%** ở L1 tính theo **từng bài**. Báo cáo 09/11 đã ghi chú chỗ này, giữ nguyên ghi chú đó.

---

## 6. Checklist trước khi nộp

- [ ] Bảng G1 đã điền số thật, có ghi cohort/K/seed/backend
- [ ] Mọi tham số nêu trong báo cáo đều ghi rõ **chọn trên validation**, test chạy một lần
- [ ] Không còn số Phần II lẫn trong báo cáo cuối (đối chiếu mục 5)
- [ ] Demo chạy được trên máy sẽ mang đi bảo vệ, không phụ thuộc đường dẫn tuyệt đối
- [ ] `python -m pytest` trong `mood-model/` qua hết, ghi số test vào báo cáo
- [ ] `taste-model`: `python -m unittest discover -s tests` qua hết
- [ ] `git rev-parse --show-toplevel` trả về **thư mục project**, không phải `/Users/nhule` (G4)
- [ ] `.gitignore` tại project loại `data/` · `*.csv` · `*.parquet` · `*.npz` · `*.joblib` · `__pycache__` · `.env.local`
- [ ] `git status` không liệt kê `train.csv` / `test.csv` / `.ssh` / file ngoài project
- [ ] `.env.local` **không** bị commit (đang chứa khóa Groq)
- [ ] Mục "Hạn chế" nêu chủ động: nhãn yếu, 3/5 nhóm không có tag cảm xúc, đánh giá có vòng lặp, chưa khảo sát người thật
- [ ] Phụ lục bản đồ mã nguồn khớp với cây thư mục thật

---

## 7. Bản đồ file sản phẩm

| Nhóm | File | Dùng để |
|---|---|---|
| Bàn giao TV2 | `mood-model/data/processed/track_mood.parquet` | hồ sơ tâm trạng 50.683 bài |
| Bàn giao TV2 | `mood-model/data/processed/mood_centroids.json` | vị trí 5 nhóm trên thanh kéo |
| Bàn giao TV2 | `mood-model/src/mood/api.py` | hàm chấm điểm tâm trạng |
| Nối hai nhánh | `mood-model/src/mood/taste.py` | nạp code TV2 |
| **Số G1** | `mood-model/reports/results/g1_taste_mood_10k/metrics.csv` | **bảng G1 + đường cong α** |
| **Số G1** | `.../selected_alpha.json`, `.../manifest.json` | α đã chọn · SHA256 dữ liệu, cấu hình |
| **Code G1** | `mood-model/src/mood/evaluate_catalogue.py` | dựng bảng G1, lớp `TasteMoodHybrid` |
| Số gu của TV2 | `music-mood-recsys` `9ddd4b5:reports/results/taste_10k_01/` | bằng chứng tái lập của TV2 (chưa merge vào main) |
| Số A2 | `data/processed/four_sets_summary_n{3,5,10}_official*.csv` | bảng bốn tập |
| Số T6/A3 | `data/processed/mood_eval_summary_official*.csv` | bảng mục tiêu tâm trạng |
| Số A1 | `data/processed/tag_weights_embed.csv` | so sánh cách gán nhãn |
| Số L1 | `data/processed/llm_compare.csv`, `llm_by_popularity.csv` | thí nghiệm LLM |
| Báo cáo | `BAO_CAO_NOI_DUNG_DA_LAM.md` | bản dựng slide (05/11) |
| Báo cáo | `BAO_CAO_THUYET_MINH_DO_AN_NHOM_2.docx` | bản cho giảng viên (09/11) |
| Báo cáo | `mood-model/reports/bao_cao_mood_model.html` | báo cáo module tâm trạng |
| Nhật ký | `mood-model/docs/decisions.md` | 30 quyết định kèm lý do |

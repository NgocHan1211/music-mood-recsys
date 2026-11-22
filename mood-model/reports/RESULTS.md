# Kết quả module tâm trạng

Mỗi task ghi dưới tiêu đề tương ứng. Mọi số liệu phải đến từ code đã chạy.

## T0. Khởi tạo

Chạy ngày 2026-10-20. Lệnh: `python -m src.mood.io [--sample]`, `python -m pytest tests -q`.

### Môi trường

- Python 3.13.13 (miniconda base). Đã cài đủ `requirements.txt`.
- `lightgbm` 4.7.0 ban đầu import lỗi (thiếu `libomp.dylib`); đã cài `libomp` bằng Homebrew → import được.
  Vậy T3 dùng được LightGBM thật, không cần fallback `HistGradientBoostingClassifier`.

### Dữ liệu

- Chưa có `data/processed/tracks.parquet` của TV1 → đọc `data/raw/Music_Info.csv`
  (copy từ `../Music Info.csv`), tự làm sạch tối thiểu theo `docs/eda_summary.md`.
- CSV có BOM ở đầu → đọc với `encoding="utf-8-sig"`.

### Kết quả `load_tracks()` trên toàn bộ dữ liệu

| Chỉ số | Giá trị | Khớp EDA |
|---|---|---|
| Số bài | 50.683 | có |
| Số cột sau khi đọc | 25 (21 cột gốc + `spotify_id_dup`, `duration_outlier`, `tags_list`, `n_tags`) | — |
| `track_id` trùng | 0 | — |
| Bài thiếu tag | 1.127 (2,22%) | có |
| Số tag khác nhau sau chuẩn hóa | 100 | có |
| Trung bình tag/bài | 3,74 (tính trên bài có tag) — 3,66 nếu tính cả bài thiếu tag | có |
| Bài có tag trùng trong cùng một bài | 0 | — |

Làm sạch tối thiểu đã áp dụng (số lượng đều khớp `docs/eda_summary.md`):

| Lỗi | Số lượng | Xử lý trong code |
|---|---|---|
| `tempo = 0` | 10 bài | gán NaN rồi điền trung vị |
| `time_signature = 0` | 10 bài | gán NaN |
| `year < 1950` | 13 bài | gán NaN |
| `spotify_id` trùng | 18 dòng (9 giá trị) | cờ `spotify_id_dup = True` |
| Thời lượng bất thường | 118 bài | cờ `duration_outlier = True`, loại khỏi tập huấn luyện ở T3 |

Ngưỡng thời lượng bất thường được chọn là **< 30 giây hoặc > 900 giây (15 phút)**, vì đây là
ngưỡng cho ra đúng 118 bài như báo cáo EDA (các ngưỡng khác: 30–600s → 577 bài, 60–600s → 966 bài).

### Bản mẫu (`--sample`)

5.000 bài, `random_state = 42`, lặp lại cho cùng kết quả (có test). Trong bản mẫu: 122 bài thiếu tag,
6 dòng `spotify_id_dup`, 15 bài `duration_outlier`.

### Code đã viết

- `src/mood/config.py`: `load_config()` (đọc `configs/mood_config.yaml`, kiểm tra `groups` khớp `GROUPS`),
  các đường dẫn chuẩn, `GROUPS`, `SEED = 42`.
- `src/mood/io.py`: `parse_tags()` (chữ thường, bỏ khoảng trắng thừa, bỏ tag rỗng và tag trùng, giữ thứ tự),
  `load_tracks(sample)` (ưu tiên parquet của TV1, nếu không thì CSV + làm sạch), CLI `--sample`.
- `tests/test_io.py`: 14 test, **tất cả qua** — chuẩn hóa tag (8 ca), đủ cột hợp đồng,
  `track_id` không trùng, `tags` khớp `tags_list`, các luật làm sạch, bản mẫu lặp lại được.

### Lưu ý cho task sau

- Thời gian đọc toàn bộ CSV ~0,3 giây → không cần cache parquet riêng.
- Tag trong dữ liệu dùng dấu gạch dưới (`alternative_rock`, `hard_rock`), nên tag ứng viên trong
  config có khoảng trắng (`"feel good"`) có thể không tồn tại — T1 phải đối chiếu.

## T1. Kiểm kê tag và rà dữ liệu

Chạy ngày 2026-10-21. Lệnh: `python -m src.mood.tag_audit [--sample]` (chạy bản mẫu trước, rồi toàn bộ 50.683 bài).
Đầu ra: `data/processed/tag_audit.csv`, `data/processed/tag_candidate_check.csv`,
`data/processed/suspect_tracks.csv`, `reports/figures/top30_tags.png`.

### 1.1 Phân loại 100 tag

Thứ tự ưu tiên khi gán category: decade → country → genre → descriptor → emotion → other.
Danh sách `genre` (74 tag) và `descriptor` (7 tag) được bổ sung vào `configs/mood_config.yaml`
từ chính 100 tag có thật; `decade` thêm `oldies`, `country` thêm `polish`, `russian`.

| Loại | Số tag | Số lượt gán | Ví dụ |
|---|---|---|---|
| genre | 74 | 142.304 | rock (10.684), indie (7.287), electronic (6.594) |
| other (mô tả giọng/nhạc cụ/phiên bản) | 7 | 14.072 | female_vocalists (4.517), instrumental (2.354), acoustic (2.043), cover (953) |
| decade | 6 | 11.025 | 80s (2.307), 90s (2.165), oldies (1.422) |
| emotion | 5 | 9.697 | xem bảng dưới |
| country | 8 | 8.215 | british (2.410), american (1.423) |

Hình: `reports/figures/top30_tags.png` (top 30 tag, tô màu theo loại) — trong top 30 chỉ có
**2 tag cảm xúc** (`chillout`, `mellow`).

### 1.2 Tag cảm xúc THẬT CÓ — kết quả quan trọng nhất của T1

Chỉ **5/100 tag** là tag cảm xúc, và cả 5 đều nằm cùng một vùng trên mặt phẳng valence–energy
(valence thấp–trung bình, energy thấp hơn trung bình chung là 0,433 / 0,686):

| Tag | Số bài | valence TB | energy TB | acousticness TB | Nhóm gợi ý |
|---|---|---|---|---|---|
| chillout | 2.655 | 0,383 | 0,523 | 0,358 | thu_gian |
| mellow | 1.901 | 0,357 | 0,437 | 0,477 | thu_gian (+buon) |
| beautiful | 1.825 | 0,293 | 0,463 | 0,451 | mơ hồ (buon / thu_gian) |
| love | 1.680 | 0,408 | 0,564 | 0,326 | mơ hồ (chủ đề lời hát, không phải tâm trạng) |
| chill | 1.636 | 0,419 | 0,528 | 0,364 | thu_gian |

Số bài có ít nhất 1 tag cảm xúc: **6.464 / 50.683 (12,75%)**.

**25 tag ứng viên trong config KHÔNG tồn tại trong dữ liệu:**
aggressive, angry, atmospheric, calm, cheerful, dark, depressing, emotional, energetic, epic,
feel good, fun, happy, heartbreak, heavy, intense, melancholic, melancholy, party, relaxing,
romantic, sad, sexy, upbeat, workout.

Tag ứng viên có tồn tại: `chill`, `chillout`, `mellow` (cảm xúc) và `acoustic`, `ambient`,
`dance` (thực chất là tag mô tả / thể loại). Trong danh sách `ambiguous` chỉ có
`beautiful`, `love`, `psychedelic` tồn tại.

**Hệ quả:** nếu chỉ dùng tag cảm xúc trực tiếp thì phủ theo nhóm là:

| Nhóm | Số bài (chỉ tag cảm xúc) | Đạt `min_tracks_per_group` = 300? |
|---|---|---|
| thu_gian | 4.618 | đạt |
| buon | 1.825 (chỉ nhờ `beautiful`, vốn mơ hồ) | đạt nhưng yếu |
| vui | **0** | KHÔNG |
| nang_dong | **0** | KHÔNG |
| cang_thang | **0** | KHÔNG |

Đây đúng là rủi ro mà `docs/eda_summary.md` đã cảnh báo, và nặng hơn dự kiến: 3/5 nhóm
không có một tag cảm xúc nào. Đây là lý do phải DỪNG và hỏi người dùng trước khi sang T2.

### 1.3 Hồ sơ âm thanh của tag (căn cứ để mở rộng ánh xạ)

Tuy không có tag cảm xúc cho 3 nhóm, nhiều tag **thể loại** lại tách rất rõ trên valence/energy
(trung bình toàn bộ: valence 0,433; energy 0,686):

| Vùng | Tag tiêu biểu (valence / energy) |
|---|---|
| Vui (valence cao) | reggae 0,72/0,64 · ska 0,71/0,79 · funk 0,66/0,68 · dance 0,60/0,78 · hip_hop 0,59/0,70 |
| Căng thẳng (valence thấp, energy rất cao) | metalcore 0,27/0,92 · grindcore 0,21/0,90 · death_metal 0,23/0,91 · hardcore 0,33/0,91 · black_metal 0,20/0,83 |
| Buồn (valence thấp, energy thấp–trung) | dark_ambient 0,16/0,44 · classical 0,20/0,28 · post_rock 0,23/0,56 · piano 0,30/0,39 |
| Thư giãn (energy thấp, acousticness cao) | acoustic 0,40/0,42 · folk 0,42/0,44 · jazz 0,47/0,43 · new_age 0,27/0,40 · ambient 0,25/0,45 |
| Năng động (energy cao, danceability cao) | house 0,51/0,73 · techno 0,43/0,74 · electro 0,51/0,72 · trance 0,39/0,78 |

Nếu cho phép dùng tag thể loại làm **bằng chứng cảm xúc phụ**, phủ theo nhóm (bản phác thảo,
chưa tính mâu thuẫn): vui 7.555 · buon 8.041 · thu_gian 11.044 · nang_dong 7.939 · cang_thang 10.299;
tổng 35.045/50.683 bài (69,1%) có ít nhất một tag dùng được.

### 1.4 Rà bài nghi ghép nhầm phiên bản

Không có `User_Listening_History.csv` trong `data/raw/` nên "50 bài phổ biến nhất" dùng proxy
**số tag/bài** theo `docs/mood_spec.md`. Lưu ý: proxy này yếu — top theo số tag là các nghệ sĩ ít
tên tuổi nhưng được tag kỹ (Chelsea Wolfe, NORA&LEO), không phải bài nhiều người nghe. Khi TV1
đưa dữ liệu lịch sử nghe, nên rà lại theo số người nghe.

Kiểm chứng 2 bài mà EDA nghi ngờ: cả hai **đều có tag `cover`** →
- `Bring Me To Life` — Katherine Jenkins (2012): tags = classical, cover, new_age; valence 0,298 energy 0,563,
  trong khi bản gốc Evanescence (2003) có energy 0,930.
- `Party In The U.S.A.` — The Barden Bellas (2012): tags = soundtrack, cover.

Vậy tag `cover` là tín hiệu chính. Các dấu hiệu đã cài trong `find_suspect_tracks()`:

| Dấu hiệu | Số bài | Mức |
|---|---|---|
| `cover_tag` (có tag `cover`) | 953 | mạnh |
| `name_version` (tên có live/remix/version/remaster/karaoke/tribute) | 285 | mạnh |
| `spotify_id_dup` | 18 | mạnh |
| `duration_outlier` | 118 | mạnh |
| `era_mismatch` (tag thập niên cách năm phát hành > 20 năm) | 2.752 | yếu |
| `title_shared` (trùng tên bài với nghệ sĩ khác) | 1.614 | yếu |
| **Hợp các lý do mạnh** | **1.367 (2,70%)** | loại khỏi huấn luyện ở T3 |
| Tổng số bài bị gắn cờ | 5.263 (10,38%) | ghi vào `suspect_tracks.csv` |

`era_mismatch` bị xếp **yếu** vì nhiễu nhiều: nhạc phát hành 2017 gắn tag `90s`/`80s` thường là
do phong cách (Chelsea Wolfe, That Poppy), không phải bản cover.

Trong 50 bài "phổ biến" theo proxy: 19 bài bị gắn cờ, 8 bài có lý do mạnh (đều là `cover_tag`:
Dorfdisko – Hilferuf, Kool Savas – Keep it Gangsta, NORA&LEO × 4, Krach – Der Brief,
Callaghan – El camino).

### 1.5 Quyết định đã được người dùng xác nhận (2026-10-21)

1. **Nguồn nhãn cho T2:** tag cảm xúc trực tiếp (trọng số 1.0) **+** tag thể loại mang cảm xúc rõ
   (trọng số thấp hơn, ~0.5). Giữ đủ 5 nhóm. Không dùng nhãn luật từ valence/energy vì sẽ tạo
   vòng lặp logic với bộ phân loại (T3) vốn dùng chính các đặc trưng đó.
2. **Tag mơ hồ:** `beautiful` → {buon: 0.5, thu_gian: 0.5}; `love` và `psychedelic` không dùng làm nhãn.

Chi tiết lý do và số liệu: `docs/decisions.md`.

## T2. Nhãn yếu từ tag

Chạy ngày 2026-10-21. Lệnh: `python -m src.mood.labeling [--sample]`.
Đầu ra: `data/processed/track_mood_labels.parquet`, `reports/figures/mood_valence_energy_box.png`.

### 2.1 Bảng trọng số tag → nhóm

Chốt trong `configs/mood_config.yaml`, khối `labeling_tags`: **53 tag** (đều có thật trong dữ liệu,
có test kiểm tra), gồm 4 tag cảm xúc trực tiếp (trọng số 1.0) và 49 tag thể loại/mô tả mang cảm xúc
(trọng số nhân `support_weight = 0.5`). Nhóm gợi ý của mỗi tag dựa trên hồ sơ valence/energy ở mục 1.3.

```
nhãn mềm(bài) = chuẩn hóa( Σ_tag trọng_số(tag → nhóm) )
```

Ví dụ bài có tag `chill` + `metal`: y_thu_gian = 1,0/1,5 = 0,667; y_cang_thang = 0,7×0,5/1,5 = 0,233;
y_nang_dong = 0,3×0,5/1,5 = 0,100.

### 2.2 Xử lý mâu thuẫn và nhãn cứng

- `conflict` = **không nhóm nào trội**: hai nhóm dẫn đầu chênh nhau < `conflict_margin` = 0,2,
  **hoặc** nhóm dẫn đầu dưới `min_label_confidence` = 0,5.
- `mood_label` = nhóm cao nhất, chỉ khi không phải conflict.
- (Bản cài đầu tiên để 102 bài có `label_status = labeled` nhưng `mood_label` rỗng — nhóm đầu
  cách nhóm hai đủ xa nhưng vẫn dưới 0,5. Đã gộp hai điều kiện lại, nay mọi bài `labeled` đều có
  nhãn cứng, có test kiểm tra.)
- Thứ tự ưu tiên `label_status`: `no_emotion_tag` → `suspect` → `conflict` → `labeled`.

### 2.3 Kết quả trên toàn bộ 50.683 bài

| label_status | Số bài | Tỷ lệ |
|---|---|---|
| labeled | 26.752 | 52,78% |
| conflict | 10.089 | 19,91% |
| suspect (lý do mạnh ở T1) | 960 | 1,89% |
| no_emotion_tag | 12.882 | 25,42% |

Tổng số bài có nhãn mềm: 37.801 (74,58%). Các bài `conflict`/`suspect` vẫn có nhãn mềm nhưng
bị loại khỏi tập huấn luyện ở T3.

**Số bài mỗi nhóm (chỉ `labeled`) — mọi nhóm đều đạt `min_tracks_per_group` = 300:**

| Nhóm | Số bài | valence (trung vị) | energy (trung vị) | acousticness (trung vị) | Tag dẫn dắt |
|---|---|---|---|---|---|
| vui | 2.446 | 0,706 | 0,643 | 0,137 | soul (1.138), reggae (990), funk (792), ska (285) |
| buon | 1.413 | 0,228 | 0,624 | 0,082 | post_rock (537), beautiful (423), emo (306), dark_ambient (206) |
| thu_gian | 8.562 | 0,369 | 0,474 | 0,444 | chillout (2.480), folk (2.126), acoustic (1.658), ambient (1.658) |
| nang_dong | 4.012 | 0,530 | 0,829 | 0,011 | dance (1.628), electro (1.156), techno (933), house (738) |
| cang_thang | 10.319 | 0,271 | 0,933 | 0,000 | metal (2.839), death_metal (1.882), heavy_metal (1.724), metalcore (1.660) |

Nhãn mềm trung bình (trên các bài có nhãn): y_vui 0,114 · y_buon 0,145 · y_thu_gian 0,253 ·
y_nang_dong 0,205 · y_cang_thang 0,283.

### 2.4 Nhóm `buon` lúc đầu chỉ có 60 bài — đã xử lý

Với bảng trọng số ban đầu, `buon` chỉ được **60 bài** (< 300) vì không tag nào cho `buon` trọng số
trội: `beautiful`, `dark_ambient`, `doom_metal`, `gothic` đều chia 50/50 với `thu_gian` hoặc
`cang_thang` nên `buon` luôn thua argmax, dù khối lượng nhãn mềm của nó không nhỏ (0,122).

Đã thử 5 phương án trên toàn bộ dữ liệu:

| Phương án | vui | buon | thu_gian | nang_dong | cang_thang | conflict |
|---|---|---|---|---|---|---|
| Trọng số ban đầu | 2.446 | **60** | 9.308 | 4.021 | 10.689 | 10.250 |
| Chỉnh trọng số theo valence (đã chọn) | 2.446 | **1.413** | 8.562 | 4.012 | 10.319 | 10.089 |
| + `min_conf` 0,45 | 2.447 | 1.448 | 8.589 | 4.015 | 10.338 | 9.987 |
| + `conflict_margin` 0,1 | 3.727 | 2.408 | 9.778 | 5.229 | 10.826 | 4.259 |
| Chỉ nới ngưỡng, giữ trọng số cũ | 3.736 | 755 | 11.152 | 5.249 | 11.482 | 4.046 |

**Đã chọn: chỉnh trọng số, giữ nguyên ngưỡng** (người dùng duyệt 2026-10-21). Các tag được chỉnh,
tất cả đều theo hướng "valence càng thấp thì càng nghiêng về buon":
`dark_ambient` 0,5→0,7 (valence 0,16 — thấp nhất toàn bộ, energy chỉ 0,44 nên u ám hơn là gắt) ·
`post_rock` 0,6→0,8 (valence 0,23) · `beautiful` 0,5→0,7 (valence 0,29 — thấp nhất trong 5 tag cảm xúc) ·
`emo` 0,6→0,7 · `doom_metal`, `gothic` 0,5→0,6 · `classical`, `piano` 0,4→0,5 ·
`gothic_metal`, `symphonic_metal` 0,4→0,5 · `mellow`, `new_age` 0,2→0,3.

### 2.5 Kiểm định Kruskal–Wallis (M8 câu hỏi 1)

Trên 26.752 bài `labeled`, so sánh phân phối đặc trưng giữa 5 nhóm:

| Đặc trưng | H | p-value |
|---|---|---|
| acousticness | 13.660 | < 1e-300 |
| energy | 12.369 | < 1e-300 |
| danceability | 8.868 | < 1e-300 |
| loudness | 8.050 | < 1e-300 |
| valence | 4.819 | < 1e-300 |

Cả 5 đặc trưng đều khác nhau có ý nghĩa giữa các nhóm. Đáng chú ý: **energy và acousticness tách
nhóm mạnh hơn valence** (H gấp 2,5–2,8 lần), tức thanh kéo chỉ dùng (valence, energy) như bản
baseline đã bỏ sót thông tin — đây là bằng chứng đầu tiên ủng hộ cải tiến M3.

Hình `reports/figures/mood_valence_energy_box.png` (boxplot valence và energy theo nhóm).

### 2.6 Hạn chế của nhãn T2 (ghi vào báo cáo cuối)

1. **`buon` và `cang_thang` gần như trùng nhau về valence** (trung vị 0,228 vs 0,271), chỉ tách rõ
   ở energy (0,624 vs 0,933). Lý do: nguồn nhãn của `buon` là post_rock/emo/doom/gothic — nhạc
   trầm buồn nhưng **mạnh**, không phải ballad nhẹ. Bộ tag này không có tag nào cho nhạc buồn nhẹ.
2. **Mất cân bằng nhóm:** `cang_thang` (10.319) gấp 7,3 lần `buon` (1.413) → T3 phải dùng
   `class_weight` cân bằng và báo cáo F1 từng nhóm, không chỉ accuracy.
3. **19,91% bài bị `conflict`**, chủ yếu do tag thể loại kéo về hai nhóm ngang nhau.
4. **25,42% bài không có tag nào dùng được** → đây chính là phần mà bộ phân loại T3 phải suy ra.
5. Nhãn của `vui` phụ thuộc nhiều vào `soul` (1.138/2.446 bài) — nếu tag `soul` bị hiểu sai thì
   nhóm `vui` bị lệch theo.

### 2.7 Test

`tests/test_labeling.py` — 7 test, tất cả qua (tổng cả module: **21 test**):
tag trong bảng trọng số đều có thật và tổng trọng số = 1; nhãn mềm tổng = 1 và không NaN một phần;
bài `no_emotion_tag` có nhãn NaN; ví dụ tính tay `chill` + `metal`; luật conflict và ngưỡng tự tin;
giá trị `label_status` hợp lệ; mọi nhóm đạt `min_tracks_per_group` trên toàn bộ dữ liệu.

## T3. Bộ phân loại

Chạy ngày 2026-10-23. Lệnh: `python -m src.mood.train_classifier [--sample]`.
Tập huấn luyện: **26.752 bài** `labeled` (loại `conflict`, `suspect`, `no_emotion_tag`),
5.314 nghệ sĩ, 12 đặc trưng. Chia fold: **GroupKFold 5 fold theo `artist`**.

### 3.1 Đặc trưng (M3)

`valence, energy, danceability, acousticness, instrumentalness, speechiness, liveness,
loudness, tempo, mode` + `key` mã hóa **sin/cos** (key là biến vòng: key 11 và key 0 phải gần
nhau — có test kiểm tra). **Không** dùng `tags` (nguồn sinh nhãn → rò rỉ), `genre` hay `artist`;
có test chặn.

### 3.2 So sánh 4 mô hình (out-of-fold, 26.752 bài)

| Mô hình | Số đặc trưng | macro-F1 | accuracy | log-loss | F1 vui | F1 buon | F1 thu_gian | F1 nang_dong | F1 cang_thang |
|---|---|---|---|---|---|---|---|---|---|
| baseline_ve (tọa độ tự đặt) | 2 | 0,363 | 0,440 | 2,374 | 0,374 | 0,170 | 0,309 | 0,273 | 0,689 |
| logreg_ve | 2 | 0,441 | 0,524 | 1,201 | 0,409 | 0,195 | 0,544 | 0,324 | 0,736 |
| logreg_full | 12 | 0,516 | 0,595 | 1,042 | 0,463 | 0,241 | 0,577 | 0,505 | 0,793 |
| **gbm_full** (LightGBM) | 12 | **0,571** | 0,677 | **0,859** | 0,505 | 0,239 | 0,702 | 0,588 | 0,822 |

Mọi mô hình dùng `class_weight="balanced"` vì nhãn lệch 7,3:1 (xem T2).
`baseline_ve` là bản gốc cần cải tiến: gán nhóm theo khoảng cách Euclid tới tọa độ tự đặt tay,
xác suất = softmax của −d²/2σ² để so được log-loss.

**Trả lời M8:**
- **Câu 3 (bộ phân loại so với baseline tọa độ tự đặt):** macro-F1 **0,363 → 0,571**, tăng **57%**
  tương đối. Accuracy 0,440 → 0,677.
- **Câu 2 (thêm đặc trưng ngoài valence/energy):** macro-F1 **0,441 → 0,516** với cùng Logistic
  Regression (+17% tương đối), và lên 0,571 với LightGBM. Riêng nhóm `nang_dong` tăng mạnh nhất:
  F1 0,324 → 0,505 → 0,588.

### 3.3 Ma trận nhầm lẫn (gbm_full)

`reports/figures/confusion_gbm_full.png` (chuẩn hóa theo hàng). Chéo chính:
vui 0,58 · buon **0,29** · thu_gian 0,67 · nang_dong 0,62 · cang_thang 0,78.

Nhóm `buon` bị nhầm sang `thu_gian` (0,37) và `cang_thang` (0,22) — đúng như hạn chế đã nêu ở
mục 2.6: nguồn nhãn của `buon` là post_rock/emo/doom/gothic, tức nhạc buồn nhưng **mạnh**, nên nó
nằm chen giữa hai nhóm kia trên mặt phẳng valence–energy.

### 3.4 Hiệu chỉnh xác suất — đã chọn KHÔNG hiệu chỉnh

| Cách | macro-F1 | accuracy | log-loss | ECE | F1 vui | **F1 buon** | F1 thu_gian | F1 nang_dong | F1 cang_thang |
|---|---|---|---|---|---|---|---|---|---|
| **none (đã chọn)** | **0,571** | 0,677 | 0,859 | **0,041** | 0,505 | **0,239** | 0,702 | 0,588 | 0,822 |
| isotonic | 0,523 | 0,715 | 0,805 | 0,047 | 0,446 | **0,007** | 0,748 | 0,579 | 0,833 |
| sigmoid | 0,523 | 0,716 | 0,803 | 0,055 | 0,450 | **0,006** | 0,749 | 0,579 | 0,832 |

Cả hai cách hiệu chỉnh đều **xóa sổ nhóm `buon`** (F1 0,007 và 0,006 — gần như không bao giờ được
dự đoán), vì chúng hiệu chỉnh theo tần suất thực tế nên triệt tiêu tác dụng của
`class_weight="balanced"` với nhóm nhỏ nhất. Đổi lại chỉ được log-loss nhỉnh hơn (0,859 → 0,803)
còn **ECE lại xấu đi** (0,041 → 0,047 / 0,055). Vì ECE của mô hình gốc đã tốt sẵn, chọn **không
hiệu chỉnh**. Đây là thay đổi so với `mood_spec.md` M3 — ghi trong `docs/decisions.md`.

`reports/figures/reliability.png`: mô hình gốc hơi quá tự tin một cách đều đặn, isotonic chỉnh
quá tay thành thiếu tự tin ở vùng xác suất cao.

### 3.5 Permutation importance — trả lời "ngoài valence/energy, đặc trưng nào mang cảm xúc"

Đo bằng mức giảm macro-F1 khi xáo trộn từng đặc trưng (fold đầu, 5 lần lặp):

| Đặc trưng | Mức giảm macro-F1 |
|---|---|
| **acousticness** | 0,1011 |
| **danceability** | 0,0994 |
| energy | 0,0608 |
| valence | 0,0399 |
| instrumentalness | 0,0377 |
| tempo | 0,0352 |
| speechiness | 0,0185 |
| loudness | 0,0145 |
| liveness | 0,0035 |
| key_sin / mode / key_cos | ≈ 0 |

**acousticness và danceability quan trọng hơn cả energy, và gấp 2,5 lần valence** — tức hai trục
của thanh kéo cũ (valence, energy) không phải là hai đặc trưng mang cảm xúc mạnh nhất. Kết quả này
khớp với kiểm định Kruskal–Wallis ở mục 2.5 (acousticness H lớn nhất, valence nhỏ nhất).
`key` và `mode` gần như vô dụng — trưởng/thứ không giúp đoán tâm trạng trong bộ dữ liệu này.

### 3.6 Mô hình cuối

LightGBM (`n_estimators=400`, `learning_rate=0.05`, `num_leaves=31`, `class_weight="balanced"`,
`random_state=42`), huấn luyện trên toàn bộ 26.752 bài có nhãn, lưu ở `models/mood_classifier.joblib`
kèm danh sách cột đặc trưng. Không có cảnh báo macro-F1 < 0,3.

### 3.7 Lỗi đã sửa trong lúc làm

`log_loss` của scikit-learn sắp nhãn theo thứ tự chữ cái, trong khi ma trận xác suất của ta xếp
theo `GROUPS`. Bản đầu truyền thẳng `labels=GROUPS` nên log-loss bị tính sai hoàn toàn
(gbm_full ra 8,220 dù accuracy 0,683). Đã xếp lại cột trước khi tính và thêm test
`test_log_loss_column_order`.

## T4. Hồ sơ tâm trạng và không gian tâm trạng

Chạy ngày 2026-10-23. Lệnh: `python -m src.mood.predict`, `python -m src.mood.mood_space`.
Đầu ra: `data/processed/track_mood.parquet`, `data/processed/mood_centroids.json`,
`reports/figures/mood_space.png`.

### 4.1 Hồ sơ tâm trạng cho mọi bài (`track_mood.parquet`)

Bài có nhãn tag sạch (`label_status = labeled`): trộn nhãn tag với dự đoán mô hình theo
`predict.tag_weight = 0.7` (70% nhãn người nghe, 30% mô hình), `mood_source = tag`.
Bài còn lại (kể cả `conflict`, `suspect`, `no_emotion_tag`): hoàn toàn từ mô hình,
`mood_source = model`.

| | Số bài |
|---|---|
| Tổng | 50.683 (mỗi bài đúng 1 dòng, tổng xác suất = 1 — có test) |
| `mood_source = tag` | 26.752 (52,8%) |
| `mood_source = model` | 23.931 (47,2%) |

Phân bố `mood_pred`: thu_gian 15.806 (31,2%) · cang_thang 15.884 (31,3%) · nang_dong 7.823 (15,4%) ·
vui 6.552 (12,9%) · buon 4.618 (9,1%).
Độ tự tin trung bình 0,696 — bài có nhãn tag 0,768, bài mô hình suy ra 0,616 (hợp lý: mô hình
thận trọng hơn ở phần dữ liệu không có bằng chứng từ người nghe).

**Nhóm `buon` được "cứu" ở bước này:** từ 1.413 bài có nhãn tag lên 4.618 bài có hồ sơ tâm trạng,
nhờ mô hình suy ra từ âm thanh cho các bài không có tag — đây chính là tác dụng của cải tiến M3.

### 4.2 Không gian tâm trạng học từ dữ liệu (`mood_centroids.json`)

Tâm và hiệp phương sai của (valence, energy), trọng số theo xác suất nhãn, tính **trên 26.752 bài
có nhãn từ tag** (đúng M4). So với tọa độ tự đặt tay của bản baseline:

| Nhóm | Tâm học được | Tọa độ tự đặt | Khoảng cách lệch | n |
|---|---|---|---|---|
| vui | (0,653; 0,643) | (0,80; 0,75) | 0,182 | 2.465 |
| buon | (0,310; **0,589**) | (0,20; **0,30**) | **0,309** | 1.415 |
| thu_gian | (0,402; 0,478) | (0,60; 0,30) | **0,266** | 8.551 |
| nang_dong | (0,490; 0,795) | (0,60; 0,90) | 0,152 | 4.008 |
| cang_thang | (0,295; 0,876) | (0,25; 0,85) | 0,052 | 10.313 |

**Hai điểm tự đặt sai nhiều nhất:**
- `buon`: tự đặt energy 0,30 nhưng thực tế **0,589** — nhạc mà người nghe gắn tag buồn trong bộ này
  không hề yên tĩnh (post_rock, emo, doom metal).
- `thu_gian`: tự đặt valence 0,60 nhưng thực tế **0,402** — nhạc thư giãn ở đây trầm hơn là tươi.

Chỉ `cang_thang` là đặt gần đúng (lệch 0,052).

Độ lệch chuẩn và tương quan mỗi nhóm (dùng cho ellipse và khoảng cách Mahalanobis ở M5):

| Nhóm | sd valence | sd energy | tương quan |
|---|---|---|---|
| vui | 0,229 | 0,195 | +0,233 |
| buon | 0,222 | 0,265 | +0,322 |
| thu_gian | 0,249 | 0,237 | +0,430 |
| nang_dong | 0,257 | 0,169 | +0,041 |
| cang_thang | 0,195 | 0,163 | −0,004 |

Hình `reports/figures/mood_space.png`: scatter 5 nhóm + ellipse 1σ + mũi tên từ điểm tự đặt cũ
tới tâm học được.

**Lưu ý phương pháp:** nếu tính tâm trên **mọi** bài (kể cả bài mô hình suy ra) thì các tâm bị kéo
về trung bình chung, ví dụ `cang_thang` (0,295; 0,876) → (0,332; 0,861), `nang_dong` valence
0,490 → 0,532. Vì vậy chỉ dùng bài có nhãn từ tag, đúng như `mood_spec.md` M4 yêu cầu.

### 4.3 Thử thanh kéo

`group_probabilities(valence, energy)` trả về phân phối trên 5 nhóm (Gaussian, chuẩn hóa):

| Điểm thanh kéo | Hai nhóm cao nhất |
|---|---|
| (0,8; 0,8) | vui 46%, nang_dong 34% |
| (0,2; 0,3) | thu_gian 52%, buon 41% |
| (0,5; 0,4) | thu_gian 44%, vui 29% |
| (0,3; 0,9) | cang_thang 55%, nang_dong 25% |

Có test kiểm tra: đặt thanh kéo đúng tại tâm của nhóm nào thì nhóm đó phải có xác suất cao nhất.

### 4.4 Test

`tests/test_predict.py` — 8 test: đúng cột hợp đồng, mỗi bài 1 dòng, tổng xác suất = 1,
`mood_pred` khớp argmax, giá trị `mood_source` hợp lệ, trọng số trộn tag/mô hình,
hiệp phương sai có trọng số khớp công thức thường, hiệp phương sai xác định dương,
thanh kéo trả về phân phối hợp lệ. Tổng toàn module: **41 test, tất cả qua**.

## T5. API cho TV2

Chạy ngày 2026-10-23. Code: `src/mood/api.py`. Test: `tests/test_api.py`.

### 5.1 Các hàm đã cài (đúng chữ ký trong `docs/data_contracts.md`)

| Hàm | Trạng thái | Ghi chú |
|---|---|---|
| `load_mood_profiles()` | xong | đọc `track_mood.parquet`, index = `track_id`, có cache |
| `quick_mood(code)` | xong | trả **tâm học được** của nhóm (M4), không phải tọa độ tự đặt |
| `slider_to_mood(valence, energy)` | xong | Gaussian trên 5 nhóm, tổng = 1 |
| `mood_scores(track_ids, valence, energy, lam)` | xong | công thức M5 |
| `user_mood_target(...)` | `NotImplementedError` | làm ở T6 |
| `mood_journey(...)` | `NotImplementedError` | làm ở T7 (tùy chọn) |

### 5.2 Công thức điểm hợp tâm trạng (M5)

```
p_target(s) = Σ_m w_m · P(m | s)        w_m = slider_to_mood(v, e), P(m|s) từ track_mood.parquet
prox(s)     = exp(−d² / 2σ²)            d = khoảng cách Euclid từ (valence, energy) của bài tới (v, e)
mood_score  = λ · p_target + (1 − λ) · prox        λ = 0,6; σ = 0,15 (config)
```

Nhờ vậy thanh kéo vẫn tinh chỉnh liên tục được (phần `prox`), nhưng điểm không chỉ phụ thuộc
2 con số valence/energy (phần `p_target` dùng cả 12 đặc trưng thông qua bộ phân loại).

### 5.3 Tốc độ và kiểm tra

- `mood_scores` cho **20.000 bài: 23 ms** (yêu cầu hợp đồng < 1 giây). Lần gọi sau chỉ 1 ms nhờ
  cache bảng tra `track_id → dòng`.
- `quick_mood` trả đúng tâm trong `mood_centroids.json` (có test so khớp).
- Đặt thanh kéo tại tâm nhóm nào thì nhóm đó có xác suất cao nhất (test cho cả 5 nhóm).
- Bài `cang_thang` có điểm trung bình cao hơn bài `thu_gian` khi thanh kéo ở `cang_thang` (test).
- `lam = 1` chỉ dùng xác suất nhóm, `lam = 0` chỉ dùng độ gần; mặc định là tổ hợp đúng theo λ (test).
- Bài không có trong hồ sơ: trả 0 kèm `UserWarning` (test).

`tests/test_api.py` — 12 test. Tổng toàn module: **53 test, tất cả qua**.

### 5.4 Lưu ý gửi TV2

Thang điểm thực tế của `mood_scores` nằm trong khoảng ~0,02–0,65 chứ không trải hết [0, 1], vì
xác suất nhóm của điểm thanh kéo luôn bị chia cho 5 nhóm. Khi trộn
`score = α × hợp_gu + (1 − α) × hợp_tâm_trạng`, nên chuẩn hóa hai vế (ví dụ min-max trên tập ứng
viên) để α đúng nghĩa là tỷ trọng. Ví dụ dùng đầy đủ đã ghi trong `README.md`.

## T6. Cá nhân hóa và đánh giá trên lịch sử nghe thật

Chạy ngày 2026-10-23. Lệnh: `python -m src.mood.evaluate_mood --n-users 2000`.
Đầu ra: `data/processed/mood_eval.csv`, `mood_eval_summary.csv`, `reports/figures/mood_eval.png`.

### 6.1 Cá nhân hóa mục tiêu tâm trạng (`user_mood_target`)

Lấy các bài user đã nghe có P(nhóm | bài) ≥ `min_prob` = 0,5, tính tâm (valence, energy) có
trọng số theo playcount, rồi co về tâm chung:

```
mu' = (n · mu_user + k · mu_chung) / (n + k),   k = shrinkage_k = 10
```

User không có bài nào thuộc nhóm → trả về đúng tâm chung (có test). Càng ít bài thì tâm cá nhân
càng gần tâm chung (có test).

### 6.2 Giao thức đánh giá

Không có file chia user của TV1 nên tự chia theo workflow chung:

| Tham số | Giá trị |
|---|---|
| Lọc 10/10 | user ≥ 10 bài, bài ≥ 10 người nghe → 7.051.696 dòng, 290.898 user, 24.286 bài |
| playcount | cắt ở 60 |
| User test | 20% (seed 42), lấy mẫu **2.000 user** để chạy cho kịp; 1.269 user đủ điều kiện |
| Che | 30% số bài mỗi user test |
| Trường hợp đánh giá | mỗi (user, nhóm m) có ≥ 3 bài bị che thuộc m → **1.890 trường hợp** |
| Ứng viên | bài bị che + 1.000 bài user chưa nghe lấy ngẫu nhiên (~1.012 bài/lượt) |
| Xếp hạng | theo `api.mood_scores` với điểm mục tiêu của từng cách |

### 6.3 Kết quả — so sánh 3 cách đặt mục tiêu

| Cách đặt mục tiêu | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---|---|---|---|
| Tọa độ tự đặt (baseline) | 0,0365 | 0,0280 | 0,0622 | 0,0382 |
| Tâm học được (M4) | 0,0315 | 0,0239 | 0,0635 | 0,0368 |
| **Tâm cá nhân (M6)** | **0,0390** | **0,0292** | **0,0701** | **0,0417** |

**Trả lời M8 câu 4:** cá nhân hóa **có** cải thiện: Recall@20 +12,6% và NDCG@20 +9,2% so với tọa độ
tự đặt, và là cách duy nhất tốt hơn baseline ở **cả 4 chỉ số**.

**Kết quả không thuận chiều, phải nói rõ:** tâm học được (M4) **không** tốt hơn tọa độ tự đặt một
cách chắc chắn — chỉ hơn ở Recall@20 (+2,0%), còn **thua** ở Recall@10 (−13,7%) và NDCG@20 (−3,7%).

Nhìn theo từng nhóm mới thấy lý do (Recall@20):

| Nhóm | Tọa độ tự đặt | Tâm học được | Tâm cá nhân |
|---|---|---|---|
| vui | **0,1932** | 0,1380 | 0,1575 |
| buon | 0,0011 | **0,0603** | 0,0514 |
| thu_gian | 0,0431 | 0,0613 | **0,0626** |
| nang_dong | **0,1055** | 0,0558 | 0,0768 |
| cang_thang | 0,0446 | 0,0495 | **0,0567** |

- Với `buon`, tọa độ tự đặt gần như **vô dụng** (Recall@20 = 0,0011) vì nó đặt ở (0,20; 0,30) trong
  khi nhạc buồn thật nằm ở (0,310; **0,589**) — tâm học được tăng Recall lên **55 lần**.
- Ngược lại, với `vui` và `nang_dong`, tọa độ tự đặt lại **tốt hơn**. Lý do: điểm tự đặt nằm ở góc
  cực đoan (0,80; 0,75) và (0,60; 0,90), mà thành phần `prox` trong `mood_score` ưu tiên bài gần
  điểm mục tiêu, nên nó kéo lên các bài cực đoan — vốn đúng là các bài được gán nhãn `vui`/
  `nang_dong`. Tâm học được bị co về giữa (0,653; 0,643) nên xếp hạng các bài trung tính lên cao.
- Cá nhân hóa **sửa được một phần** chỗ hụt này: ở `vui` nó kéo Recall từ 0,1380 lên 0,1575,
  ở `nang_dong` từ 0,0558 lên 0,0768, đồng thời vẫn giữ lợi thế ở `buon`, `thu_gian`, `cang_thang`.

Hình: `reports/figures/mood_eval.png`.

### 6.4 Hạn chế của đánh giá này (phải ghi trong báo cáo cuối)

1. **Có yếu tố vòng lặp:** "bài bị che thuộc nhóm m" được xác định bằng chính `mood_pred` của
   module này. Vậy phép đo trả lời câu "xếp hạng có khớp nhãn của chính mình không" trên tập ứng
   viên thật của user, chứ không phải "người dùng có thấy đúng tâm trạng không". Muốn đo đúng nghĩa
   thì cần nhãn tâm trạng do người gán, bộ dữ liệu này không có.
2. Chỉ dùng **2.000/58.179 user test** (1.269 user đủ điều kiện) để chạy kịp; ứng viên là 1.000 bài
   ngẫu nhiên chưa nghe chứ không phải toàn bộ bài chưa nghe — cả hai đều đã ghi rõ ở mục 6.2.
3. Chỉ đánh giá **riêng phần tâm trạng**, không có ALS. Điểm tuyệt đối vì vậy thấp (Recall@20 ≈ 0,07)
   và không so sánh được với số của TV1; chỉ dùng để so 3 cách đặt mục tiêu với nhau.
4. Giá trị tuyệt đối thấp còn vì mỗi lượt có ~1.012 ứng viên mà chỉ 3–10 bài là đúng.

### 6.5 Test

`tests/test_evaluate.py` — 10 test: công thức Recall@k và NDCG@k, chia user lặp lại được, che bài
đúng tỷ lệ và không chồng lấn, shrinkage (ít bài → gần tâm chung), rơi về tâm chung khi user không
có bài thuộc nhóm, trọng số playcount đổi được tâm, bài dưới `min_prob` bị bỏ qua, ba cách đặt mục
tiêu cho ba điểm khác nhau. Tổng toàn module: **64 test, tất cả qua**.

## T7. Hành trình cảm xúc (tùy chọn)

Chạy ngày 2026-10-23. Lệnh: `python -m src.mood.journey [--start buon --end vui --steps 8]`.
Code: `api.mood_journey()` + `src/mood/journey.py`.
Đầu ra: `data/processed/mood_journey_example.csv`, `reports/figures/mood_journey.png`.

### 7.1 Cách làm

`mood_journey(start, end, n_steps)` nội suy tuyến tính `n_steps` điểm trên mặt phẳng
(valence, energy), gồm cả hai đầu mút. Mỗi điểm gọi `api.mood_scores` trên toàn bộ 50.683 bài và
lấy bài điểm cao nhất **chưa được chọn**, nên playlist không lặp bài.

### 7.2 Ví dụ: buồn → vui, 8 bước

Từ tâm học được của `buon` (0,310; 0,589) tới tâm của `vui` (0,653; 0,643):

| Bước | Điểm mục tiêu | Bài được chọn | valence | energy | mood_pred |
|---|---|---|---|---|---|
| 1 | (0,310; 0,589) | Earwig Town — Chairlift | 0,321 | 0,578 | buon |
| 2 | (0,359; 0,597) | Batcat — Mogwai | 0,355 | 0,607 | buon |
| 3 | (0,408; 0,605) | Just Another — Pete Yorn | 0,413 | 0,598 | thu_gian |
| 4 | (0,457; 0,612) | Sweet Lullaby — Deep Forest | 0,461 | 0,615 | thu_gian |
| 5 | (0,506; 0,620) | Light My Fire (Live) — Massive Attack | 0,502 | 0,613 | vui |
| 6 | (0,555; 0,628) | Ring The Alarm — Tenor Saw | 0,554 | 0,645 | vui |
| 7 | (0,604; 0,635) | Pro Lover — Usher | 0,610 | 0,620 | vui |
| 8 | (0,653; 0,643) | Love Me Down — Britney Spears | 0,658 | 0,645 | vui |

valence đi từ 0,321 lên 0,658, energy giữ gần như không đổi (0,578 → 0,645). Nhóm dự đoán chuyển
dần **buon → thu_gian → vui**, không nhảy cóc. Mỗi bài chọn được đều cách điểm mục tiêu dưới 0,02
trên cả hai trục.

Hình `reports/figures/mood_journey.png`: đường mục tiêu (nét đứt) và các bài được chọn (chấm đỏ)
trên nền scatter 5 nhóm.

### 7.3 Vì sao tính năng này đáng kể cho đồ án

Đây là việc mà cách làm cũ (lọc bài theo nhãn tâm trạng có sẵn) không làm được: nó cần một **không
gian tâm trạng liên tục** (M4) và một **hàm chấm điểm liên tục** (M5). Chính là hướng mà giảng viên
góp ý — xử lý đầu vào tâm trạng phải phức tạp hơn, không chỉ chọn bài theo nhãn.

### 7.4 Hạn chế

- Đường đi là **đoạn thẳng** trên (valence, energy). Vùng giữa hai nhóm có thể thưa bài; ở ví dụ
  trên thì không, nhưng với cặp xa nhau như `thu_gian` → `cang_thang` thì các bước giữa có thể rơi
  vào vùng ít bài.
- Chỉ xét độ hợp tâm trạng, **chưa tính độ hợp gu**. Khi ghép với ALS của TV2, nên chấm
  `α × hợp_gu + (1 − α) × mood_score` tại từng điểm thay vì chỉ dùng `mood_score`.

### 7.5 Test

`tests/test_journey.py` — 5 test: đúng hai đầu mút và số điểm, các bước cách đều nhau, valence đơn
điệu và nằm trong khoảng hai đầu, `n_steps < 2` báo lỗi, playlist không trùng bài và mỗi bài cách
điểm mục tiêu dưới 0,15. Tổng toàn module: **69 test, tất cả qua**.

## T8. Tổng kết

Viết ngày 2026-10-23. Mọi số liệu dưới đây lấy từ các file do code sinh ra:
`model_comparison.csv`, `calibration_comparison.csv`, `mood_eval_summary.csv`,
`permutation_importance.csv`, `track_mood_labels.parquet`, `track_mood.parquet`,
`mood_centroids.json`.

### 8.1 Bốn câu hỏi của M8

#### Câu 1. Valence/energy có tách được các nhóm không?

**Có, nhưng valence là đặc trưng tách nhóm YẾU NHẤT trong năm đặc trưng được kiểm định.**

Kruskal–Wallis trên 26.752 bài có nhãn (tất cả p < 1e-300):

| Đặc trưng | H |
|---|---|
| acousticness | 13.660 |
| energy | 12.369 |
| danceability | 8.868 |
| loudness | 8.050 |
| **valence** | **4.819** |

Boxplot (`mood_valence_energy_box.png`) cho thấy `buon` và `cang_thang` gần như trùng nhau về
valence (trung vị 0,228 vs 0,271) và chỉ tách được nhờ energy (0,624 vs 0,933). Nói cách khác,
hai trục của thanh kéo ban đầu **không đủ** để phân biệt 5 nhóm.

#### Câu 2. Thêm đặc trưng khác cải thiện bao nhiêu so với chỉ valence/energy?

**Cải thiện rõ: macro-F1 +17% với cùng một thuật toán, +29% nếu đổi sang LightGBM.**

| Mô hình | Đặc trưng | macro-F1 | So với `logreg_ve` |
|---|---|---|---|
| logreg_ve | 2 | 0,4414 | — |
| logreg_full | 12 | 0,5159 | **+16,9%** |
| gbm_full | 12 | 0,5711 | **+29,4%** |

Nhóm hưởng lợi nhiều nhất là `nang_dong` (F1 0,324 → 0,505 → 0,588) và `thu_gian`
(0,544 → 0,577 → 0,702).

Permutation importance giải thích vì sao: **acousticness (0,1011) và danceability (0,0994) quan
trọng hơn energy (0,0608) và gấp 2,5 lần valence (0,0399)**. `key` và `mode` gần như vô dụng.
Khớp đúng với kết quả Kruskal–Wallis ở câu 1.

#### Câu 3. Bộ phân loại so với baseline tọa độ tự đặt?

**Hơn 57% về macro-F1.**

| Mô hình | macro-F1 | accuracy | log-loss |
|---|---|---|---|
| baseline_ve (tọa độ tự đặt) | 0,3628 | 0,4399 | 2,3744 |
| **gbm_full** | **0,5711** | **0,6771** | **0,8585** |
| Chênh lệch | **+57,4%** | +53,9% | −63,8% |

Đánh giá bằng GroupKFold 5 fold theo `artist` nên cùng một nghệ sĩ không nằm ở cả train và test.

#### Câu 4. Cá nhân hóa có cải thiện đánh giá trên lịch sử nghe thật không?

**Có. Đây là cách duy nhất tốt hơn baseline ở cả 4 chỉ số.** 1.890 cặp (user, nhóm) từ 1.269 user test:

| Cách đặt mục tiêu | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---|---|---|---|
| Tọa độ tự đặt | 0,0365 | 0,0280 | 0,0622 | 0,0382 |
| Tâm học được (M4) | 0,0315 | 0,0239 | 0,0635 | 0,0368 |
| **Tâm cá nhân (M6)** | **0,0390** | **0,0292** | **0,0701** | **0,0417** |
| Cá nhân vs tự đặt | +6,8% | +4,3% | **+12,6%** | **+9,2%** |

**Nói thẳng phần không thuận chiều:** riêng tâm học được (M4) **chưa chắc** tốt hơn tọa độ tự đặt —
hơn ở Recall@20 (+2,0%) nhưng thua ở Recall@10 (−13,7%) và NDCG@20 (−3,7%). Nguyên nhân đã phân
tích ở mục 6.3: với `vui` và `nang_dong`, điểm tự đặt nằm ở góc cực đoan nên thành phần `prox` của
`mood_score` kéo đúng các bài cực đoan lên; còn tâm học được bị co về giữa. Bù lại, với `buon` thì
tọa độ tự đặt gần như vô dụng (Recall@20 = 0,0011 so với 0,0603 — **gấp 55 lần**).

### 8.2 Năm cải tiến so với bản gốc — đã làm được gì

| # | Cải tiến | Trạng thái | Bằng chứng |
|---|---|---|---|
| 1 | Nhãn mềm từ tag người nghe, có xử lý mâu thuẫn | xong | 37.801 bài có nhãn mềm, 26.752 nhãn cứng, 10.089 `conflict` được loại |
| 2 | Bộ phân loại suy ra tâm trạng cho bài không có tag | xong | macro-F1 0,5711 vs 0,3628 của baseline; 23.931 bài được mô hình gán hồ sơ |
| 3 | Vị trí nhóm trên thanh kéo học từ dữ liệu | xong | `mood_centroids.json`; `buon` lệch 0,309 và `thu_gian` lệch 0,266 so với điểm tự đặt |
| 4 | Điểm hợp tâm trạng kết hợp xác suất nhóm và độ gần | xong | `api.mood_scores`, 20.000 bài trong 23 ms |
| 5 | Cá nhân hóa mục tiêu theo từng người, đánh giá dữ liệu thật | xong | Recall@20 +12,6% trên 1.890 cặp (user, nhóm) |
| (+) | Hành trình cảm xúc (tùy chọn M7) | xong | playlist buồn → vui 8 bước, nhóm chuyển buon → thu_gian → vui |

### 8.3 Hạn chế — phải nói rõ khi bảo vệ

1. **Nhãn là nhãn yếu, không phải nhãn người gán.** Không ai trong bộ dữ liệu nói "bài này vui";
   nhãn được suy ra từ tag Last.fm theo bảng trọng số do nhóm tự đặt.
2. **Chỉ có 5/100 tag là tag cảm xúc thật** (chillout, mellow, beautiful, love, chill) và cả 5 nằm
   cùng một vùng valence–energy. 25 tag ứng viên như happy, sad, angry, energetic **không tồn tại**
   trong dữ liệu.
3. **Ba trong năm nhóm (`vui`, `nang_dong`, `cang_thang`) không có một tag cảm xúc nào** — nhãn của
   chúng hoàn toàn dựa trên **tag thể loại** (reggae/ska/funk → vui; house/techno/dance → nang_dong;
   metalcore/death_metal/hardcore → cang_thang) với trọng số 0,5. Đây là giả định mạnh nhất của
   toàn module: *thể loại nói lên cảm xúc*. Nhóm `buon` cũng chỉ còn `beautiful` là tag cảm xúc
   trực tiếp, phần lớn dựa vào post_rock/emo/doom/gothic.
4. **Không nhóm nào dùng "nhãn luật" từ valence/energy.** Phương án này đã bị loại ở T1 vì sẽ tạo
   vòng lặp logic với bộ phân loại (nhãn sinh từ valence rồi lại học từ valence).
5. **Nhóm `buon` yếu nhất:** F1 chỉ 0,2385, bị nhầm sang `thu_gian` (0,37) và `cang_thang` (0,22).
   Vì nguồn nhãn của nó là nhạc buồn nhưng **mạnh** (post_rock, emo, doom metal) nên nó không nằm
   ở góc "trầm và yên" như trực giác. Bộ tag này không có tag nào cho nhạc buồn nhẹ.
6. **Mất cân bằng nhóm 7,3:1** (cang_thang 10.319 vs buon 1.413) → phải dùng `class_weight` cân bằng
   và đọc macro-F1 thay vì accuracy.
7. **1.367 bài (2,70%) nghi ghép nhầm phiên bản** (có tag `cover`, tên chứa live/remix/remaster,
   `spotify_id` trùng, thời lượng bất thường) đã bị loại khỏi tập huấn luyện. Hai bài EDA nghi ngờ
   đều được xác nhận: Katherine Jenkins "Bring Me To Life" (energy 0,563) vs bản gốc Evanescence
   (0,930). Ngoài ra 2.752 bài lệch tag thập niên và 1.614 bài trùng tên với nghệ sĩ khác chỉ được
   ghi nhận, không loại, vì tín hiệu yếu.
8. **"50 bài phổ biến" dùng proxy số tag/bài** ở T1 vì lúc đó chưa có lịch sử nghe. Nay đã có
   `User Listening History.csv`, nên rà lại theo số người nghe thật sẽ chính xác hơn.
9. **Đánh giá T6 có yếu tố vòng lặp:** "bài bị che thuộc nhóm m" được xác định bằng chính
   `mood_pred` của module. Phép đo trả lời "xếp hạng có khớp nhãn của chính mình không", không phải
   "người dùng có thấy đúng tâm trạng không". Ngoài ra chỉ dùng 2.000/58.179 user test và 1.000 bài
   ứng viên ngẫu nhiên mỗi lượt.
10. **Không hiệu chỉnh xác suất**, khác đặc tả M3: cả isotonic và sigmoid đều làm F1 nhóm `buon` rơi
    từ 0,2385 xuống 0,007/0,006 trong khi ECE gốc đã tốt (0,0407) và còn bị hiệu chỉnh làm xấu đi.
11. **Thang điểm `mood_scores` chỉ chạy trong ~0,02–0,65**, không trải hết [0, 1]. TV2 nên chuẩn hóa
    hai vế trước khi trộn `α × hợp_gu + (1 − α) × hợp_tâm_trạng`.
12. **Chưa ghép với ALS.** Mọi con số ở T6 chỉ đo riêng phần tâm trạng nên giá trị tuyệt đối thấp
    (Recall@20 ≈ 0,07 trên ~1.012 ứng viên) và không so sánh trực tiếp được với số của TV1.

### 8.4 File bàn giao

| File | Cho ai | Nội dung |
|---|---|---|
| `data/processed/track_mood.parquet` | **TV2** | hồ sơ tâm trạng 50.683 bài: `p_*`, `mood_pred`, `mood_confidence`, `mood_source`, `valence`, `energy` |
| `data/processed/mood_centroids.json` | **TV2** | tâm + hiệp phương sai 5 nhóm (vị trí nút chọn nhanh và thanh kéo) |
| `src/mood/api.py` | **TV2** | `quick_mood`, `slider_to_mood`, `mood_scores`, `user_mood_target`, `mood_journey` |
| `data/processed/track_mood_labels.parquet` | **TV1** | nhãn từ tag + `label_status`, để đánh giá có điều kiện tâm trạng |
| `data/processed/tag_audit.csv`, `suspect_tracks.csv` | TV1 | kiểm kê tag, danh sách bài nghi ghép nhầm phiên bản |
| `models/mood_classifier.joblib` | nội bộ | LightGBM + danh sách cột đặc trưng |
| `reports/figures/*.png` | báo cáo | 8 hình: top tag, boxplot valence/energy, ma trận nhầm lẫn, reliability, permutation importance, không gian tâm trạng, so sánh 3 cách đặt mục tiêu, hành trình cảm xúc |

### 8.5 Chạy lại toàn bộ

```bash
python -m src.mood.io                       # kiểm tra dữ liệu
python -m src.mood.tag_audit                # T1
python -m src.mood.labeling                 # T2
python -m src.mood.train_classifier         # T3 (vài phút)
python -m src.mood.predict                  # T4
python -m src.mood.mood_space               # T4
python -m src.mood.evaluate_mood --n-users 2000   # T6 (vài phút)
python -m src.mood.journey                  # T7
python -m pytest tests -q                   # 69 test (Phần I)
```

Mọi bước dùng `random_state = 42`. Thêm `--sample` để chạy nhanh trên 5.000 bài.

---

# Phần II — Ghép với mô hình gu và ba cải tiến từ tài liệu (A2, A1, A3)

Ba việc này đến từ hai bài tham khảo đã chốt trong họp:

- **HDBN** — Jing và cộng sự, *Emotion-aware Personalized Music Recommendation with a
  Heterogeneity-aware Deep Bayesian Network*, ACM TOIS 2025 (arXiv:2406.14090).
- **Read Between the Tracks** — Hausberger, Jósár & Schedl, *Exploring LLM-driven
  Intent-based Music Recommendations*, NLP4MusA 2026, tr. 44–50.

## 9. Tầng nối sang mô hình gu của TV2

`src/mood/taste.py` nạp nguyên code trong `taste-model/src/` bằng `importlib` dưới tên
gói `taste_src` (không trùng gói `src` của mood-model), nên **không cài lại ALS** và
không sửa một dòng nào của TV2. Dùng lại: `ALS`, `Popularity`, `combine_scores`.

| | |
|---|---|
| Cấu hình ALS | mặc định của TV2: `factors=32`, `regularization=0.1`, `als_alpha=10`, `iterations=15`, `weighting="log"`, backend `implicit` (chưa grid-search — việc đó thuộc TV2) |
| Cohort | **50.000 user** = 20.000 user test + 30.000 user khác (seed 42) |
| Dữ liệu huấn luyện | **1.065.240** tương tác, 23.348 bài |
| Chống rò rỉ | **145.227** tương tác bị che đã bị loại khỏi dữ liệu huấn luyện ALS |

Bộ chia thấy được/bị che được chuyển sang hàm mới `evaluate_mood.hidden_for_user()`:
gieo hạt từ `user_id` thay vì dùng một bộ sinh ngẫu nhiên chung, nên bất kỳ module nào
cũng tái lập đúng một bộ chia. Đây là điều kiện bắt buộc để phần bị che thật sự nằm
ngoài dữ liệu huấn luyện gu. Luồng ngẫu nhiên của `evaluate_mood.main()` **không đổi**,
nên mọi số của T6 ở Phần I tái lập đúng nguyên văn.

Tuân theo `taste-model/docs/EVALUATION_PROTOCOL.md`: mục 3 (loại bài heldout ngoài
catalogue mô hình, báo số lượng) và mục 5 (chỉ dùng phần thấy được để cá nhân hóa mục
tiêu tâm trạng; nhãn do chính module mood sinh nên đây là **kiểm tra nhất quán với nhãn
yếu**, không phải xác nhận cảm nhận người dùng thật).

## 10. A2 — Đánh giá bốn tập: mô hình tâm trạng thêm được gì so với mô hình gu?

### 10.1 Vấn đề

Recall@20 ở T6 trộn hai thứ vào một số: không biết điểm cao là nhờ gu hay nhờ tâm trạng.
Khung bốn tập của bài NLP4MusA tách được hai thứ đó.

### 10.2 Thiết kế

Với mỗi cặp (user u, nhóm m), kho nhạc chia thành bốn tập không giao nhau; lấy N bài mỗi
tập, trộn lẫn thành 4N ứng viên, rồi đo **NDCG@N riêng cho từng tập**:

| Tập | Nghĩa | Nguồn |
|---|---|---|
| `set1` | đúng gu **và** đúng tâm trạng | bài bị che, `mood_pred == m` |
| `set2` | đúng gu, sai tâm trạng | bài bị che, `mood_pred != m` |
| `set3` | đúng tâm trạng, chưa từng nghe | kho nhạc, `mood_pred == m` |
| `set4` | sai cả hai | kho nhạc, `mood_pred != m` |

Khác bài gốc ở một chỗ quan trọng: "đúng gu" ở đây là **phần bị che**, không phải lịch sử
đã biết. Nếu dùng lịch sử đã biết thì ALS đã học chính các bài đó nên set1/set2 bị rò rỉ.

Bảy bộ xếp hạng trên cùng tập ứng viên: `random`, `pop` (Popularity của TV2), `taste`
(ALS), `mood` (`mood_scores` với mục tiêu cá nhân hóa), và `hybrid30/50/70`
(`combine_scores` của TV2 với `taste_weight` = 0,3 / 0,5 / 0,7).

### 10.3 Kết quả — N = 5 bài mỗi tập (20 ứng viên, 4.096 cặp user-nhóm)

| Bộ xếp hạng | set1 gu+tâm trạng | set2 chỉ gu | set3 chỉ tâm trạng | set4 không gì |
|---|---|---|---|---|
| ngẫu nhiên | 0,2465 | 0,2503 | 0,2553 | 0,2480 |
| phổ biến (Pop) | 0,4263 | 0,4563 | 0,0600 | 0,0574 |
| chỉ gu (ALS) | 0,4961 | 0,4691 | 0,0202 | 0,0146 |
| chỉ tâm trạng | 0,4377 | 0,0863 | 0,4047 | 0,0713 |
| ghép 0,3 gu | 0,5550 | 0,1297 | 0,2710 | 0,0443 |
| **ghép 0,5 gu** | **0,6146** | 0,2387 | 0,1274 | 0,0193 |
| ghép 0,7 gu | 0,5784 | 0,3635 | 0,0476 | 0,0105 |

### 10.4 Bốn điều đọc được

1. **Phép đo không lệch.** `random` cho ≈ 0,25 ở cả bốn tập, đúng kỳ vọng với 5/20 ứng
   viên là bài liên quan.
2. **Mô hình gu không phân biệt được tâm trạng.** Chênh lệch set1 − set2 của `taste` chỉ
   **+0,0270**; của `pop` còn âm (−0,0300). Cả hai nâng bài người dùng thích, bất kể bài
   đó thuộc tâm trạng nào. Đây là bằng chứng số cho thấy mô-đun tâm trạng **không trùng
   lặp** với mô-đun gu.
3. **Mô hình tâm trạng không phân biệt được người dùng.** `mood` đạt 0,4377 ở set1 và
   0,4047 ở set3 — gần như bằng nhau, tức là nó không biết bài nào thuộc gu ai.
4. **Ghép lại thắng cả hai, và 0,5 là điểm tốt nhất.** `hybrid50` đạt **0,6146** ở set1,
   hơn `taste` **+23,9%** và hơn `mood` +40,4%. Mức phân biệt tâm trạng tăng từ
   **+0,0270** (chỉ gu) lên **+0,3759**.

### 10.5 Kiểm tra độ bền theo cỡ tập

| N bài/tập | số cặp | set1 chỉ gu | set1 ghép 0,5 | mức tăng | phân biệt tâm trạng (ghép 0,5) |
|---|---|---|---|---|---|
| 3 | 12.697 | 0,4907 | 0,6035 | +23,0% | +0,3518 |
| 5 | 4.096 | 0,4961 | 0,6146 | +23,9% | +0,3759 |
| **10** (đúng cỡ bài gốc) | 584 | 0,5040 | 0,6337 | **+25,7%** | +0,4198 |

Kết luận không phụ thuộc cỡ tập; ở N = 10 — đúng thiết kế của bài gốc — hiệu ứng mạnh
nhất. Số bài bị che bị loại vì ngoài catalogue ALS (cold-item): 222.

Hạn chế phải nói rõ: nhãn `mood_pred` dùng để định nghĩa set1/set3 do **chính module này**
sinh ra, nên A2 đo mức **nhất quán nội bộ** giữa hồ sơ tâm trạng và hành vi nghe thật,
chứ không chứng minh người dùng cảm nhận đúng nhóm đó. Muốn kết luận mạnh hơn thì cần
nhãn độc lập hoặc khảo sát mù (mục 6 trong giao thức của TV2).

## 11. A1 — Gán nhãn tag bằng embedding thay cho bảng trọng số chỉnh tay

### 11.1 Thiết kế

Làm theo bài NLP4MusA: nhúng **tên tag** và **tên nhóm tâm trạng**, tính cosine, rồi
**chuẩn hóa z-score theo từng nhóm trên toàn bộ tag** trước khi chọn giá trị cao nhất.
Bước z-score là chỗ đáng giá nhất: nó xóa độ lệch thang đo giữa các nhóm — đúng chỗ mà
bảng trọng số tay đã phải chỉnh bù cho nhóm `buon` ở T2.

Giữ tag có `max z >= 1.0`; `argmax` đưa tag về một nhóm (như bài gốc), `soft` chia
`relu(z)` chuẩn hóa cho nhiều nhóm.

### 11.2 Chọn mô hình nhúng — và một thước đo khách quan

Mô hình đúng của bài gốc, `google/embeddinggemma-300m`, **bị khóa trên HuggingFace**
(`GatedRepoError` 401). Ba mô hình mở được chấm bằng một thước không phụ thuộc bảng tay:
mỗi tag có một vị trí (valence, energy) trung bình thật trong dữ liệu, nhóm "đúng" là tâm
học được gần nhất theo khoảng cách Mahalanobis.

| Mô hình nhúng | khớp vị trí âm thanh (tag giữ lại) |
|---|---|
| all-MiniLM-L6-v2 (22M) | 32,4% |
| all-mpnet-base-v2 (110M) | 35,1% |
| **BAAI/bge-base-en-v1.5** (109M) | **50,0%** |
| google/embeddinggemma-300m | không tải được (repo bị khóa) |
| **bảng trọng số tay (T2)** | **83,0%** |

### 11.3 Kết quả (bge-base-en-v1.5, z ≥ 1.0, toàn bộ 50.683 bài)

| Cách gán nhãn | số tag | bài có nhãn | khớp vị trí âm thanh | trùng nhãn với bảng tay | macro-F1 (T3) |
|---|---|---|---|---|---|
| **bảng tay (T2)** | 53 | 26.752 | **83,0%** | — | **0,5711** |
| embedding argmax | 44 | **28.999** | 50,0% | 74,8% | 0,5327 |
| embedding soft | 44 | 11.555 | 50,0% | 86,2% | không huấn luyện được |
| bảng tay + embedding cho tag chưa phủ | **65** | **30.362** | 72,3% | 98,9% | 0,5562 |

### 11.4 Kết luận: **giữ bảng trọng số tay**

Embedding **tăng độ phủ** (+2.247 bài với argmax, +3.610 bài với bản ghép) nhưng **giảm
chất lượng nhãn** (83,0% → 50,0% / 72,3%) và **giảm macro-F1** (0,5711 → 0,5327 / 0,5562).
Không biến thể nào thắng.

Lý do thất bại đọc được ngay từ bảng z-score: bài gốc nhúng **tên playlist** — nhiều từ,
mang nghĩa mô tả ("chill vibes") — còn ta chỉ có **tên thể loại trần**. Mô hình văn bản
đọc theo mặt chữ, nên:

- `buon` (buồn) khớp nhất với **`melodic_death_metal`** (chữ "death" ≈ bi thương, "melodic"
  ≈ dễ nghe), trong khi âm thanh thật của tag này là valence 0,27 / energy 0,90;
- `vui` khớp nhất với **`mellow`** (val 0,36 / ene 0,44) — tức là nhầm sang nhóm thư giãn;
- phần làm đúng lại là chỗ tên gọi trùng nghĩa âm nhạc: `thu_gian` ← mellow, chill,
  downtempo, ambient; `cang_thang` ← grindcore, hardcore, dark_ambient.

Biến thể `soft` sụp hẳn: trải trọng số ra nhiều nhóm làm 25.822 bài thành `conflict` và
nhóm `thu_gian` còn **0 bài**, không huấn luyện nổi bộ phân loại.

**Giá trị còn lại của A1:** nó biến "bảng trọng số do nhóm tự đặt" từ một điểm yếu bị
chất vấn thành một **lựa chọn đã kiểm chứng** — nhóm đã thử cách tự động, đo bằng hai
thước độc lập, và có số để nói vì sao cách tay tốt hơn trên dữ liệu này.

## 12. A3 — "Nói ngầm" (bài ví dụ) so với "nói tỏ" (thanh kéo)

### 12.1 Thiết kế

Bài NLP4MusA phát hiện: đưa 3 bài người dùng từng nghe cho ý định đó **tốt hơn** gọi tên
ý định (Mistral-7B: NDCG 0,37 so với 0,29). Ba kiểu prompt của họ khớp một-đối-một với
các cách đặt mục tiêu ở T6:

| Bài gốc | Mục tiêu tương ứng |
|---|---|
| tỏ (chỉ tên ý định) | `manual`, `learned` — chỉ dùng tọa độ nhóm |
| cả hai (tên + ví dụ) | `personal` — tọa độ nhóm **+** lịch sử người dùng |
| **ngầm (chỉ bài ví dụ)** | **`seed3`** — tâm của đúng N bài ví dụ, **không** co về tâm chung |

`seed3` khác `personal` ở hai điểm: chỉ dùng N bài thay vì toàn bộ lịch sử, và không dùng
tới tọa độ nhóm. Chọn bài tái lập được theo hash `user_id|mood` nên không ăn vào bộ sinh
ngẫu nhiên chung — ba mục tiêu cũ giữ nguyên số đã công bố.

### 12.2 Kết quả (1.269 user test, 1.890 cặp user-nhóm)

| Mục tiêu | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | so với tọa độ tự đặt |
|---|---|---|---|---|---|
| `manual` tọa độ tự đặt | 0,0365 | 0,0280 | 0,0622 | 0,0382 | — |
| `learned` tâm học được | 0,0315 | 0,0239 | 0,0635 | 0,0368 | +2,0% |
| **`personal` tâm cá nhân** | **0,0390** | **0,0292** | **0,0701** | **0,0417** | **+12,6%** |
| `seed3` 3 bài ví dụ | 0,0348 | 0,0255 | 0,0623 | 0,0367 | +0,1% |
| `seed3` 5 bài ví dụ | 0,0387 | 0,0285 | 0,0687 | 0,0407 | +10,4% |
| `seed3` 10 bài ví dụ | 0,0340 | 0,0258 | 0,0659 | 0,0386 | +5,9% |

### 12.3 Kết luận: **phát hiện của bài gốc không chuyển sang hệ này**

Với đúng 3 bài như bài gốc, `seed3` **ngang bằng** tọa độ tự đặt (+0,1%) và thua rõ
`personal` (0,0623 so với 0,0701). Tăng lên 5 bài thì tốt hơn (+10,4%) nhưng vẫn không
vượt `personal`; lên 10 bài lại tụt vì tỷ lệ phải lui về tâm chung tăng vọt:

| Số bài ví dụ | tỷ lệ cặp không đủ bài, phải lui về tâm chung |
|---|---|
| 3 | 14,4% (273/1.890) |
| 5 | 32,1% (607/1.890) |
| 10 | 69,2% (1.308/1.890) |

Trung vị số bài thuộc một nhóm trong phần thấy được chỉ là **7 bài**, nên 10 bài ví dụ là
quá nhiều so với dữ liệu.

**Giải thích:** LLM đọc được **ngữ nghĩa** từ tên nghệ sĩ và tên bài (thể loại, thời kỳ,
ngôn ngữ, phong cách sản xuất), nên 3 bài ví dụ mang rất nhiều thông tin. Một điểm
(valence, energy) lấy trung bình 3 bài thì chỉ còn hai con số và nhiễu rất lớn. Bước co về
tâm chung của `personal` chính là phần chống nhiễu đó — bỏ nó đi là mất.

**Hệ quả thiết kế:** giữ thanh kéo, và cách tốt nhất vẫn là `personal` (tọa độ nhóm + lịch
sử nghe, tức kiểu "cả hai" của bài gốc). Nếu làm giao diện chọn bài hạt giống thì nên lấy
**5 bài trở lên và vẫn co về tâm chung**, chứ không thay thế thanh kéo.

## 13. File mới và cách chạy lại Phần II

| File | Nội dung |
|---|---|
| `src/mood/taste.py` | tầng nối sang taste-model, huấn luyện ALS trên cohort đánh giá |
| `src/mood/evaluate_four_sets.py` | A2 — khung bốn tập |
| `src/mood/labeling_embed.py` | A1 — nhãn tag bằng embedding + z-score |
| `src/mood/evaluate_mood.py` | A3 — thêm mục tiêu `seed3` và hàm `hidden_for_user` |
| `data/processed/taste_als.npz`, `taste_cohort.csv` | mô hình gu đã huấn luyện + cohort |
| `data/processed/four_sets_summary_n{3,5,10}.csv`, `four_sets_eval_n*.csv` | kết quả A2 |
| `data/processed/labeling_compare.csv`, `tag_weights_embed.csv`, `tag_group_similarity.csv`, `tag_audio_position.csv` | kết quả A1 |
| `data/processed/mood_eval_summary.csv`, `mood_eval_summary_seeds{5,10}.csv` | kết quả A3 |
| `reports/figures/four_sets_n{3,5,10}.png`, `mood_eval.png` | hình Phần II |
| `tests/test_integration.py` | 26 test cho A1/A2/A3, tầng nối, bộ chia chính thức và kiểm tra rò rỉ đầu-cuối |

```bash
python -m src.mood.taste                            # huấn luyện gu trên cohort (~2 phút)
python -m src.mood.evaluate_four_sets --n-per-set 5  # A2
python -m src.mood.labeling_embed --refresh-embed    # A1 (lần đầu: tải mô hình nhúng)
python -m src.mood.labeling_embed                    # A1 (đọc cache, không nạp torch)
python -m src.mood.evaluate_mood --n-seeds 3         # A3
python -m pytest tests -q                            # 95 test
```

Lưu ý kỹ thuật: `torch` (do sentence-transformers nạp) và LightGBM mang hai bản OpenMP
khác nhau, nạp cả hai trong một tiến trình gây segfault trên macOS (đã gặp: exit 139).
Vì vậy `labeling_embed.py` cache bảng cosine ra `tag_group_similarity.csv` và các lần
chạy sau không nạp lại mô hình nhúng.

---

# Phần III — Chạy lại trên bộ dữ liệu chính thức của nhóm (2026-11-02)

## 14. Vì sao phải chạy lại

TV1 bàn giao bộ chia train/test chung ở `project/data/` (xem `data_preprocessing_report.md`).
So với dữ liệu Phần II, ba thứ đổi:

| | Phần II (bộ chia tự làm) | Phần III (bộ chia chính thức) |
|---|---|---|
| Lọc | 10/10 một lượt | **10-core lặp** đến khi ổn định |
| Kho nhạc | 24.286 bài | **22.557 bài** (bỏ 1.729 bài dưới 10 người nghe) |
| Phần bị che | tự gieo hạt theo `user_id`, **30%** mỗi user | tập test chính thức, **~20%** mỗi user (trung vị 4 bài) |
| Tương tác | 7.051.696 | 5.627.321 train + 1.400.194 test |
| Cột mới | — | `engagement_score`, `engagement_level` |

Hai task phụ thuộc lịch sử nghe (**A2**, **A3**) phải chạy lại. **A1** chỉ dùng `tags` và
đặc trưng âm thanh của `Music_Info.csv` — file này không đổi (20/10) — nên về nguyên tắc
không bị ảnh hưởng; vẫn chạy lại trên catalogue hẹp hơn để xác nhận bằng số chứ không
bằng lập luận.

### 14.1 Tầng đọc bộ chia mới — `src/mood/splits.py`

Đọc `project/data/{train,test}.csv` (930 MB + 222 MB), cache ra parquet, và cung cấp:

| Hàm | Việc |
|---|---|
| `official_split(track_filter)` | trả `(train, test)` đã lọc theo kho bài |
| `user_views(train, test, users, weight_col)` | mỗi user: (bài thấy được, trọng số, bài bị che) |
| `hidden_map(test, users)` | phần bị che của từng user |
| `als_frame(frame, weight_col)` | khung dữ liệu cho ALS của TV2, đổi được cột trọng số |

Kiểm tra bộ chia của TV1: **0 cặp (user, bài) trùng giữa train và test**, **0 bài test nằm
ngoài train**. Nhờ vậy chống rò rỉ trở thành hiển nhiên — không cần `hidden_for_user` nữa,
tuy hàm đó vẫn giữ để tái lập các số Phần II (`--split local`).

### 14.2 Mô hình gu mới

| | Phần II | Phần III |
|---|---|---|
| Dữ liệu huấn luyện | 1.065.240 tương tác, cohort 50.000 user | **5.627.321 tương tác, toàn bộ 289.572 user** |
| Kích thước ALS | 50.000 × 23.348 | **289.572 × 22.557** |
| Rò rỉ | 0 (sau khi loại 145.227 tương tác bị che) | **0 (theo thiết kế bộ chia)** |
| Bài heldout ngoài catalogue (cold-item) | 222 | **0** |

## 15. A2 trên bộ chia chính thức — kết luận **không đổi**

| N | bộ chia | số cặp | set1 chỉ gu | set1 ghép 0,5 | mức tăng | phân biệt tâm trạng (ghép 0,5) |
|---|---|---|---|---|---|---|
| 3 | cũ | 12.697 | 0,4907 | 0,6035 | +23,0% | +0,3518 |
| 3 | **chính thức** | **18.001** | 0,4923 | 0,6106 | **+24,0%** | +0,3594 |
| 5 | cũ | 4.096 | 0,4961 | 0,6146 | +23,9% | +0,3759 |
| 5 | **chính thức** | **4.515** | 0,5048 | 0,6228 | **+23,4%** | +0,3906 |
| 10 | cũ | 584 | 0,5040 | 0,6337 | +25,7% | +0,4198 |
| 10 | **chính thức** | **449** | 0,5103 | 0,6372 | **+24,9%** | +0,4307 |

Bảng đầy đủ ở N = 3 (18.001 cặp user-nhóm, 10.974 user):

| Bộ xếp hạng | set1 gu+tâm trạng | set2 chỉ gu | set3 chỉ tâm trạng | set4 không gì |
|---|---|---|---|---|
| ngẫu nhiên | 0,2508 | 0,2475 | 0,2490 | 0,2527 |
| phổ biến (Pop) | 0,4320 | 0,4487 | 0,0603 | 0,0589 |
| chỉ gu (ALS) | 0,4923 | 0,4706 | 0,0217 | 0,0154 |
| chỉ tâm trạng | 0,4342 | 0,0901 | 0,4037 | 0,0720 |
| ghép 0,3 gu | 0,5527 | 0,1326 | 0,2688 | 0,0459 |
| **ghép 0,5 gu** | **0,6106** | 0,2512 | 0,1175 | 0,0207 |
| ghép 0,7 gu | 0,5669 | 0,3753 | 0,0447 | 0,0130 |

Cả bốn phát hiện của Phần II giữ nguyên: `random` ≈ 0,25 ở cả bốn tập; mô hình gu gần như
mù tâm trạng (+0,0217); mô hình tâm trạng mù người dùng (0,4342 so với 0,4037); ghép 0,5
thắng cả 0,3 và 0,7. **Kết luận A2 bền vững trước thay đổi dữ liệu.**

## 16. A2 với `engagement_score` — cột mới của TV1 **cải thiện kết quả**

`engagement_score` ∈ [0, 1] gộp bốn tín hiệu (playcount log, percentile trong lịch sử user,
playcount tương đối so với trung bình bài, bằng chứng nghệ sĩ). Đưa vào ALS bằng cách nhân
thang 60 (bằng `playcount_cap`) rồi dùng đúng `_confidence` của TV2 (`1 + alpha·log1p(count)`),
không sửa code TV2.

| N=3, 18.001 cặp | set1 chỉ gu | set1 ghép 0,5 | mức tăng | phân biệt tâm trạng |
|---|---|---|---|---|
| trọng số `playcount` | 0,4923 | 0,6106 | +24,0% | +0,3594 |
| **trọng số `engagement_score`** | 0,4987 | **0,6505** | **+30,4%** | **+0,4133** |

Tương quan giữa `engagement_score` và `log1p(playcount)` là 0,872 — tức ba tín hiệu còn lại
chỉ đóng góp phần nhỏ, nhưng phần nhỏ đó đủ nâng set1 thêm **+0,0399** và nâng mức phân biệt
tâm trạng thêm **+0,0539**.

**Đề xuất gửi TV2:** thử `engagement_score` làm trọng số tin cậy trong grid-search của mô
hình gu. Đây mới là một quan sát ở một cấu hình, chưa phải kết luận đã tối ưu.

## 17. A1 trên catalogue chính thức — kết luận **không đổi**

Chạy lại toàn bộ so sánh trên 22.557 bài của catalogue 10-core (thay vì 50.683 bài):

| Cách gán nhãn | số tag | bài có nhãn | khớp vị trí âm thanh | trùng nhãn với bảng tay | macro-F1 (T3) |
|---|---|---|---|---|---|
| **bảng tay (T2)** | 53 | 12.245 | **84,9%** | — | **0,5656** |
| embedding argmax | 44 | 13.113 | 52,3% | 75,9% | 0,5176 |
| embedding soft | 44 | 5.166 | 52,3% | 85,6% | không huấn luyện được |
| bảng tay + embedding cho tag chưa phủ | 65 | **13.694** | 73,8% | 98,8% | 0,5541 |

So với Phần II (83,0% / 0,5711 cho bảng tay; 50,0% / 0,5327 cho embedding argmax): thứ hạng
y nguyên, khoảng cách y nguyên. Biến thể `soft` vẫn sụp vì nhóm `thu_gian` còn 0 bài.
**Giữ bảng trọng số tay.**

## 18. A3 trên bộ chia chính thức — con số **đổi**, kết luận mạnh hơn

### 18.1 Kết quả

| Mục tiêu | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 | so với tọa độ tự đặt |
|---|---|---|---|---|---|
| `manual` tọa độ tự đặt | 0,0397 | 0,0284 | 0,0692 | 0,0399 | — |
| `learned` tâm học được | 0,0357 | 0,0257 | 0,0659 | 0,0376 | **−4,7%** |
| **`personal` tâm cá nhân** | 0,0396 | 0,0290 | **0,0711** | **0,0413** | **+2,7%** |
| `seed3` 3 bài ví dụ | 0,0330 | 0,0229 | 0,0630 | 0,0347 | **−9,0%** |

10.094 cặp (user, nhóm) từ 7.735 user — gấp 5,3 lần số cặp của Phần II (1.890).

### 18.2 Vì sao `personal` tụt từ +12,6% xuống +2,7% — phân rã bằng hai phép đối chứng

Ba thứ đổi cùng lúc, nên chạy thêm hai cấu hình trung gian để tách:

| # | bộ chia | kho nhạc | số cặp | `personal` so với tọa độ tự đặt | `seed3` |
|---|---|---|---|---|---|
| 1 | cũ (che 30%) | 50.683 | 1.890 | **+12,6%** | +0,1% |
| 2 | cũ (che 30%) | 50.683 | **18.029** | +10,1% | −0,0% |
| 3 | **chính thức** | 50.683 | 10.094 | +8,1% | −5,0% |
| 4 | **chính thức** | **22.557** | 10.094 | **+2,7%** | −9,0% |

| Yếu tố | Thay đổi | Phần giải thích |
|---|---|---|
| Cỡ mẫu (1.890 → 18.029 cặp) | +12,6% → +10,1% | −2,5 điểm |
| Bộ chia (che 30% → tập test chính thức) | +10,1% → +8,1% | −2,0 điểm |
| **Kho nhạc (50.683 → 22.557 bài)** | +8,1% → +2,7% | **−5,4 điểm — lớn nhất** |

**Đọc ra:** lọc 10-core bỏ đi 28.126 bài ít người nghe nhất. Đó chính là chỗ mục tiêu cá
nhân hóa phát huy tác dụng — trong một kho chỉ còn bài tương đối phổ biến, bốn cách đặt
mục tiêu lấy ra những bài khá giống nhau nên cá nhân hóa thêm được ít. Con số **+12,6%
trong Phần I không phải sai, nhưng nó đo trên một kho nhạc rộng hơn kho nhóm thực sự phục
vụ**; trên dữ liệu chính thức mức lợi là **+2,7%**.

### 18.3 `seed3` — bị loại dứt khoát

Ở Phần II, 3 bài ví dụ ngang bằng tọa độ tự đặt (+0,1%). Trên dữ liệu chính thức với số cặp
gấp 5 lần, nó **kém hơn rõ rệt (−9,0%)**, và kém ở cả ba cấu hình đối chứng (−0,0%, −5,0%,
−9,0%). Phát hiện "implicit thắng explicit" của Hausberger, Jósár & Schedl (NLP4MusA 2026)
**không chuyển sang hệ này** — kết luận này nay chắc chắn hơn Phần II.

### 18.4 Hệ quả thiết kế

- Giữ thanh kéo, giữ `personal` — vẫn là cách tốt nhất, nhưng **phải báo cáo +2,7% chứ không
  phải +12,6%** khi nói về dữ liệu chính thức.
- `learned` nay kém hơn `manual` (−4,7%). Tọa độ tự đặt trong config vẫn là lựa chọn đúng.
- Không đưa `seed3` vào `api.py`.

## 19. File mới và cách chạy lại Phần III

| File | Nội dung |
|---|---|
| `src/mood/splits.py` | tầng đọc bộ chia train/test chính thức của TV1 |
| `data/processed/split_{train,test}.parquet` | cache cột cần dùng |
| `data/processed/taste_als_official.npz`, `taste_als_official_engage.npz` | hai mô hình gu mới |
| `data/processed/four_sets_summary_n{3,5,10}_official.csv` | A2 trên bộ chia chính thức |
| `data/processed/four_sets_summary_n3_official_engage.csv` | A2 với trọng số gắn bó |
| `data/processed/labeling_compare_official.csv`, `tag_audio_position_official.csv` | A1 trên catalogue chính thức |
| `data/processed/mood_eval_summary_official.csv` | A3 trên bộ chia chính thức |
| `data/processed/mood_eval_summary_n20k.csv`, `mood_eval_summary_official_uniall.csv` | hai phép đối chứng của mục 18.2 |
| `reports/figures/four_sets_n{3,5,10}_official.png`, `four_sets_n3_official_engage.png`, `mood_eval_official.png` | hình Phần III |

```bash
python -m src.mood.splits                                              # kiểm tra bộ chia
python -m src.mood.taste --split official                              # ALS trên train chính thức (~3 phút)
python -m src.mood.taste --split official --weight engagement_score    # biến thể trọng số gắn bó
python -m src.mood.evaluate_four_sets --split official --n-per-set 3 --n-users 60000
python -m src.mood.evaluate_four_sets --split official --weight engagement_score --n-per-set 3 --n-users 60000
python -m src.mood.labeling_embed --catalogue official
python -m src.mood.evaluate_mood --split official --n-users 20000
python -m src.mood.evaluate_mood --split local --n-users 20000 --out-tag _n20k          # đối chứng cỡ mẫu
python -m src.mood.evaluate_mood --split official --n-users 20000 --universe all --out-tag _uniall  # đối chứng kho nhạc
python -m pytest tests -q                                              # 95 test
```

Mọi lệnh cũ của Phần II vẫn chạy được bằng `--split local`; các số đã công bố ở Phần I và
Phần II không bị ghi đè.

## 20. Tổng kết Phần III

| Task | Kết luận Phần II | Trên dữ liệu chính thức | Đổi? |
|---|---|---|---|
| **A2** | ghép 0,5 hơn chỉ-gu +23,9%, `taste_weight = 0,5` | +24,0% / +23,4% / +24,9% (N=3/5/10) | **không** |
| **A2 + engagement** | — | **+30,4%** — tốt hơn `playcount` | **mới** |
| **A1** | giữ bảng tay (83,0% / 0,5711) | giữ bảng tay (84,9% / 0,5656) | **không** |
| **A3 `seed3`** | ngang tọa độ tự đặt (+0,1%), loại | **kém hơn rõ (−9,0%)**, loại | **mạnh hơn** |
| **A3 `personal`** | +12,6% so với tọa độ tự đặt | **+2,7%** | **có — phải sửa số** |

---

# Phần IV — Bảng G1: đánh giá trên toàn catalogue (2026-11-22)

## 21. Vì sao cần bảng này

Phần I–III trả lời "tâm trạng thêm được gì cho gu" (A2) và "cách đặt mục tiêu nào tốt hơn"
(T6/A3). Không phần nào trả lời câu cơ bản nhất của một hệ gợi ý: **đưa một người dùng vào, hệ
có xếp đúng bài họ thật sự đã nghe lên đầu không?**

A2 chỉ xếp **12 bài dựng sẵn** (4 tập × N=3), nên không nói được chất lượng trên kho thật. Bảng
này lấy **toàn bộ catalogue đã fit** làm ứng viên và đo trên phần test bị giữ lại.

Bảng nằm sẵn trong `taste-model/docs/HANDOFF_CHECKLIST.md` của TV2 ("Bảng báo cáo để điền sau"),
trước đây để trống với dòng *"Chưa chạy thật"*.

## 22. Cách bảo đảm hai dòng so được với nhau

`src/mood/evaluate_catalogue.py` **không viết lại pipeline của TV2**. Nó nạp và gọi đúng các hàm
của TV2 — `load_processed`, `inner_train_validation`, `make_truth`, `measure`, `evaluate` — nên
cohort, catalogue, tập ứng viên, cách dựng ground truth và công thức metric đều y nguyên.

Khi dựng bảng này, bộ code của TV2 còn bị tách làm hai chỗ: `models.py` và
`run_real_taste.py` ở `music-mood-recsys/src/`, còn bản `data.py`/`evaluate.py` mà
`run_real_taste.py` cần thì chỉ có ở `taste-model/src/` (repo nhóm giữ bản scaffold cũ có cùng
tên nhưng khác API). **TV2 đã sửa cùng ngày** (commit `26516b5`, PR #2): hai file đẩy lên repo
nhóm **giống hệt từng byte** bản đã dùng cho G1, nên **số liệu không bị ảnh hưởng**.

`evaluate_catalogue.py` vì thế chọn nguồn **theo từng file** chứ không theo cả thư mục, và nhận
ra bản scaffold cũ qua hàm đặc trưng (`clean_interactions`, `ranking_metrics`) để bỏ qua. Cách
này chạy đúng ở cả hai tình huống: repo nhóm đã cập nhật (lấy hết từ đó) hay chưa (ghép hai nguồn).

**Kiểm chứng:** chạy lại baseline của TV2 trên máy này ra **đúng số của TV2** —
`popularity` khớp tuyệt đối (sai số 0), `als_binary` khớp tới `1e-6` (nhiễu float do numpy 2.5 /
pandas 3.0 so với numpy 1.26 / pandas 2.2 của TV2). Đối chiếu với
`music-mood-recsys` commit `9ddd4b5:reports/results/taste_10k_01/metrics.csv`.
Hai file dữ liệu cũng khớp **SHA256 từng bit** với manifest của TV2.

### Cấu hình và giao thức

| | Giá trị |
|---|---|
| Mô hình gu | ALS **binary**, factors 32, iterations 8, reg 0,1, alpha 10, seed 42 |
| Chọn cấu hình gu | TV2 chọn trên validation NDCG@10 (binary 0,0862 > engagement 0,0847 > log 0,0835) |
| Chọn α | quét 8 giá trị trên **validation**, lấy max NDCG@10 → **α = 0,9** |
| Test | chạy **đúng một lần** ở α đã chọn |
| Cohort | 10.000 user train · 2.000 user test · catalogue **17.691 bài** |
| Mục tiêu tâm trạng | nhóm nghe trội nhất trong **lịch sử thấy được**, rồi `user_mood_target` co về tâm chung |
| Phủ hồ sơ tâm trạng | **100%** bài trong train sample |

Mục tiêu tâm trạng **không bao giờ** dùng phần bị giữ lại: ở validation suy từ inner-train, ở test
suy từ train.

## 23. Kết quả

### 23.1 Bảng G1 (test, 2.000 user)

| Mô hình | Recall@10 | NDCG@10 | MAP@10 | Recall@50 | NDCG@50 | MAP@50 |
|---|---|---|---|---|---|---|
| Popularity | 0,0307 | 0,0243 | 0,0124 | 0,0846 | 0,0418 | 0,0155 |
| **ALS binary** (TV2 chọn) | **0,1464** | **0,1302** | **0,0788** | **0,3007** | **0,1812** | **0,0908** |
| ALS + mood, α = 0,9 *(chọn trên validation)* | 0,1453 | 0,1290 | 0,0778 | 0,2992 | 0,1802 | 0,0898 |
| ALS + mood, α = 0,5 *(điểm vận hành của A2, chỉ báo cáo)* | 0,0662 | 0,0593 | 0,0319 | 0,1222 | 0,0779 | 0,0351 |

Dòng α = 0,5 **không được dùng để chọn gì**; nó là điểm vận hành định trước lấy từ A2, báo cáo để
thấy mức đánh đổi. Cột `selection_use` trong `metrics.csv` ghi rõ phân biệt này.

### 23.2 Đọc ra ba điều

**1. Module gu có giá trị rõ ràng.** ALS hơn Popularity **4,77× Recall@10 · 5,36× NDCG@10 ·
6,38× MAP@10**. Đây là câu trả lời cho "cá nhân hóa có hơn việc đưa bài phổ biến không".

**2. Ở α = 0,9 ghép tâm trạng gần như không mất gì.** Chênh dưới 1,3% ở cả 6 chỉ số
(NDCG@10 −0,92%). Trên validation α = 0,9 còn *hơn* ALS thuần (0,0887 vs 0,0862, +2,96%) nhưng
mức tăng đó **không sang được test** → kết luận đúng là **ngang nhau**, mức tăng trên validation
là nhiễu. Không được báo cáo +2,96% như một cải thiện.

**3. Đường cong α rất dốc.** NDCG@10 trên validation:

| α | 0 | 0,25 | 0,4 | 0,5 | 0,6 | 0,75 | **0,9** | 1,0 |
|---|---|---|---|---|---|---|---|---|
| NDCG@10 | 0,0004 | 0,0166 | 0,0338 | 0,0455 | 0,0605 | 0,0803 | **0,0887** | 0,0862 |
| so với α=1 | −99,5% | −80,8% | −60,8% | −47,3% | −29,8% | −6,8% | **+3,0%** | — |

Trên test, α = 0,5 mất **−54,5% NDCG@10** so với ALS thuần. Nhưng vẫn **hơn Popularity 2,44×
NDCG@10 · 2,16× Recall@10** — tức có điều khiển tâm trạng thì hệ vẫn cá nhân hóa tốt hơn baseline
rõ rệt, không trở thành vô dụng.

## 24. Hai bảng căng nhau — và đó là lập luận cho thiết kế thanh kéo

| | A2 (bốn tập) | G1 (toàn catalogue) |
|---|---|---|
| Ground truth | bài bị che **thuộc nhóm tâm trạng m** | bài user **thật sự đã nghe**, không điều kiện tâm trạng |
| α tốt nhất | **0,5** | **0,9–1,0** |
| Ở α = 0,5 | set1 **+24,0%** so với chỉ-gu | **−54,5%** NDCG@10 so với chỉ-gu |

Cả hai đều đúng vì **tối ưu hai mục tiêu khác nhau**. Khi người dùng có yêu cầu tâm trạng, α = 0,5
đáng giá; khi mục tiêu chỉ là đoán tiếp thói quen nghe, α cao hơn tốt hơn.

**Hệ quả thiết kế:** không tồn tại một α tốt nhất, nên **để người dùng kéo α là lựa chọn đúng**,
không phải cách lảng tránh việc tối ưu. Trước Phần IV nhóm chỉ biện luận điều này bằng cảm tính;
giờ có hai bảng số cho hai đầu của sự đánh đổi.

Khi bị hỏi *"sao thêm tâm trạng lại kém đi?"*: **vì tập test không chứa yêu cầu tâm trạng — người
dùng trong dữ liệu không nói họ muốn nghe gì, họ chỉ nghe. Dựng đúng tình huống có yêu cầu tâm
trạng (A2) thì ghép 0,5 thắng chỉ-gu 24,0%.**

## 25. File mới và cách chạy lại Phần IV

```bash
cd mood-model
python -m src.mood.evaluate_catalogue \
    --train ../data/train.csv --test ../data/test.csv \
    --out reports/results/g1_taste_mood_10k \
    --max-train-users 10000 --eval-users 2000 \
    --backend implicit --report-alphas 0.5
```

| File | Nội dung |
|---|---|
| `src/mood/evaluate_catalogue.py` | dựng bảng G1; nạp code TV2 từ hai nguồn, lớp `TasteMoodHybrid` |
| `reports/results/g1_taste_mood_10k/metrics.csv` | 14 dòng: validation (quét α) + test |
| `reports/results/g1_taste_mood_10k/selected_alpha.json` | α đã chọn + quy tắc chọn |
| `reports/results/g1_taste_mood_10k/manifest.json` | SHA256 dữ liệu, cấu hình, đường dẫn 2 file code TV2 |

## 26. Hạn chế của Phần IV

1. **Cohort 10.000 user**, không phải toàn bộ 289.572 — giữ nguyên cỡ của TV2 để hai dòng so được.
   Catalogue vì thế là **17.691 bài**, không phải 22.557.
2. **895 positive ở test bị loại** vì bài không nằm trong catalogue đã fit (số của TV2, giữ nguyên).
3. **Chưa có khoảng tin cậy.** Chênh −0,92% ở α = 0,9 nằm trong khoảng không phân biệt được bằng
   mắt; cần bootstrap theo user mới kết luận chắc là "ngang nhau".
4. **Mục tiêu tâm trạng là suy đoán, không phải yêu cầu thật.** Hệ đoán nhóm trội từ lịch sử nghe.
   Đây là cách duy nhất làm được khi dữ liệu không có yêu cầu tâm trạng theo lượt nghe.
5. **A2 và G1 dùng hai cấu hình ALS khác nhau** cho tới khi A2 được chạy lại bằng `binary`/8 vòng
   (xem `docs/decisions.md` 2026-11-22). Đây là việc P0 còn lại của Phần IV.

---

## 27. A2 chạy lại bằng cấu hình gu chính thức — kết luận **không đổi**

### 27.1 Vì sao phải chạy lại

A2 ở Phần II/III dùng ALS `weighting="log"` (tức `log1p(playcount)`) với **15 vòng** — cấu hình
mặc định, `docs/decisions.md` 2026-10-30 ghi rõ *"không grid-search, cần TV2 xác nhận"*. Ngày
22/11 TV2 chọn cấu hình bằng validation NDCG@10: **`binary`, 8 vòng**. Nếu để nguyên thì G1 và A2
dùng **hai mô hình gu khác nhau** trong cùng báo cáo.

Lưu ý thuật ngữ: `--weight playcount` nói **cột nào** nuôi confidence (`playcount` hay
`engagement_score`); `weighting` mới là **hàm tin cậy** (`log` / `binary` / `engagement`). Hai
thứ khác nhau, trước đây dễ bị gọi lẫn.

### 27.2 Kết quả — NDCG@3, set1 (đúng gu **và** đúng tâm trạng)

| Cấu hình ALS | cohort | n_cases | chỉ gu | ghép 0,5 | mức tăng | phân biệt tâm trạng (ghép 0,5) |
|---|---|---|---|---|---|---|
| `log` / 15 vòng *(đã công bố)* | 289.572 user · 22.557 bài | 18.001 | 0,4923 | 0,6106 | **+24,0%** | +0,3594 |
| **`binary` / 8 vòng** *(chính thức)* | 289.572 user · 22.557 bài | 18.001 | 0,4922 | **0,6120** | **+24,3%** | +0,3605 |
| **`binary` / 8 vòng** | **10.000 user · 17.691 bài** (cohort của G1) | 2.872 | 0,4796 | 0,6003 | **+25,2%** | +0,3604 |

Thứ hạng trọng số ghép **không đổi** ở cả ba: 0,5 luôn tốt hơn 0,3 và 0,7.

| | ghép 0,3 | **ghép 0,5** | ghép 0,7 |
|---|---|---|---|
| `log`/15, đầy đủ | 0,5527 | **0,6106** | 0,5669 |
| `binary`/8, đầy đủ | 0,5542 | **0,6120** | 0,5674 |
| `binary`/8, 10k | 0,5439 | **0,6003** | 0,5583 |

### 27.3 Đọc ra

**1. Kết luận A2 bền với cấu hình gu.** Đổi hàm tin cậy và số vòng chỉ làm mức tăng nhúc nhích
+24,0% → +24,3%. Mức phân biệt tâm trạng gần như không đổi (+0,3594 → +0,3605). Nghĩa là
"tâm trạng bổ sung thật cho gu" **không phải hệ quả của một cấu hình ALS may mắn**.

**2. Thu nhỏ cohort cũng không đổi kết luận** (+25,2% trên 10.000 user). Nhưng cỡ mẫu nhỏ hơn
6,3 lần (2.872 so với 18.001 cặp) và có **830 bài bị che bị loại vì ngoài catalogue** (cohort đầy
đủ: 0), nên con số này nhiễu hơn — dùng để đối chứng, không dùng làm số chính.

**3. Mô hình gu càng mù tâm trạng hơn ở cohort nhỏ**: mức phân biệt của `taste` giảm từ +0,0208
xuống **+0,0100**. Củng cố luận điểm gốc của A2.

### 27.4 Số nào dùng cho báo cáo

**Dùng `binary`/8 trên cohort đầy đủ: ghép 0,5 đạt 0,6120 ở set1, hơn chỉ-gu +24,3%.** Lý do: đó
là mô hình gu chính thức của nhóm, cùng cấu hình với bảng G1, nên cả hai bảng nói về **một hệ
thống**. Giữ dòng `log`/15 (+24,0%) trong báo cáo như **phép kiểm tra độ bền**, không phải số chính.

Kết luận thiết kế của A2 (`taste_weight = 0,5`) **không thay đổi**.

### 27.5 File

| File | Nội dung |
|---|---|
| `data/processed/taste_als_official_binary8.npz` | ALS chính thức, cohort đầy đủ (rò rỉ 0) |
| `data/processed/taste_als_official_binary8_g1_10k.npz` | ALS chính thức, cohort 10k của G1 (rò rỉ 0) |
| `data/processed/cohort_g1_10k.csv` | đúng 10.000 user của G1 (kiểm: 2.000/2.000 user G1 nằm trong) |
| `data/processed/four_sets_summary_n3_official_binary8*.csv` | bảng bốn tập |
| `reports/figures/four_sets_n3_official_binary8*.png` | hình |

```bash
# ALS chinh thuc, cohort day du
python -m src.mood.taste --split official --als-weighting binary --als-iterations 8
python -m src.mood.evaluate_four_sets --split official --n-per-set 3 --n-users 60000 \
    --als-weighting binary --als-iterations 8

# ALS chinh thuc, cohort 10k cua G1
python -m src.mood.taste --split official --als-weighting binary --als-iterations 8 \
    --users-file data/processed/cohort_g1_10k.csv --cohort-tag g1_10k
python -m src.mood.evaluate_four_sets --split official --n-per-set 3 --n-users 60000 \
    --als-weighting binary --als-iterations 8 \
    --cohort-tag g1_10k --users-file data/processed/cohort_g1_10k.csv
```

Mặc định của `taste.py` và `evaluate_four_sets.py` **không đổi** (`log`/15 vòng, hậu tố rỗng), nên
mọi số Phần I–III vẫn tái lập đúng và không file nào bị ghi đè.

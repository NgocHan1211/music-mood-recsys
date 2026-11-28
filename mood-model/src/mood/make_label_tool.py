"""Sinh trang gán nhãn tâm trạng để nghe-và-bấm, thay cho việc gõ tay trong Google Sheets.

Vì sao cần: gán 180 bài trong bảng tính nghĩa là bấm link, chờ tab mới, nghe, quay
lại bảng, gõ chữ — lặp 180 lần. Trang này nhúng sẵn trình phát 30 giây và 5 nút nhóm
đặt đúng toạ độ centroid đã học, nên mỗi bài chỉ còn hai động tác: nghe và bấm.

Trang là MỘT file HTML tĩnh, mở bằng cách bấm đôi, không cần cài gì, không cần
tài khoản. Nhãn tự lưu vào localStorage của trình duyệt và xuất ra CSV đúng schema
`annotation/nhan_<ten>.csv` mà `score_agreement.py` đọc.

Trang KHÔNG chứa nhãn máy (mood_label / mood_pred / tags) — giữ nguyên tính mù.
Xem `docs/huong_dan_gan_nhan_mood.md`.

Chạy:
    python -m src.mood.make_label_tool
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import GROUPS, PROCESSED_DIR

MOOD_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_DIR = MOOD_ROOT / "annotation"

# Nhãn tiếng Việt + gợi ý ngắn cho từng nhóm. Gợi ý nói về CẢM GIÁC, không nói về
# thể loại hay tag — nói thể loại thì người gán chỉ chép lại quyết định của bảng tag.
GROUP_UI = {
    "vui":        ("Vui",         "Sáng, phấn, nghe xong thấy nhẹ người"),
    "buon":       ("Buồn",        "Sắc thái mất mát, đau — KHÔNG cứ phải nhạc nhẹ"),
    "thu_gian":   ("Thư giãn",    "Êm, chậm, không kéo cảm xúc đi đâu"),
    "nang_dong":  ("Năng động",   "Dồn, muốn chuyển động, nhưng không đe dọa"),
    "cang_thang": ("Căng thẳng",  "Gắt, dữ, bồn chồn, tối"),
}


def load_tracks(blind_csv: Path) -> list[dict]:
    with blind_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [{"id": r["item_id"], "url": r["link_nghe_30s"], "sp": r["link_spotify"],
             "name": r["ten_bai"], "artist": r["nghe_si"]} for r in rows]


def load_centroids() -> dict[str, list[float]]:
    path = PROCESSED_DIR / "mood_centroids.json"
    cent = json.loads(path.read_text(encoding="utf-8"))["centroids"]
    return {g: [round(cent[g]["mean"][0], 4), round(cent[g]["mean"][1], 4)] for g in GROUPS}


def build_html(tracks: list[dict], centroids: dict[str, list[float]]) -> str:
    groups = [{"key": g, "label": GROUP_UI[g][0], "hint": GROUP_UI[g][1],
               "hotkey": str(i + 1), "pos": centroids[g]}
              for i, g in enumerate(GROUPS)]
    template = (Path(__file__).parent / "label_tool_template.html").read_text(encoding="utf-8")
    return (template
            .replace("/*__TRACKS__*/null", json.dumps(tracks, ensure_ascii=False))
            .replace("/*__GROUPS__*/null", json.dumps(groups, ensure_ascii=False)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sinh trang gán nhãn tâm trạng (1 file HTML tĩnh, mở bằng trình duyệt).")
    parser.add_argument("--blind", type=Path,
                        default=ANNOTATION_DIR / "mood_gold_blind.csv",
                        help="file mù sinh bởi build_label_set.py")
    parser.add_argument("--out", type=Path, default=ANNOTATION_DIR / "gan_nhan.html")
    args = parser.parse_args()

    tracks = load_tracks(args.blind)
    centroids = load_centroids()
    html = build_html(tracks, centroids)
    args.out.write_text(html, encoding="utf-8")

    print(f"{len(tracks)} bài, 5 nhóm đặt theo centroid đã học:")
    for g in GROUPS:
        v, e = centroids[g]
        print(f"  {g:12} valence {v:.3f}  energy {e:.3f}")
    print(f"\nĐã ghi: {args.out.relative_to(MOOD_ROOT)}  ({len(html) / 1024:.0f} KB)")
    print("Mở bằng cách bấm đôi vào file, hoặc: open " + str(args.out))


if __name__ == "__main__":
    main()

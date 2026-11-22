"""L1 — Gán nhãn tâm trạng bằng LLM đọc TÊN BÀI và TÊN NGHỆ SĨ.

Khác A1 (`labeling_embed.py`) ở chỗ quan trọng nhất: A1 nhúng **tên tag**
(`melodic_death_metal` — nghèo nghĩa, và đã thua bảng tay 52,3% so với 84,9%).
L1 dùng hai cột module tâm trạng CHƯA BAO GIỜ đụng tới: `name` và `artist`.

Lý do làm: 9.888 / 22.557 bài của catalogue chính thức (43,8%) không gán được nhãn
từ tag — gồm cả Bohemian Rhapsody, Wonderwall, Mr. Brightside. Đây là hạn chế số 1
nhóm đã tự ghi trong báo cáo ("3/5 nhóm không có tag cảm xúc nào").

Hai biến thể, để tách phần đóng góp của tên bài khỏi phần đóng góp của tag:
    llm_ten      chỉ tên bài + nghệ sĩ
    llm_ten_tag  thêm danh sách tag

Hai nhánh đánh giá:
    hiệu chuẩn  chạy trên bài ĐÃ có nhãn tay -> đo mức trùng khớp và mức khớp vị trí
                âm thanh của cả hai bên trên cùng một tập
    mở rộng     chạy trên bài tag bó tay -> đo độ phủ tăng thêm và macro-F1 của bộ
                phân loại T3 huấn luyện trên nhãn tay + nhãn LLM

Luôn báo cáo tách theo ĐỘ PHỔ BIẾN: LLM biết sẵn bài nổi tiếng nên kết quả ở nhóm
bài ít người nghe mới là phần khái quát được.

Module KHÔNG ghi đè `track_mood_labels.parquet`.

Chạy:  python -m src.mood.labeling_llm --stage calib   (rồi --stage expand, --stage eval)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import api
from .config import GROUPS, PROCESSED_DIR, SEED, load_config
from .io import load_tracks
from .labeling import LABELS_PARQUET

# Ba đường chạy. "mlx" là mặc định vì miễn phí và không cần đăng ký gì:
#   local          mô hình mở chạy ngay trên máy (torch + MPS), không tốn tiền
#                  (MLX không dùng được: nhân Metal của nó không biên dịch trên GPU M5)
#   anthropic      API Anthropic, chất lượng tốt nhất, có tính phí
#   openai_compat  mọi API theo chuẩn OpenAI (Groq, OpenRouter... có bậc miễn phí)
BACKENDS = {
    "local": "Qwen/Qwen2.5-3B-Instruct",
    "anthropic": "claude-haiku-5-5",
    "openai_compat": "openai/gpt-oss-120b",
}
BACKEND = "local"
MODEL = BACKENDS[BACKEND]
# Mô hình nhỏ chạy máy dễ lạc số thứ tự khi danh sách dài -> lô nhỏ hơn
BATCH_SIZE = {"local": 20, "anthropic": 40, "openai_compat": 40}
# Bậc miễn phí của Groq: 1.000 lượt/ngày và 8.000 token/phút cho mọi mô hình.
# Token mỗi phút mới là chỗ nghẽn -> tự điều tiết để không bị 429 liên tục.
TOKENS_PER_MIN = 8000
MAX_WORKERS = 8
VARIANTS = {"llm_ten": False, "llm_ten_tag": True}   # tên biến thể -> có cho xem tag không
UNKNOWN = "khong_ro"

COMPARE_CSV = PROCESSED_DIR / "llm_compare.csv"
POPULARITY_CSV = PROCESSED_DIR / "llm_by_popularity.csv"


def model_tag(model: str | None = None) -> str:
    """Phần tên mô hình dùng trong tên file, bỏ tiền tố nhà cung cấp và ký tự lạ."""
    m = MODEL if model is None else model
    return re.sub(r"[^0-9a-zA-Z._-]", "-", m.split("/")[-1])


def cache_csv(variant: str, stage: str, backend: str | None = None,
              model: str | None = None):
    """Mỗi (backend, MÔ HÌNH, biến thể, nhánh) một file cache riêng.

    Phải có cả tên mô hình: hai mô hình khác nhau trên cùng một backend mà dùng chung
    file thì kết quả của chúng sẽ bị trộn vào nhau.
    """
    backend = BACKEND if backend is None else backend
    return PROCESSED_DIR / f"llm_labels_{backend}_{model_tag(model)}_{variant}_{stage}.csv"


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

GROUP_DESC = {
    "vui": "vui tươi, phấn chấn, tích cực",
    "buon": "buồn, u sầu, nặng nề về cảm xúc",
    "thu_gian": "êm dịu, nhẹ nhàng, để thư giãn",
    "nang_dong": "sôi động, nhiều năng lượng, để nhảy hoặc vận động",
    "cang_thang": "căng thẳng, dữ dội, gây bất an hoặc kích động",
}

SYSTEM = (
    "Bạn phân loại tâm trạng của bài hát. Với mỗi bài, chọn ĐÚNG MỘT nhóm:\n"
    + "\n".join(f"- {g}: {d}" for g, d in GROUP_DESC.items())
    + f"\n- {UNKNOWN}: không đủ căn cứ để chọn, hoặc bài không rõ ràng thuộc nhóm nào.\n\n"
    "Căn cứ vào cảm giác chung khi nghe bài đó, không căn cứ vào lời bài hát riêng lẻ.\n"
    f"Thà trả lời {UNKNOWN} còn hơn đoán bừa khi bạn không biết bài hát.\n"
    "Chỉ trả về JSON, không giải thích."
)

SYSTEM_COMPACT = (
    SYSTEM.replace("Chỉ trả về JSON, không giải thích.", "")
    + "Trả lời đúng định dạng, mỗi bài một dòng:\n"
      "<số thứ tự>:<tên nhóm>\n"
      "Không viết gì khác, không giải thích, không bỏ sót số nào."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "ket_qua": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "i": {"type": "integer"},
                    "nhom": {"type": "string", "enum": GROUPS + [UNKNOWN]},
                    "chac_chan": {"type": "string", "enum": ["cao", "vua", "thap"]},
                },
                "required": ["i", "nhom", "chac_chan"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ket_qua"],
    "additionalProperties": False,
}


def build_prompt(rows: pd.DataFrame, with_tags: bool) -> str:
    """Danh sách bài đánh số, mỗi dòng một bài."""
    lines = []
    for i, r in enumerate(rows.itertuples(index=False)):
        line = f'{i}. "{r.name}" — {r.artist}'
        # pandas biến None thành NaN -> phải kiểm tra bằng notna, nếu không
        # bài không có tag sẽ bị gửi đi dưới dạng "[tag: nan]"
        if with_tags and pd.notna(r.tags) and str(r.tags).strip():
            line += f"  [tag: {r.tags}]"
        lines.append(line)
    return ("Phân loại tâm trạng cho từng bài dưới đây. "
            "Trả về đúng một phần tử cho mỗi số thứ tự.\n\n" + "\n".join(lines))


# --------------------------------------------------------------------------- #
# Gọi API
# --------------------------------------------------------------------------- #

_LOADED = {}       # cache mô hình / client, nạp một lần cho cả tiến trình
_PACE = threading.Lock()
_SPENT: list[tuple[float, int]] = []    # (thời điểm, số token) trong 60 giây gần nhất


def _throttle(est_tokens: int) -> None:
    """Chặn lại cho tới khi gửi thêm `est_tokens` vẫn nằm dưới hạn mức mỗi phút."""
    while True:
        with _PACE:
            now = time.time()
            _SPENT[:] = [(t, n) for t, n in _SPENT if now - t < 60]
            used = sum(n for _, n in _SPENT)
            if used + est_tokens <= TOKENS_PER_MIN:
                _SPENT.append((now, est_tokens))
                return
            wait = 60 - (now - _SPENT[0][0]) + 0.2
        time.sleep(max(wait, 0.2))


def _record(actual: int, est: int) -> None:
    """Chỉnh lại sổ sách bằng số token thật mà API báo."""
    with _PACE:
        _SPENT.append((time.time(), max(actual - est, 0)))
_LINE = re.compile(r"^\D*(\d+)\s*[:.\-)]\s*([a-z_]+)", re.M)


def _parse_lines(text: str, n: int) -> dict[int, str]:
    """Đọc định dạng gọn `số:nhóm` (dùng cho mô hình không ép được JSON schema)."""
    out: dict[int, str] = {}
    for m in _LINE.finditer(text):
        i, g = int(m.group(1)), m.group(2)
        if 0 <= i < n and g in GROUPS:
            out[i] = g
    return out


# --------------------------------------------------------------------------- #

def _local_classify(rows: pd.DataFrame, with_tags: bool) -> list[tuple[str, str]]:
    """Mô hình mở chạy trên máy qua torch + MPS. Tất định (greedy), tái lập được."""
    if "model" not in _LOADED:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"  (nạp {MODEL} lên {dev} — lần đầu phải tải về)", flush=True)
        tok = AutoTokenizer.from_pretrained(MODEL)
        tok.padding_side = "left"          # decoder-only: phải đệm bên trái
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _LOADED["tok"] = tok
        _LOADED["model"] = AutoModelForCausalLM.from_pretrained(
            MODEL, dtype=torch.float16).to(dev).eval()
        _LOADED["dev"], _LOADED["torch"] = dev, torch

    torch, tok, model = _LOADED["torch"], _LOADED["tok"], _LOADED["model"]
    msgs = [{"role": "system", "content": SYSTEM_COMPACT},
            {"role": "user", "content": build_prompt(rows, with_tags)}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(_LOADED["dev"])
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=12 * len(rows) + 40, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    text = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    got = _parse_lines(text, len(rows))
    return [(got.get(i, UNKNOWN), "") for i in range(len(rows))]


def _anthropic_classify(rows: pd.DataFrame, with_tags: bool) -> list[tuple[str, str]]:
    """API Anthropic, ép định dạng bằng JSON schema."""
    import anthropic
    if "client" not in _LOADED:
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise RuntimeError(
                "Chưa có thông tin xác thực. Đặt ANTHROPIC_API_KEY rồi chạy lại, "
                "hoặc dùng --backend local (miễn phí, chạy trên máy).")
        _LOADED["client"] = anthropic.Anthropic(max_retries=5)
    resp = _LOADED["client"].messages.create(
        model=MODEL,
        max_tokens=4000,
        # Phân loại là việc đơn giản -> effort thấp để khỏi tốn token suy nghĩ
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": build_prompt(rows, with_tags)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"API từ chối: {resp.stop_details}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    got = {int(it["i"]): (it["nhom"], it.get("chac_chan", "")) for it in json.loads(text)["ket_qua"]}
    return [got.get(i, (UNKNOWN, "")) for i in range(len(rows))]


def _openai_compat_classify(rows: pd.DataFrame, with_tags: bool) -> list[tuple[str, str]]:
    """Mọi API theo chuẩn OpenAI (Groq, OpenRouter...). Đặt LLM_BASE_URL và LLM_API_KEY."""
    from openai import OpenAI
    if "client" not in _LOADED:
        key = os.environ.get("LLM_API_KEY")
        if not key:
            raise RuntimeError("Chưa có LLM_API_KEY (và LLM_BASE_URL) cho backend openai_compat.")
        _LOADED["client"] = OpenAI(
            api_key=key, max_retries=8,       # tự thử lại khi dính 429
            base_url=os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    user = build_prompt(rows, with_tags)
    # Ước lượng thô: ~4 ký tự một token, cộng phần sinh ra
    est = (len(SYSTEM_COMPACT) + len(user)) // 4 + 14 * len(rows) + 200
    _throttle(est)

    kw = {}
    if "gpt-oss" in MODEL:          # mô hình suy luận -> hạ mức suy nghĩ cho việc phân loại
        kw["reasoning_effort"] = "low"
    resp = _LOADED["client"].chat.completions.create(
        model=MODEL, temperature=0, max_tokens=14 * len(rows) + 800,
        messages=[{"role": "system", "content": SYSTEM_COMPACT},
                  {"role": "user", "content": user}], **kw)
    if resp.usage:
        _record(resp.usage.total_tokens, est)
    got = _parse_lines(resp.choices[0].message.content or "", len(rows))
    return [(got.get(i, UNKNOWN), "") for i in range(len(rows))]


CLASSIFIERS = {"local": _local_classify, "anthropic": _anthropic_classify,
               "openai_compat": _openai_compat_classify}


def classify_batch(rows: pd.DataFrame, with_tags: bool) -> list[dict]:
    """Một lô. Bài nào mô hình không trả về -> coi như khong_ro (không đoán bừa)."""
    pairs = CLASSIFIERS[BACKEND](rows, with_tags)
    return [{"track_id": t, "llm_nhom": g, "llm_chac_chan": c}
            for t, (g, c) in zip(rows["track_id"].tolist(), pairs)]


def run_variant(tracks: pd.DataFrame, variant: str, stage: str,
                workers: int = MAX_WORKERS, refresh: bool = False) -> pd.DataFrame:
    """Chạy một biến thể trên `tracks`, có cache theo file."""
    path = cache_csv(variant, stage)
    if path.exists() and not refresh:
        done = pd.read_csv(path, dtype={"track_id": str})
        todo = tracks[~tracks["track_id"].isin(set(done["track_id"]))]
        if todo.empty:
            print(f"  [{variant}/{stage}] dùng cache {path.name} ({len(done):,} bài)")
            return done
        print(f"  [{variant}/{stage}] cache có {len(done):,} bài, còn {len(todo):,} bài")
    else:
        done, todo = pd.DataFrame(), tracks

    with_tags = VARIANTS[variant]
    size = BATCH_SIZE[BACKEND]
    chunks = [todo.iloc[i:i + size] for i in range(0, len(todo), size)]
    # Mô hình cục bộ dùng chung một GPU -> chạy tuần tự, không có lợi khi đa luồng
    workers = 1 if BACKEND == "local" else workers
    print(f"  [{variant}/{stage}] {len(todo):,} bài -> {len(chunks)} lượt × {size} bài "
          f"({MODEL}, {workers} luồng)", flush=True)

    rows: list[dict] = []
    t0 = time.time()

    def _note(n):
        if n % 20 == 0 or n == len(chunks):
            done_n = n * size
            rate = done_n / max(time.time() - t0, 1e-9)
            left = (len(todo) - done_n) / rate if rate > 0 else 0
            print(f"    {n}/{len(chunks)} lượt | {rate:.1f} bài/s "
                  f"| còn ~{left/60:.0f} phút", flush=True)

    if workers == 1:
        for n, c in enumerate(chunks, 1):
            rows += classify_batch(c, with_tags)
            _note(n)
            if n % 20 == 0:                       # ghi cache dọc đường, mất điện không mất công
                tmp = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) \
                    else pd.DataFrame(rows)
                tmp.to_csv(path, index=False)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(classify_batch, c, with_tags) for c in chunks]
            for n, fut in enumerate(futures, 1):
                rows += fut.result()
                _note(n)
                if n % 20 == 0:               # ghi cache dọc đường: hỏng giữa chừng không mất công
                    tmp = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) \
                        else pd.DataFrame(rows)
                    tmp.to_csv(path, index=False)

    out = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


# --------------------------------------------------------------------------- #
# Thước đo
# --------------------------------------------------------------------------- #

def audio_group(df: pd.DataFrame) -> pd.Series:
    """Nhóm gần nhất theo (valence, energy) thật của từng bài — thước khách quan của A1."""
    centroids = api.load_centroids()["centroids"]
    mu = {g: np.array(centroids[g]["mean"]) for g in GROUPS}
    inv = {g: np.linalg.inv(np.array(centroids[g]["cov"])) for g in GROUPS}
    pts = df[["valence", "energy"]].to_numpy(dtype=float)
    d = np.column_stack([
        np.einsum("ij,jk,ik->i", pts - mu[g], inv[g], pts - mu[g]) for g in GROUPS])
    return pd.Series([GROUPS[i] for i in d.argmin(axis=1)], index=df.index)


def audio_match(pred: pd.Series, audio: pd.Series) -> float:
    """Tỷ lệ nhãn trùng với nhóm gần nhất theo âm thanh (bỏ bài khong_ro)."""
    ok = pred.notna() & (pred != UNKNOWN)
    return float((pred[ok] == audio[ok]).mean()) if ok.any() else np.nan


def popularity_bins(n_users: pd.Series) -> pd.Series:
    """Chia theo số người nghe — để lộ rò rỉ trí nhớ của LLM."""
    q = n_users.quantile([0.25, 0.5, 0.75]).tolist()
    return pd.cut(n_users, [-1] + q + [np.inf],
                  labels=["ít nghe nhất (25%)", "25-50%", "50-75%", "nổi nhất (25%)"])


def expanded_labels(hand: pd.DataFrame, llm: pd.DataFrame) -> pd.DataFrame:
    """Nhãn tay giữ nguyên; bài tay bó tay thì lấy nhãn LLM (bỏ khong_ro)."""
    out = hand.copy()
    add = llm[(llm["llm_nhom"] != UNKNOWN) & llm["llm_nhom"].notna()]
    mp = dict(zip(add["track_id"], add["llm_nhom"]))
    fill = out["label_status"].isin(["no_emotion_tag", "conflict"]) & out["track_id"].isin(mp)
    out.loc[fill, "mood_label"] = out.loc[fill, "track_id"].map(mp)
    out.loc[fill, "label_status"] = "labeled"
    return out


# --------------------------------------------------------------------------- #
# Chọn bài
# --------------------------------------------------------------------------- #

def prepare(catalogue_only: bool = True):
    """Trả về (df bài, nhãn tay, tập hiệu chuẩn, tập mở rộng, số người nghe)."""
    df = load_tracks()
    hand = pd.read_parquet(LABELS_PARQUET)
    if catalogue_only:
        from .splits import load_part
        train = load_part("train")
        keep = set(train["track_id"].unique())
        df = df[df["track_id"].isin(keep)].reset_index(drop=True)
        hand = hand[hand["track_id"].isin(keep)].reset_index(drop=True)
        n_users = train.groupby("track_id").size()
    else:
        n_users = pd.Series(dtype=int)

    merged = df.merge(hand[["track_id", "mood_label", "label_status"]], on="track_id")
    calib = merged[merged["label_status"] == "labeled"]
    expand = merged[merged["label_status"].isin(["no_emotion_tag", "conflict"])]
    return df, hand, calib, expand, n_users


def main() -> None:
    global BACKEND, MODEL
    parser = argparse.ArgumentParser(description="L1 — gán nhãn tâm trạng bằng LLM.")
    parser.add_argument("--backend", choices=list(BACKENDS), default=BACKEND,
                        help="mlx = mô hình mở chạy trên máy, miễn phí (mặc định)")
    parser.add_argument("--model", default=None, help="đổi mô hình của backend đang chọn")
    parser.add_argument("--stage", choices=["calib", "expand", "eval", "all"], default="all")
    parser.add_argument("--n-calib", type=int, default=1500,
                        help="số bài đã có nhãn tay dùng để hiệu chuẩn")
    parser.add_argument("--n-expand", type=int, default=0,
                        help="giới hạn số bài tag bó tay (0 = chạy hết)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--refresh", action="store_true", help="bỏ cache, gọi lại API")
    args = parser.parse_args()

    BACKEND = args.backend
    MODEL = args.model or BACKENDS[BACKEND]

    cfg = load_config()
    df, hand, calib, expand, n_users = prepare()
    rng = np.random.default_rng(SEED)

    calib_s = calib.iloc[np.sort(rng.choice(len(calib),
                                            size=min(args.n_calib, len(calib)),
                                            replace=False))]
    expand_s = expand if not args.n_expand else expand.iloc[
        np.sort(rng.choice(len(expand), size=min(args.n_expand, len(expand)), replace=False))]

    print(f"Catalogue chính thức: {len(df):,} bài")
    print(f"  hiệu chuẩn (đã có nhãn tay): {len(calib):,} -> dùng {len(calib_s):,}")
    print(f"  mở rộng (tag bó tay):        {len(expand):,} -> dùng {len(expand_s):,}")
    print(f"  backend: {BACKEND} | mô hình: {MODEL} | {BATCH_SIZE[BACKEND]} bài/lượt\n")

    if args.stage in ("calib", "all"):
        print("-- Nhánh hiệu chuẩn --")
        for v in VARIANTS:
            run_variant(calib_s, v, "calib", args.workers, args.refresh)
    if args.stage in ("expand", "all"):
        print("-- Nhánh mở rộng --")
        for v in VARIANTS:
            run_variant(expand_s, v, "expand", args.workers, args.refresh)
    if args.stage in ("eval", "all"):
        evaluate(df, hand, calib_s, expand_s, n_users, cfg)


def evaluate(df, hand, calib_s, expand_s, n_users, cfg) -> None:
    audio_all = pd.Series(audio_group(df).to_numpy(), index=df["track_id"].to_numpy())
    rows, pop_rows = [], []

    print("\n" + "=" * 78)
    print("-- Nhánh HIỆU CHUẨN: LLM so với nhãn tay, trên cùng tập bài --")
    base = calib_s.set_index("track_id")["mood_label"]
    a_hand = audio_match(base, audio_all.reindex(base.index))
    print(f"  bảng tay (T2)      : khớp vị trí âm thanh {a_hand:.1%}  (n = {len(base):,})")
    rows.append({"nhanh": "hiệu chuẩn", "cach": "bảng tay (T2)", "n": len(base),
                 "khop_am_thanh": a_hand, "trung_nhan_tay": 1.0, "ty_le_khong_ro": 0.0})

    for v in VARIANTS:
        p = cache_csv(v, "calib")
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype={"track_id": str}).drop_duplicates("track_id")
        pred = d.set_index("track_id")["llm_nhom"].reindex(base.index)
        known = pred.notna() & (pred != UNKNOWN)
        agree = float((pred[known] == base[known]).mean())
        am = audio_match(pred, audio_all.reindex(base.index))
        unk = float((~known).mean())
        print(f"  {v:<19}: khớp vị trí âm thanh {am:.1%} | trùng nhãn tay {agree:.1%} "
              f"| khong_ro {unk:.1%}")
        rows.append({"nhanh": "hiệu chuẩn", "cach": v, "n": int(known.sum()),
                     "khop_am_thanh": am, "trung_nhan_tay": agree, "ty_le_khong_ro": unk})

        if len(n_users):
            sub = pd.DataFrame({"pred": pred, "hand": base,
                                "audio": audio_all.reindex(base.index),
                                "n_users": n_users.reindex(base.index)}).dropna(subset=["n_users"])
            sub["nhom_pho_bien"] = popularity_bins(sub["n_users"])
            for g, s in sub.groupby("nhom_pho_bien", observed=True):
                k = s["pred"].notna() & (s["pred"] != UNKNOWN)
                pop_rows.append({"cach": v, "nhom_pho_bien": str(g), "n": len(s),
                                 "trung_nhan_tay": float((s.loc[k, "pred"] == s.loc[k, "hand"]).mean()),
                                 "khop_am_thanh": float((s.loc[k, "pred"] == s.loc[k, "audio"]).mean()),
                                 "ty_le_khong_ro": float((~k).mean())})

    print("\n-- Nhánh MỞ RỘNG: bài tag bó tay --")
    hand_cat = hand[hand["track_id"].isin(set(df["track_id"]))]
    n_hand = int((hand_cat["label_status"] == "labeled").sum())
    try:
        from .labeling_embed import classifier_macro_f1
        f_hand = classifier_macro_f1(df, hand_cat, cfg)
        print(f"  bảng tay (T2)      : {n_hand:,} bài có nhãn | macro-F1 {f_hand['macro_f1']:.4f}")
        rows.append({"nhanh": "mở rộng", "cach": "bảng tay (T2)", "n": n_hand,
                     "macro_f1": f_hand["macro_f1"], "accuracy": f_hand["accuracy"]})
    except Exception as exc:
        print(f"  [!] bộ phân loại trên nhãn tay lỗi: {exc}")

    for v in VARIANTS:
        p = cache_csv(v, "expand")
        if not p.exists():
            continue
        d = pd.read_csv(p, dtype={"track_id": str}).drop_duplicates("track_id")
        pred = d.set_index("track_id")["llm_nhom"]
        known = pred[(pred.notna()) & (pred != UNKNOWN)]
        am = audio_match(pred, audio_all.reindex(pred.index))
        exp = expanded_labels(hand_cat, d)
        n_exp = int((exp["label_status"] == "labeled").sum())
        row = {"nhanh": "mở rộng", "cach": f"tay + {v}", "n": n_exp,
               "khop_am_thanh": am, "ty_le_khong_ro": float(1 - len(known) / len(pred))}
        try:
            sc = classifier_macro_f1(df, exp, cfg)
            row.update({"macro_f1": sc["macro_f1"], "accuracy": sc["accuracy"]})
            extra = f" | macro-F1 {sc['macro_f1']:.4f}"
        except Exception as exc:
            row["loi"] = str(exc)
            extra = f" | bộ phân loại lỗi: {exc}"
        print(f"  tay + {v:<13}: {n_exp:,} bài có nhãn (+{n_exp - n_hand:,}) "
              f"| khớp vị trí âm thanh {am:.1%}{extra}")
        rows.append(row)

        if len(n_users):
            sub = pd.DataFrame({"pred": pred, "audio": audio_all.reindex(pred.index),
                                "n_users": n_users.reindex(pred.index)}).dropna(subset=["n_users"])
            sub["nhom_pho_bien"] = popularity_bins(sub["n_users"])
            for g, s in sub.groupby("nhom_pho_bien", observed=True):
                k = s["pred"].notna() & (s["pred"] != UNKNOWN)
                pop_rows.append({"cach": f"mở rộng · {v}", "nhom_pho_bien": str(g), "n": len(s),
                                 "trung_nhan_tay": np.nan,
                                 "khop_am_thanh": float((s.loc[k, "pred"] == s.loc[k, "audio"]).mean()),
                                 "ty_le_khong_ro": float((~k).mean())})

    out = pd.DataFrame(rows)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(COMPARE_CSV, index=False)
    print("\n" + "=" * 78)
    print("-- Tổng hợp L1 --")
    cols = [c for c in ["nhanh", "cach", "n", "khop_am_thanh", "trung_nhan_tay",
                        "ty_le_khong_ro", "macro_f1", "accuracy"] if c in out]
    print(out[cols].to_string(index=False, float_format="{:.4f}".format))

    if pop_rows:
        pop = pd.DataFrame(pop_rows)
        pop.to_csv(POPULARITY_CSV, index=False)
        print("\n-- Tách theo độ phổ biến (kiểm tra rò rỉ trí nhớ của LLM) --")
        print(pop.to_string(index=False, float_format="{:.4f}".format))
        print(f"\nĐã lưu: {POPULARITY_CSV}")
    print(f"        {COMPARE_CSV}")
    print("\nLưu ý: nhãn chính thức (track_mood_labels.parquet) KHÔNG bị ghi đè.")


if __name__ == "__main__":
    main()

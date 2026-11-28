"""Test cho đối chiếu hệ nhãn — tập trung vào chỗ đã từng sai."""
from __future__ import annotations

import pandas as pd
import pytest

from src.mood.compare_label_systems import agreement, cohen_kappa, pairwise


def test_cohen_kappa_trung_hoan_toan():
    a = pd.Series(["vui", "buon", "thu_gian", "vui"])
    assert cohen_kappa(a, a.copy()) == pytest.approx(1.0)


def test_cohen_kappa_bo_khong_ro_va_rong():
    """`khong_ro` và ô rỗng không phải một nhóm — phải bị loại khỏi phép tính."""
    a = pd.Series(["vui", "buon", "khong_ro", ""])
    b = pd.Series(["vui", "buon", "vui", "vui"])
    ag, n = agreement(a, b)
    assert n == 2 and ag == pytest.approx(1.0)
    assert cohen_kappa(a, b) == pytest.approx(1.0)


def test_agreement_tra_ve_nan_khi_khong_co_bai_nao_so_duoc():
    a = pd.Series(["khong_ro", ""])
    b = pd.Series(["vui", "buon"])
    ag, n = agreement(a, b)
    assert n == 0 and pd.isna(ag)


def test_kappa_thap_hon_ty_le_khop_khi_nhan_lech():
    """Nhãn lệch nặng: khớp cao nhưng kappa phải thấp, nếu không là tính sai."""
    a = pd.Series(["cang_thang"] * 9 + ["vui"])
    b = pd.Series(["cang_thang"] * 9 + ["buon"])
    ag, _ = agreement(a, b)
    assert ag == pytest.approx(0.9)
    assert cohen_kappa(a, b) < 0.5


# --------------------------------------------------------------------------- #
# Lỗi đã từng mắc: `mood_pred` là bản sao của `mood_label` ở bài mood_source='tag',
# nên nếu không lọc thì T2 ↔ T4 ra 100% và kappa 1,0 như hai hệ độc lập cùng kết luận.
# --------------------------------------------------------------------------- #

def _bang_mau() -> pd.DataFrame:
    return pd.DataFrame({
        "track_id": list("abcd"),
        "mood_label": ["vui", "buon", "", ""],
        "mood_pred": ["vui", "buon", "thu_gian", "cang_thang"],
        "mood_source": ["tag", "tag", "model", "model"],
    })


def test_loc_mood_source_chi_giu_bai_mo_hinh_that_doan():
    d = _bang_mau()
    pred_model = d["mood_pred"].where(d["mood_source"].eq("model"))
    assert pred_model.notna().sum() == 2
    assert list(pred_model.dropna()) == ["thu_gian", "cang_thang"]


def test_t2_va_t4_mo_hinh_loai_tru_nhau():
    """Không bài nào có cả hai -> mọi phép so T2 ↔ T4 phải ra n = 0, không ra 100%."""
    d = _bang_mau()
    d["mood_pred_model"] = d["mood_pred"].where(d["mood_source"].eq("model"))
    ag, n = agreement(d["mood_label"], d["mood_pred_model"])
    assert n == 0, "T2 và T4-mô-hình phải loại trừ nhau theo thiết kế"
    assert pd.isna(ag)


def test_khong_loc_thi_ra_100_phan_tram_gia_tao():
    """Chứng minh lỗi cũ: so thẳng mood_pred với mood_label ra 100% vô nghĩa."""
    d = _bang_mau()
    ag, n = agreement(d["mood_label"], d["mood_pred"])
    assert n == 2 and ag == pytest.approx(1.0)


def test_pairwise_sap_xep_giam_va_du_cap():
    d = pd.DataFrame({
        "mood_label": ["vui", "buon", "vui", "buon"],
        "mood_pred_model": ["vui", "buon", "buon", "buon"],
        "llm_ten": ["vui", "vui", "vui", "vui"],
    })
    out = pairwise(d, ["mood_label", "mood_pred_model", "llm_ten"])
    assert len(out) == 3                                  # C(3,2)
    khop = out["khớp_%"].dropna().tolist()
    assert khop == sorted(khop, reverse=True)

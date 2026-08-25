"""어디버려? — 모바일용 Streamlit 앱.

폰에서 촬영하거나 목록에서 직접 골라 배출 방법을 확인한다.
결과는 스크롤 없이 바로 보이도록 팝업(st.dialog)으로 띄운다.

데스크톱 데모는 app/app_pyqt6.py 를 쓴다.

실행:
    streamlit run app/app.py

폰에서 열기: 같은 Wi-Fi에 연결한 뒤 실행 시 표시되는 Network URL 접속
    (예: http://192.168.0.10:8501). 방화벽에서 8501 포트를 열어야 할 수 있다.
"""
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from PIL import Image

from configs.classes import CLASSES, CLASS_KOR_NAME
from configs.config import CKPT_DIR, CONF_HIGH, CONF_MID, RULES_PATH
from src.inference import WastePredictor

# ── 팔레트 (PyQt UI와 동일, 기준색 #167331) ────────────────────────────────────────
C_ACCENT = "#167331"
C_ACCENT_TINT = "#E8F3EC"
C_GREEN = "#2E9E52"
C_YELLOW = "#D97706"
C_YELLOW_TINT = "#FEF3C7"
C_RED = "#DC2626"
C_RED_TINT = "#FEE2E2"
C_TEXT = "#1B2A20"
C_MUTED = "#5F7267"
C_BORDER = "#DCE7E0"

LEVEL_STYLE = {
    "high": (C_GREEN, C_ACCENT_TINT, "신뢰도 높음"),
    "mid": (C_YELLOW, C_YELLOW_TINT, "확인 필요"),
    "low": (C_RED, C_RED_TINT, "인식 실패"),
}

# 후보를 신뢰도 구간별로 묶어 보여준다 (configs/config.py 임계값 기준)
CONF_BANDS = [
    (CONF_HIGH, "이 품목일 가능성이 높아요", C_GREEN),
    (CONF_MID, "가능성이 있어요", C_YELLOW),
    (0.0, "가능성이 낮아요", C_MUTED),
]

st.set_page_config(
    page_title="어디버려?",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
.block-container {{ max-width: 480px; padding: 0.8rem 1rem 3rem; }}
#MainMenu, footer, header {{ visibility: hidden; }}

.stButton > button,
[data-testid="stCameraInput"] button,
[data-testid="stFileUploader"] button {{
    width: 100%; min-height: 48px; border-radius: 10px;
    font-size: 1rem; font-weight: 700;
}}
.stTabs [data-baseweb="tab"] {{
    flex: 1; justify-content: center;
    font-size: 1rem; font-weight: 600; padding: 0.6rem 0;
}}
[data-testid="stFileUploaderDropzone"] {{ padding: 1rem; }}
img {{ border-radius: 10px; }}

/* 팝업 안에서도 라디오 항목이 손가락으로 누를 만큼 커야 한다 */
[data-testid="stDialog"] label {{ font-size: 0.95rem; padding: 0.15rem 0; }}

.app-header {{ display: flex; align-items: center; gap: 0.5rem; padding: 0.2rem 0 0.5rem; }}
.app-header .title {{ font-size: 1.5rem; font-weight: 800; color: {C_TEXT}; }}
.app-header .sub {{ font-size: 0.8rem; color: {C_MUTED}; }}

.card {{
    border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.7rem;
    background: #FFFFFF; border: 1px solid {C_BORDER};
}}
.card.tinted {{ border: none; }}
.tag {{
    display: inline-block; border-radius: 6px; padding: 0.2rem 0.7rem;
    font-size: 0.75rem; font-weight: 700; color: #fff;
}}
.result-name {{ font-size: 1.9rem; font-weight: 800; margin: 0.35rem 0 0.2rem; }}
.muted {{ color: {C_MUTED}; font-size: 0.86rem; line-height: 1.5; }}
.rule-title {{ font-size: 1rem; font-weight: 700; color: {C_TEXT}; }}
.rule-body {{ font-size: 0.95rem; color: {C_TEXT}; line-height: 1.65; }}
.band {{
    font-size: 0.78rem; font-weight: 700; margin: 0.5rem 0 0.1rem;
}}
</style>
""", unsafe_allow_html=True)


# ── 데이터 ─────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_rules():
    try:
        return json.loads(RULES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


@st.cache_resource(show_spinner=False)
def load_predictor(name):
    return WastePredictor(CKPT_DIR / f"{name}.pt")


def pick_model():
    """improved 를 우선 쓰고, 없으면 있는 체크포인트 아무거나."""
    for name in ("improved", "baseline"):
        if (CKPT_DIR / f"{name}.pt").exists():
            return name
    found = sorted(CKPT_DIR.glob("*.pt")) if CKPT_DIR.is_dir() else []
    return found[0].stem if found else None


RULES = load_rules()


def esc(text):
    return html.escape(str(text or ""))


def band_of(confidence):
    for threshold, label, color in CONF_BANDS:
        if confidence >= threshold:
            return label, color
    return CONF_BANDS[-1][1], CONF_BANDS[-1][2]


# ── 조각 렌더러 ────────────────────────────────────────────────────────────────────
def confidence_card(result):
    top = result["top1"]
    level = result["confidence_level"]
    color, tint, label = LEVEL_STYLE.get(level, (C_MUTED, "#F4F8F5", "-"))
    name = "인식하지 못했어요" if level == "low" else top["name_kor"]

    st.markdown(f"""
    <div class="card tinted" style="background:{tint};">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span class="tag" style="background:{color};">{label}</span>
        <span class="muted">신뢰도 {top['confidence']:.1%}</span>
      </div>
      <div class="result-name" style="color:{color if level == 'low' else C_TEXT};">{esc(name)}</div>
      <div class="muted">{esc(result['message'])}</div>
    </div>
    """, unsafe_allow_html=True)

    if result.get("low_data_warning"):
        st.markdown(f"""
        <div class="card tinted" style="background:{C_YELLOW_TINT};">
          <div style="font-weight:700;color:#92400E;font-size:0.9rem;">
            ⚠️ 학습 데이터가 적은 품목입니다</div>
          <div style="color:#92400E;font-size:0.84rem;margin-top:0.3rem;line-height:1.5;">
            오인식 가능성이 높으니 제품에 표시된 재질을 꼭 함께 확인하세요.</div>
        </div>
        """, unsafe_allow_html=True)


def rule_card(slug, source_label=""):
    """선택된 품목의 배출 방법. 팝업 하단에 놓인다."""
    rule = RULES.get(slug)
    kor = CLASS_KOR_NAME.get(slug, slug)
    if not rule:
        st.markdown(
            f'<div class="card"><div class="muted">{esc(kor)}의 배출 규칙이 없습니다.</div></div>',
            unsafe_allow_html=True)
        return

    notes = []
    if rule.get("needs_material_check"):
        notes.append("제품에 표시된 재질을 함께 확인하세요.")
    if rule.get("local_rule_required"):
        notes.append("배출 기준은 지역별로 다를 수 있어요. 거주 지자체 안내를 확인하세요.")
    if not rule.get("verified"):
        notes.append("공식 기준 검증 전인 초안입니다.")
    notes_html = "".join(
        f'<div class="muted" style="margin-top:0.35rem;">· {esc(n)}</div>' for n in notes)
    source_html = (f'<div class="muted" style="margin-top:0.6rem;font-size:0.76rem;">'
                   f'출처: {esc(rule["source"])}</div>') if rule.get("source") else ""
    header_note = (f'<div class="muted" style="margin-bottom:0.2rem;">{esc(source_label)}</div>'
                   if source_label else "")

    st.markdown(f"""
    <div class="card">
      {header_note}
      <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;">
        <span class="rule-title">🗑️ {esc(kor)}</span>
        <span class="tag" style="background:{C_ACCENT};">
          {esc(rule.get("disposal_category") or "조사 중")}</span>
      </div>
      <div class="rule-body" style="margin-top:0.5rem;">
        {esc(rule.get("instruction") or "배출 방법 데이터를 준비 중입니다.")}</div>
      {notes_html}{source_html}
    </div>
    """, unsafe_allow_html=True)


def manual_picker(key_prefix, expanded):
    """전체 18종에서 직접 고르기. 배출 분류로 한 번 걸러 목록을 짧게 만든다."""
    cats = sorted({RULES[s].get("disposal_category") or "기타"
                   for s in CLASSES if s in RULES})
    with st.expander("목록에서 직접 고르기", expanded=expanded):
        cat = st.selectbox("배출 분류", ["전체"] + cats, key=f"{key_prefix}_cat")
        items = [s for s in CLASSES
                 if cat == "전체" or (RULES.get(s, {}).get("disposal_category") or "기타") == cat]
        return st.radio(
            "품목",
            items,
            index=None,
            format_func=lambda s: CLASS_KOR_NAME.get(s, s),
            key=f"{key_prefix}_item",
        )


# ── 팝업 ───────────────────────────────────────────────────────────────────────────
@st.dialog("배출 안내")
def result_dialog():
    result = st.session_state.get("result")

    if result:
        confidence_card(result)

        # 후보를 신뢰도 구간으로 묶어 보여준다
        cands = result["candidates"]
        st.markdown('<div class="rule-title">이 중에 맞는 게 있나요?</div>',
                    unsafe_allow_html=True)
        labels = {}
        last_band = None
        for cand in cands:
            band, color = band_of(cand["confidence"])
            if band != last_band:
                st.markdown(
                    f'<div class="band" style="color:{color};">{esc(band)}</div>',
                    unsafe_allow_html=True)
                last_band = band
            labels[cand["class"]] = f"{cand['name_kor']}  ·  {cand['confidence']:.0%}"

        picked = st.radio(
            "후보",
            list(labels),
            index=0,
            format_func=labels.get,
            label_visibility="collapsed",
            key="cand_pick",
        )
        # 인식이 확실하지 않으면 직접 고르기를 미리 펼쳐 둔다
        manual = manual_picker("dlg", expanded=result["confidence_level"] == "low")
        slug = manual or picked
        note = "직접 고른 품목입니다." if manual else ""
    else:
        slug = st.session_state.get("manual_slug")
        note = "직접 고른 품목입니다."

    if slug:
        rule_card(slug, note)
    else:
        st.markdown('<div class="muted">품목을 고르면 배출 방법이 나옵니다.</div>',
                    unsafe_allow_html=True)


def run_prediction(image_file):
    """버튼을 눌렀을 때만 추론한다. 눌러야 도는 게 분명해야 기다릴 수 있다."""
    try:
        with st.spinner("모델 준비 중..."):
            predictor = load_predictor(model_name)
    except Exception as exc:
        st.error(f"모델을 불러오지 못했습니다: {exc}")
        return

    with st.spinner("사진을 살펴보는 중..."):
        st.session_state["result"] = predictor.predict(Image.open(image_file))

    st.session_state["manual_slug"] = None
    for key in ("cand_pick", "dlg_item", "dlg_cat"):
        st.session_state.pop(key, None)
    result_dialog()


# ── 메인 ───────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="app-header"><span style="font-size:1.6rem">♻️</span>'
    '<div><div class="title">어디버려?</div>'
    '<div class="sub">사진을 찍거나 목록에서 골라보세요</div></div></div>',
    unsafe_allow_html=True,
)

model_name = pick_model()
if model_name is None:
    st.error("학습된 모델이 없습니다. 먼저 `python -m src.train` 을 실행하세요.")
    st.stop()

tab_cam, tab_file, tab_list = st.tabs(["📷 촬영", "🖼️ 사진 선택", "📋 품목 직접 입력"])

with tab_cam:
    shot = st.camera_input("촬영", label_visibility="collapsed")
    if shot and st.button("결과 보기", key="go_cam", use_container_width=True,
                          type="primary"):
        run_prediction(shot)

with tab_file:
    upload = st.file_uploader(
        "사진 선택", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
    if upload:
        st.image(upload, use_container_width=True)
        if st.button("결과 보기", key="go_file", use_container_width=True,
                     type="primary"):
            run_prediction(upload)

with tab_list:
    st.markdown('<div class="muted" style="margin-bottom:0.5rem;">'
                '찾는 물건을 목록에서 고르세요.</div>', unsafe_allow_html=True)
    cats = sorted({RULES[s].get("disposal_category") or "기타"
                   for s in CLASSES if s in RULES})
    cat = st.selectbox("배출 분류", ["전체"] + cats, key="tab_cat")
    items = [s for s in CLASSES
             if cat == "전체" or (RULES.get(s, {}).get("disposal_category") or "기타") == cat]
    slug = st.selectbox("품목", items, key="tab_item",
                        format_func=lambda s: CLASS_KOR_NAME.get(s, s))
    if st.button("배출 방법 보기", key="go_list", use_container_width=True,
                 type="primary"):
        st.session_state["result"] = None
        st.session_state["manual_slug"] = slug
        result_dialog()

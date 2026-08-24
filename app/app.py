"""어디버려? — 모바일용 Streamlit 앱.

폰에서 촬영 → 인식 → Confidence 판단 → 배출 규칙 조회 → 배출 안내.

데스크톱 데모는 app/app_pyqt6.py 를 쓴다. 이쪽은 좁은 화면과 터치를 전제로
카메라를 기본 입력으로 두고, 한 화면에 결과가 다 들어오도록 구성했다.

실행:
    streamlit run app/app.py

폰에서 열기: 같은 Wi-Fi에 연결한 뒤 실행 시 표시되는 Network URL 접속
    (예: http://192.168.0.10:8501). 방화벽에서 8501 포트를 열어야 할 수 있다.
"""
import html
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from PIL import Image

from configs.config import CKPT_DIR
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

st.set_page_config(
    page_title="어디버려?",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── 모바일 레이아웃 ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* 폰 화면 기준으로 여백을 줄이고, 데스크톱에서도 폰 폭을 유지한다 */
.block-container {{
    max-width: 480px;
    padding: 0.8rem 1rem 4rem;
}}
#MainMenu, footer, header {{ visibility: hidden; }}

/* 터치 목표를 충분히 크게 (48px 이상) */
.stButton > button,
[data-testid="stCameraInput"] button,
[data-testid="stFileUploader"] button {{
    width: 100%;
    min-height: 48px;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 700;
}}
.stTabs [data-baseweb="tab"] {{
    flex: 1;
    justify-content: center;
    font-size: 1rem;
    font-weight: 600;
    padding: 0.6rem 0;
}}
/* 업로더가 폰에서 너무 높게 잡히는 것을 막는다 */
[data-testid="stFileUploaderDropzone"] {{ padding: 1rem; }}
img {{ border-radius: 10px; }}

.app-header {{
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.2rem 0 0.6rem;
}}
.app-header .title {{ font-size: 1.5rem; font-weight: 800; color: {C_TEXT}; }}
.app-header .sub {{ font-size: 0.8rem; color: {C_MUTED}; }}

.card {{
    border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 0.7rem;
    background: #FFFFFF; border: 1px solid {C_BORDER};
}}
.card.tinted {{ border: none; }}
.tag {{
    display: inline-block; border-radius: 6px; padding: 0.2rem 0.7rem;
    font-size: 0.75rem; font-weight: 700; color: #fff;
}}
.result-name {{ font-size: 2rem; font-weight: 800; margin: 0.4rem 0 0.2rem; }}
.muted {{ color: {C_MUTED}; font-size: 0.88rem; line-height: 1.5; }}
.rule-title {{ font-size: 1rem; font-weight: 700; color: {C_TEXT}; margin-bottom: 0.5rem; }}
.rule-body {{ font-size: 0.95rem; color: {C_TEXT}; line-height: 1.65; }}
.bar-bg {{
    background: {C_ACCENT_TINT}; border-radius: 5px; height: 8px; width: 100%;
    overflow: hidden; margin-top: 0.25rem;
}}
.bar-fill {{ height: 100%; border-radius: 5px; }}
.cand-row {{ margin-bottom: 0.6rem; }}
.cand-label {{
    display: flex; justify-content: space-between;
    font-size: 0.85rem; color: {C_TEXT};
}}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="app-header"><span style="font-size:1.6rem">♻️</span>'
    '<div><div class="title">어디버려?</div>'
    '<div class="sub">사진 한 장으로 배출 방법을 알려드려요</div></div></div>',
    unsafe_allow_html=True,
)


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


def esc(text):
    return html.escape(str(text or ""))


def render_result(result):
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
          <div style="font-weight:700;color:#92400E;font-size:0.92rem;">
            ⚠️ 학습 데이터가 적은 품목입니다
          </div>
          <div style="color:#92400E;font-size:0.85rem;margin-top:0.3rem;line-height:1.5;">
            오인식 가능성이 높으니 제품에 표시된 재질을 꼭 함께 확인하세요.
          </div>
        </div>
        """, unsafe_allow_html=True)

    rule = result.get("rule")
    if rule:
        notes = []
        if rule.get("needs_material_check"):
            notes.append("제품에 표시된 재질을 함께 확인하세요.")
        if rule.get("local_rule_required"):
            notes.append("배출 기준은 지역별로 다를 수 있어요. 거주 지자체 안내를 확인하세요.")
        if not rule.get("verified"):
            notes.append("공식 기준 검증 전인 초안입니다.")
        notes_html = "".join(
            f'<div class="muted" style="margin-top:0.35rem;">· {esc(n)}</div>' for n in notes)
        source_html = (f'<div class="muted" style="margin-top:0.6rem;font-size:0.78rem;">'
                       f'출처: {esc(rule["source"])}</div>') if rule.get("source") else ""

        st.markdown(f"""
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span class="rule-title">🗑️ 배출 안내</span>
            <span class="tag" style="background:{C_ACCENT};">
              {esc(rule.get("disposal_category") or "조사 중")}</span>
          </div>
          <div class="rule-body">
            {esc(rule.get("instruction") or "배출 방법 데이터를 준비 중입니다.")}</div>
          {notes_html}{source_html}
        </div>
        """, unsafe_allow_html=True)

    with st.expander("다른 후보 보기"):
        for i, cand in enumerate(result["candidates"]):
            color_i = C_ACCENT if i == 0 else C_MUTED
            st.markdown(f"""
            <div class="cand-row">
              <div class="cand-label">
                <span style="font-weight:{700 if i == 0 else 400};">{esc(cand['name_kor'])}</span>
                <span class="muted">{cand['confidence']:.1%}</span>
              </div>
              <div class="bar-bg">
                <div class="bar-fill" style="width:{cand['confidence'] * 100:.1f}%;
                     background:{color_i};"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ── 입력 ───────────────────────────────────────────────────────────────────────────
model_name = pick_model()
if model_name is None:
    st.error("학습된 모델이 없습니다. 먼저 `python -m src.train` 을 실행하세요.")
    st.stop()

st.caption("버리려는 물건 하나가 잘 보이도록 찍어 주세요.")

tab_cam, tab_file = st.tabs(["📷 촬영", "🖼️ 사진 선택"])
with tab_cam:
    shot = st.camera_input("촬영", label_visibility="collapsed")
with tab_file:
    upload = st.file_uploader(
        "사진 선택", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

image_file = shot or upload
if not image_file:
    st.stop()

image = Image.open(image_file)
if upload and not shot:
    st.image(image, use_container_width=True)

try:
    with st.spinner("모델 준비 중..."):
        predictor = load_predictor(model_name)
except Exception as exc:
    st.error(f"모델을 불러오지 못했습니다: {exc}")
    st.stop()

with st.spinner("인식 중..."):
    result = predictor.predict(image)

render_result(result)

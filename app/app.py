"""어디버려? Streamlit 웹 데모.

촬영/업로드 → 인식 → Confidence 판단 → 배출 규칙 조회 → 배출 안내.

실행:
    streamlit run app/app.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st
from PIL import Image

from configs.config import CKPT_DIR
from src.inference import WastePredictor

st.set_page_config(page_title="폐기물 배출 안내 AI", page_icon="♻️")


@st.cache_resource
def load_predictor(name):
    return WastePredictor(CKPT_DIR / f"{name}.pt")


st.title("♻️ 어디버려?")
st.caption("버리려는 물건 하나가 잘 보이도록 촬영해 주세요.")

model_name = st.sidebar.selectbox("모델", ["improved", "baseline"])
source = st.radio("입력 방식", ["이미지 업로드", "카메라 촬영"], horizontal=True)

image_file = (st.file_uploader("이미지 선택", type=["jpg", "jpeg", "png"])
              if source == "이미지 업로드" else st.camera_input("촬영"))

if image_file:
    image = Image.open(image_file)
    st.image(image, caption="입력 이미지", width=360)

    try:
        predictor = load_predictor(model_name)
    except FileNotFoundError:
        st.error(f"'{model_name}' 체크포인트가 없습니다. 먼저 학습을 실행하세요.")
        st.stop()

    result = predictor.predict(image)
    top = result["top1"]
    level = result["confidence_level"]

    if level == "high":
        st.success(f"**{top['name_kor']}** (신뢰도 {top['confidence']:.1%})")
    elif level == "mid":
        st.warning(f"**{top['name_kor']}** (신뢰도 {top['confidence']:.1%})")
    else:
        st.error("인식 실패")

    st.write(result["message"])

    rule = result["rule"]
    if rule:
        st.subheader("배출 안내")
        st.write(f"**배출 분류:** {rule['disposal_category'] or '조사 중'}")
        steps = rule["instruction"] or ["배출 방법 데이터를 준비 중입니다."]
        if isinstance(steps, str):  # instruction이 문자열이던 옛 형식도 받아준다
            steps = [steps]
        st.write("\n".join(f"- {step}" for step in steps))
        if rule["needs_material_check"]:
            st.info("제품에 표시된 재질을 함께 확인해 주세요.")
        if rule["local_rule_required"]:
            st.info("배출 기준은 지역별로 다를 수 있습니다. 거주 지자체 안내를 확인하세요.")
        if rule["source"]:
            st.caption(f"출처: {rule['source']}")

    with st.expander("Top-3 후보"):
        for candidate in result["candidates"]:
            st.write(f"- {candidate['name_kor']} — {candidate['confidence']:.1%}")

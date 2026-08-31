"""단일 이미지 추론 + Confidence 판단 + 배출 규칙 조회 (계획서 7, 8장).

사용:
    python -m src.inference --image sample.jpg --name baseline
"""
import argparse
import json

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from configs.classes import CLASSES, CLASS_KOR_NAME
from configs.config import (CKPT_DIR, CONF_HIGH, CONF_MID, LOW_DATA_CLASSES,
                            RULES_PATH, SEARCH_RECOMMENDED_CLASSES)
from src.model import get_device, load_checkpoint
from src.transforms import build_transforms


class WastePredictor:
    def __init__(self, ckpt_path, device=None):
        self.device = device or get_device()
        self.model, self.meta = load_checkpoint(ckpt_path, self.device)
        self.transform = build_transforms(train=False)
        self.rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))

    @torch.no_grad()
    def predict(self, image, topk=3):
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        # 폰으로 세로로 찍은 사진 상당수는 픽셀은 안 돌리고 EXIF 태그로만
        # "몇 도 돌려서 보라"고 표시한다. 이걸 무시하면 모델에는 옆으로
        # 눕거나 거꾸로 뒤집힌 이미지가 들어가 인식이 크게 나빠질 수 있다.
        image = ImageOps.exif_transpose(image)
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(tensor), dim=1)[0].cpu()

        values, indices = probs.topk(min(topk, len(CLASSES)))
        candidates = [{"class": CLASSES[i], "name_kor": CLASS_KOR_NAME[CLASSES[i]],
                       "confidence": round(float(v), 4)}
                      for v, i in zip(values, indices)]

        top = candidates[0]
        confidence = top["confidence"]

        if confidence >= CONF_HIGH:
            level, message = "high", "인식 결과에 따라 배출 방법을 안내합니다."
        elif confidence >= CONF_MID:
            level, message = "mid", "결과가 확실하지 않습니다. 물품의 재질 표시를 함께 확인하세요."
        else:
            level = "low"
            message = "물품을 정확히 인식하지 못했습니다. 다른 각도나 단색 배경에서 다시 촬영해 주세요."

        # 학습 데이터가 극히 적은 클래스는 confidence가 높아도 신뢰할 수 없다.
        # electric_fry_pan/fry_pan처럼 배출 분류가 다른 클래스를 잘못 안내하면
        # 실제 피해(잘못된 배출)로 이어지므로 high여도 mid로 낮춰 재질 확인을 강제한다.
        low_data = top["class"] in LOW_DATA_CLASSES
        if low_data and level == "high":
            level = "mid"
        if low_data:
            message = ("이 물품 종류는 학습 데이터가 적어 오인식 가능성이 높습니다. "
                       "물품의 재질 표시를 반드시 함께 확인하세요.")

        # 사진 인식 자체가 사실상 의미 없는 극소수 클래스는, 재질 확인 정도가
        # 아니라 목록에서 직접 찾도록 적극적으로 유도한다 (예: electric_fry_pan).
        search_recommended = top["class"] in SEARCH_RECOMMENDED_CLASSES
        if search_recommended:
            message = ("이 물품은 사진만으로는 정확히 인식하기 어렵습니다. "
                       "'품목 직접 입력'에서 찾아 확인하는 걸 추천합니다.")

        result = {
            "confidence_level": level,
            "message": message,
            "low_data_warning": low_data,
            "search_recommended": search_recommended,
            "top1": top,
            "candidates": candidates,
            "rule": None,
        }
        if level != "low":
            result["rule"] = self.rules.get(top["class"])
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    predictor = WastePredictor(CKPT_DIR / f"{args.name}.pt")
    result = predictor.predict(args.image)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# 모델 한계와 개선 방향

improved 모델(Accuracy 92.2%, Macro F1 0.86) 학습·평가 과정에서 확인한
문제와, 각각에 대해 실제로 해본 것/보류한 것/앞으로 할 것을 정리한다.

## 1. 학습 곡선 — epoch을 늘려서 해결될 문제가 아니다

`improved` 학습(15 epoch, layer4 unfreeze)에서 best 체크포인트는 **5~6
epoch**에서 나왔다. 그 뒤로도 계속 돌렸지만:

- train accuracy는 99.8%까지 올라갔다 (사실상 다 외움)
- val accuracy는 92~93%에서 정체되거나 오히려 떨어졌다

**결론: epoch을 더 늘려도 안 나아진다.** 전형적인 overfitting이고, best
체크포인트는 이미 자동 저장되므로(`src/train.py`) 손해는 없다. 근본
원인은 아래 3번 항목이다.

## 2. 데이터는 AI-Hub 원본을 손실 없이 다 썼다

`prepare_raw.py`가 원본 폴더 구조를 그대로 파싱해 train/val/test로
나눴고, `--limit`으로 클래스 불균형만 완화했다(최대 40배 → 6배).
데이터를 더 못 넣어서 정확도가 낮은 게 아니다 — 아래 3번이 진짜 원인.

## 3. 근본 원인: 실사용 사진과 학습 데이터의 도메인이 다르다

AI-Hub 원본은 촬영 조건이 통제된 사진들이다. 실제 사용자가 폰으로
찍을 사진(작은 물건을 손으로 들고 찍는 경우, 어수선한 배경, 다양한
조명)과는 상당히 다를 것으로 예상되고, 이게 학습 데이터에 아예 없다.
**val accuracy가 92%대에서 못 올라가는 진짜 이유는 데이터 양이 아니라
이 도메인 차이일 가능성이 높다.**

이건 지금 가진 데이터로는 검증도 해결도 안 된다 — 실제로 사용자가
찍을 법한 사진을 모아서 확인해야 한다 (아래 "다음 할 일" 참고).

## 4. 마스킹(배경 제거)을 검토했으나 기각했다

"물체 영역만 잘라내면 배경 잡음이 줄어 정확도가 오르지 않을까" 라는
아이디어를 재학습 없이 먼저 실측했다 — YOLOv8n(COCO 사전학습, 80종)으로
가장 큰 물체를 찾아 크롭한 뒤 기존 분류기에 넣어 confidence를 비교.

**결과 (electric_fry_pan/electric_iron/vacuum_cleaner/fry_pan/pet_bottle
30장):**

| 지표 | 값 |
|---|---|
| YOLO 탐지 성공률 | 43% (13/30) |
| 탐지됐을 때 실제 라벨 | cup, bowl, scissors, skateboard, fire hydrant 등 — 전부 오답 |
| 크롭 후 confidence 상승 (+0.02 이상) | 4장 |
| 크롭 후 confidence 하락 (-0.02 이상) | 6장 |

이미 100% 확신하던 `fry_pan` 샘플이 크롭 후 `mobile_phone`(69.5%)으로
바뀌는 등, **잘 되던 것까지 망가뜨렸다.** 원인은 명확하다 — COCO 80종
안에 프라이팬·다리미·청소기 같은 생활폐기물이 없어서, 범용 탐지기가
엉뚱한 걸 그 물체로 착각해 자른다.

**결론: 마스킹은 지금 방식(범용 COCO 탐지기)으로는 안 된다.** 하려면
이 18종에 대한 bbox 라벨을 새로 만들어 커스텀 탐지기를 학습시켜야
하는데, 그럴 데이터가 없다. 이 방향은 보류한다.

## 5. 극소수 클래스는 사진 인식을 포기하고 목록으로 유도한다

`electric_fry_pan`은 원본이 41건(205장)뿐이라 test 10장 중 2장만
맞힌다(recall 0.20). 데이터를 늘릴 수 없는 상황에서 이 클래스는 사진
인식 자체를 밀어붙이지 않고, 예측되면 바로 "품목 직접 입력" 목록으로
안내하도록 앱에 반영했다 (`SEARCH_RECOMMENDED_CLASSES`,
[configs/config.py](../configs/config.py)).

단, 이 안전장치는 **모델이 실제로 `electric_fry_pan`이라고 답했을
때만** 작동한다. 나머지 8장처럼 `sealed_container`/`fry_pan` 등으로
완전히 잘못 분류되는 경우는 예측 클래스 자체가 다르므로 막을 방법이
없다 — 이건 사후 UI로 해결 가능한 문제가 아니다.

## 6. DINOv2를 검토했으나, 이번에도 재학습/파이프라인 변경 없이 기각했다

"극소수 클래스는 ResNet18 대신 DINOv2(자기지도학습 기반 비전 파운데이션
모델) 임베딩 + 별도 분류기로 판단하면 어떨까"를 실측했다. DINOv2는
backbone을 완전히 얼려두고 그 위에 가벼운 분류기만 학습시키는 방식이라
파라미터가 적어 소량 데이터에서 과적합이 덜하다는 게 아이디어의 근거였다.

**1차 결과(18-way 선형분류기, test 8,664장 전체):**

| | improved(ResNet18) | DINOv2(vits14, frozen)+선형분류기 |
|---|---|---|
| 전체 Accuracy | 92.2% | 87.1% |
| Macro F1 | 0.86 | 0.81 |
| `electric_fry_pan` recall | 0.20 | 0.80 |
| `electric_iron` recall | 0.53 | 0.76 |

약한 두 클래스의 recall만 보면 극적으로 좋아 보였다. 하지만 **precision을
보니 얘기가 달라졌다** — `electric_fry_pan`으로 예측된 25건 중 실제
정답은 7건뿐(precision 0.28), `electric_iron`도 96건 중 41건뿐(precision
0.43). 나머지는 멀쩡한 `sealed_container`·`vacuum_cleaner`를 이 클래스로
잘못 우긴 것이다. 18개 클래스가 하나의 소프트맥스에서 경쟁하다 보니 이
두 클래스 쪽으로 쏠리는 편향이 생긴 것으로 보인다.

**2차로 이진분류기(해당 클래스 vs 나머지 전체, `class_weight="balanced"`)
로 바꿔 precision-recall 곡선을 threshold별로 다시 봤다:**

| threshold | electric_fry_pan (precision/recall) | electric_iron (precision/recall) |
|---|---|---|
| 0.70 | 0.30 / 0.60 | 0.50 / 0.73 |
| 0.90 | 0.42 / 0.50 | 0.54 / 0.69 |
| 0.99 | 1.00 / **0.20** | 0.65 / 0.58 |

`electric_fry_pan`은 precision을 100%까지 올리면 recall이 ResNet과
똑같은 0.20으로 떨어진다 — **얻는 게 없다.** test가 10장뿐이라 애초에
통계적으로 신뢰할 만한 threshold를 고를 근거 자체가 부족하다.
`electric_iron`은 그나마 낫지만(threshold 0.95에서 precision 0.63/
recall 0.67), 이대로 배포하면 **멀쩡한 물건 3개 중 1개를 전기다리미로
잘못 안내**하게 되어 마스킹 실험 때와 같은 실수(잘 되던 것까지
망가뜨림)를 반복하는 셈이다.

**결론: DINOv2 방향 자체는 유망하지만(recall 개선은 실제로 확인됨),
지금 데이터양(electric_fry_pan test 10장, electric_iron test 55장)으로는
정밀도와 재현율을 동시에 잡을 근거가 부족하다. 배포하지 않고 기각한다.**
현재 배포된 `SEARCH_RECOMMENDED_CLASSES` 안내가 여전히 가장 안전한
완화책이다. 나중에 electric_fry_pan/electric_iron 실사진을 더 모으면
재검토할 가치는 있다 (DINOv2 학습 자체는 재학습이 필요 없어 비용이
크지 않다 — 임베딩 재추출 + 분류기 재학습에 수 분 수준).

## 다음 할 일 (실사진 수집 — 자동화 불가, 팀 작업 필요)

이 문서의 3번(도메인 차이)이 핵심 병목이고, 검증·해결 둘 다 **실사용
환경과 비슷한 사진**이 있어야 가능하다.

1. 팀원들이 실제로 폰 카메라로 18종 물건을 찍어 50~100장 모은다
   (다양한 배경·각도·손으로 든 경우 포함)
2. 지금 모델(`improved.pt`)로 그 사진들을 먼저 평가해서, 정말 도메인
   차이 때문에 성능이 떨어지는지 확인한다
3. 확인되면 그 사진들로 fine-tuning 하거나, 최소한 학습 데이터 증강에
   반영한다 (지금도 RandomResizedCrop/Flip/ColorJitter는 기본
   적용 중 — [src/transforms.py](../src/transforms.py))

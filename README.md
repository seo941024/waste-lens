# 어디버려? — 생활용품 이미지 인식 기반 폐기물 배출 안내 AI

> waste-lens · 사진 한 장으로 생활용품을 인식하고, 신뢰도를 판단한 뒤 배출 방법을 안내하는 웹 MVP.

"이거 어디에 버려요?" 라는 질문에 답하는 것이 이 프로젝트의 목표입니다.
재질을 분류하는 것이 아니라, 버리기 헷갈리는 물건의 **배출 방법을 안내**합니다.

## 구조

```
configs/   classes.py(18개 클래스 정의), config.py(경로·하이퍼파라미터)
src/       inspect_source, prepare_raw, dataset, transforms,
           model, train, evaluate, inference
rules/     disposal_rules.json  (배출 규칙 DB — 공식 기준 조사 후 채움)
app/       app.py (Streamlit 데모)
results/   체크포인트, 평가 리포트, Confusion Matrix
data/      train/ val/ test/  (Git 제외)
```

## 준비

```bash
git clone https://github.com/seo941024/waste-lens.git
```

```bash
cd waste-lens
```

```bash
pip install -r requirements.txt
```

저장소에는 **코드만** 들어 있습니다. `data/`와 `results/checkpoints/`는 Git에서 제외되어 있으므로
이미지와 학습된 모델은 포함되지 않습니다. AI-Hub 「생활 폐기물 이미지」(dataSetSn=140)를
[aihub.or.kr](https://www.aihub.or.kr)에서 직접 내려받은 뒤 아래 2~5단계를 순서대로 실행하세요.
AI-Hub 데이터는 재배포가 금지되어 있어 저장소에 담을 수 없습니다.

GPU가 없어도 동작하지만 학습은 CPU에서 매우 느립니다. CUDA 환경이라면
[pytorch.org](https://pytorch.org/get-started/locally/)에서 GPU용 torch 설치 명령을 확인해
`requirements.txt` 대신 먼저 설치하세요.

## 데이터에 대해

AI-Hub에서 받아야 할 것은 **원천데이터(이미지)뿐**입니다. 라벨링데이터는 필요 없습니다.
이 프로젝트는 이미지 *분류*이고 정답은 폴더명에서 나오기 때문입니다.
라벨 JSON에 든 bbox 좌표는 객체 *검출*용이라 여기서는 쓰지 않습니다.

압축을 풀면 구조가 이렇습니다. 셋 다 중요합니다.

```
생활 폐기물 이미지/
  Training/[T원천]도기류_화분_화분/14_X001_C014_1209/14_X001_C014_1209_0.jpg
                                   └ 건(물체) 폴더    └ 같은 물체 5장 (_0~_4)
  Validation/[V원천]도기류_화분_화분/...
```

- **품목명은 폴더명 마지막 토큰**입니다 (`[T원천]<중분류>_<품목>_<품목>`).
- **건(물체) ID는 이미지의 부모 폴더명**입니다. 같은 건 5장이 train과 test로 흩어지면
  같은 물체를 외운 채 맞히는 셈이라 정확도가 가짜로 부풀려집니다. 항상 건 단위로 묶습니다.
- **Training/Validation이 이미 나뉘어** 있어 별도 분할이 필요 없습니다.

## 1. 원본 구조 파악 (선택)

구조가 위와 다르거나 확인만 하고 싶을 때 씁니다. 아무것도 옮기지 않습니다.

```bash
python -m src.inspect_source --source "C:/Users/이름/Downloads/생활 폐기물 이미지"
```

파일 종류별 개수, 폴더 구조, 안 풀린 압축, 1건 5장 파일명 패턴을 보여줍니다.

## 2. data/ 로 정리

먼저 `--dry-run`으로 클래스별 장수를 확인합니다. 아무것도 저장하지 않습니다.

```bash
python -m src.prepare_raw --source "C:/Users/이름/Downloads/생활 폐기물 이미지" --dry-run
```

문제없으면 `--dry-run`을 빼고 실행합니다. Training은 그대로 `data/train`이 되고,
AI-Hub가 test셋을 따로 주지 않으므로 **Validation을 건 단위로 반씩 갈라** `data/val`과 `data/test`로 씁니다.

```bash
python -m src.prepare_raw --source "C:/Users/이름/Downloads/생활 폐기물 이미지" --limit 1200
```

`--limit`은 **클래스당 최대 건수**입니다(장수가 아닙니다. 1건 = 5장).
이 데이터셋은 불균형이 40배까지 납니다 — 페트병 36,961장 vs 전기다리미 913장.
그대로 학습하면 모델이 페트병만 찍어도 점수가 나오므로 `--limit`을 쓰는 편이 좋습니다.

`--max-side`(기본 512)는 저장 시 이미지를 줄여둡니다. 학습은 224px로 하는데
원본을 그대로 두면 매 epoch마다 큰 JPEG를 디코딩하느라 GPU가 놉니다.
결과 파일명은 `flower_pot-14_X001_C014_1209_01.jpg` 형태입니다.

## 3. 학습

```bash
python -m src.train --name baseline --no-augment --epochs 10
```

```bash
python -m src.train --name improved --unfreeze-layer4 --epochs 15 --lr 1e-4
```

기본값은 batch 64, workers 6, bf16 AMP입니다. RTX 3080 10GB 실측 기준
batch 64는 VRAM 1.03GB(10%)만 쓰므로 공유 GPU 메모리로 넘어가지 않습니다.

**Windows에서는 첫 epoch가 1분쯤 걸립니다.** DataLoader 워커 프로세스를 띄우는
일회성 비용이고, 2번째 epoch부터는 정상 속도가 나옵니다 (실측 62.8s → 6.4s).
고장이 아니니 기다리세요. 문제가 생기면 `--workers 0`으로 끄되 4배 이상 느려집니다.

## 4. 평가

```bash
python -m src.evaluate --name improved
```

Accuracy / Precision / Recall / F1, Confusion Matrix 이미지, Threshold별 Coverage·Accepted Accuracy, High-confidence wrong prediction 목록이 `results/`에 저장됩니다.

### 알려진 한계: `electric_fry_pan`

AI-Hub 원본 자체가 41건(205장)뿐이라 재학습으로 해결되지 않습니다.
`improved` 기준 test 10장 중 2장만 정답이고, 나머지는 대부분 `sealed_container`로
높은 confidence와 함께 오분류됩니다 — confidence 임계값으로도 못 걸러냅니다.
모델이 `electric_fry_pan`이라고 답한 경우에 한해 `configs/config.py`의
`LOW_DATA_CLASSES`가 confidence를 낮추고 재질 확인을 안내하지만, 애초에
`sealed_container`로 오분류된 경우는 이 안전장치가 걸리지 않습니다.
전기프라이팬은 촬영을 피하거나, 결과가 이상하면 재질 표시를 직접 확인하세요.

## 5. 추론 · 웹 데모

```bash
python -m src.inference --image sample.jpg --name improved
```

```bash
streamlit run app/app.py
```

## 주의

`rules/disposal_rules.json`의 `instruction`, `source`는 비어 있습니다.
환경부·지자체 등 공식 기준을 확인해 채우고 `verified`를 `true`로 바꾼 뒤 서비스에 사용하세요.
`disposal_category`는 AI-Hub 라벨 계열을 참고한 초안이며 실제 배출 기준이 아닙니다.

# 어디버려? — 생활용품 이미지 인식 기반 폐기물 배출 안내 AI

> waste-lens · 사진 한 장으로 생활용품을 인식하고, 신뢰도를 판단한 뒤 배출 방법을 안내하는 웹 MVP.

"이거 어디에 버려요?" 라는 질문에 답하는 것이 이 프로젝트의 목표입니다.
재질을 분류하는 것이 아니라, 버리기 헷갈리는 물건의 **배출 방법을 안내**합니다.

## 구조

```
configs/   classes.py(18개 클래스 정의), config.py(경로·하이퍼파라미터)
src/       inspect_source, prepare_raw, split_data, dataset, transforms,
           model, train, evaluate, inference
rules/     disposal_rules.json  (배출 규칙 DB — 공식 기준 조사 후 채움)
app/       app.py (Streamlit 데모)
results/   체크포인트, 평가 리포트, Confusion Matrix
data/      raw/ train/ val/ test/  (Git 제외)
```

## 준비

```bash
pip install -r requirements.txt
```

## 1. 원본 구조 파악

AI-Hub 「생활 폐기물 이미지」(dataSetSn=140)는 **폴더가 중분류(고철류·전자제품 등) 단위**이고,
우리가 필요한 세부 품목명(프라이팬 등)은 **라벨 JSON 안**에 있습니다.
또한 **1건 = 같은 물체 5장**이므로 데이터 누수 방지가 필수입니다.

다만 라벨이 JSON인지, 품목명이 어디 있는지는 **직접 열어봐야** 압니다.
다운로드/압축해제가 끝나면 먼저 조사만 합니다 (아무것도 옮기지 않습니다).

```bash
python -m src.inspect_source --source "D:/aihub"
```

파일 종류별 개수, 폴더 구조, 안 풀린 압축, 1건 5장 파일명 패턴,
라벨 형식(JSON/XML/CSV)과 키별 값 분포를 보여줍니다.
출력에서 ① 세부 품목명 키 ② 건 ID 키 ③ bbox 키를 확인하세요.

## 2. data/raw 로 정리

확인한 키 이름을 넣고 먼저 `--dry-run`으로 장수를 봅니다.

```bash
python -m src.prepare_raw --source "D:/aihub/생활폐기물" --mode json --detail-key DETAILS --group-key ID --bbox-key BOX --crop-bbox --dry-run
```

문제없으면 `--dry-run`을 빼고 실행합니다. 클래스당 장수를 제한하려면 `--limit 1000`을 붙이세요.
`--crop-bbox`는 1920×1080 원본에서 물체 영역만 잘라내 분류 정확도를 높입니다.

`--max-side`(기본 512)는 저장 시 이미지를 줄여둡니다. 학습은 224px로 하는데
원본을 그대로 두면 매 epoch마다 큰 JPEG를 디코딩하느라 GPU가 놉니다.
실측 결과 1920×1080 → 512px 축소만으로 **디코딩이 2.6배** 빨라집니다.
결과 파일명은 `fry_pan-g00001_01.jpg` 형태로, `g00001`이 건(물체) ID입니다.

## 3. 데이터 분할 (70/15/15)

```bash
python -m src.split_data --group-by-prefix --prefix-sep _
```

`--group-by-prefix`는 같은 건 ID의 5장을 한 split에 묶어 **데이터 누수를 막습니다. 반드시 사용하세요.**
빼고 돌리면 같은 물체가 train과 test에 동시에 들어가 정확도가 가짜로 부풀려집니다.

## 4. 학습

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

## 5. 평가

```bash
python -m src.evaluate --name improved
```

Accuracy / Precision / Recall / F1, Confusion Matrix 이미지, Threshold별 Coverage·Accepted Accuracy, High-confidence wrong prediction 목록이 `results/`에 저장됩니다.

## 6. 추론 · 웹 데모

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

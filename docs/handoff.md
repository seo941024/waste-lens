# waste-lens 인수인계 문서

다른 컴퓨터에서 바로 이어서 작업할 수 있도록 현재 상태와 다음 할 일을 정리한다.
마지막 갱신: 2026-09-05 · 브랜치 `jisu` · 최신 커밋 `1bd8356`

---

## 0. 가장 먼저 — 환경 설정

**Python 3.11을 써야 한다.** 이 프로젝트의 패키지(torch, PyQt6, streamlit,
ultralytics)는 전부 3.11에 설치돼 있고, PATH에 따라 `python`이 3.10을 가리켜
`ModuleNotFoundError`가 나는 경우가 있다.

```bash
# 확인
python -c "import PyQt6, torch, streamlit, ultralytics; print('OK')"

# 위가 실패하면 전체 경로로 실행
"C:\Users\AISW_203_106\AppData\Local\Programs\Python\Python311\python.exe" -c "import PyQt6; print('OK')"
```

Git Bash에서 세션 내내 3.11을 쓰려면:

```bash
export PATH="/c/Users/AISW_203_106/AppData/Local/Programs/Python/Python311:/c/Users/AISW_203_106/AppData/Local/Programs/Python/Python311/Scripts:$PATH"
```

의존성 설치는 `pip install -r requirements.txt`.
데이터(`data/`)와 체크포인트(`results/checkpoints/`)는 git에 없으므로
별도로 옮기거나 재생성해야 한다 (§4 참고).

---

## 1. 프로젝트 구성 — 지금 두 갈래다

| | 개인용 배출 안내 앱 (기존) | 선별장 벨트 QA (신규) |
|---|---|---|
| 위치 | `app/`, `src/`, `configs/`, `rules/` | `belt_qa/` |
| 목적 | 사진 1장 → 이게 뭔지 맞히고 배출법 안내 | 컨베이어벨트 위 미분류 폐기물 실시간 경고 |
| 문제 유형 | 단일 물체 분류 (18종) | 다중 물체 탐지 + 추적 |
| 모델 | ResNet18 fine-tuning (`improved.pt`) | YOLO 탐지 (아직 학습 전) |
| 상태 | **완성, 동작함** (Accuracy 92.2%) | **파이프라인 골격만 있음** |

기존 앱은 그대로 두고 벨트 QA를 추가하는 방향으로 결정했다. 발표 때는
"개인용 앱(완성) + 산업 규모 확장(진행중)"으로 묶어서 설명한다.

---

## 2. 기존 앱 — 실행 방법

```bash
# PyQt6 데스크톱
python app/app_pyqt6.py

# Streamlit 모바일 (폰에서 보려면 같은 Wi-Fi에서 Network URL 접속)
python -m streamlit run app/app.py
```

### 화면 구성

**PyQt6** (`app/app_pyqt6.py`) — 사이드바 5개 페이지
1. **배출 안내** — 드래그&드롭/파일선택 → 인식. 결과는 `요약 / 배출 안내 /
   다른 후보` 섹션 탭으로 나뉜다 (긴 안내문이 잘리지 않게). 왼쪽 아래에
   "목록에서 직접 찾기" 카드가 있어 사진 없이도 조회 가능.
2. **배출 규칙** — 18종 카드 목록
3. **배출 요일** — 시/도 > 구 선택 (서울 4개 구)
4. **모델 성능** — `results/*_eval.json` 읽어서 표시
5. **설정** — 체크포인트 선택, 임계값, 알려진 한계

**Streamlit** (`app/app.py`) — 모바일 전용, 4개 탭
`📷 촬영 / 🖼️ 사진 선택 / 📋 품목 직접 입력 / 🗓️ 배출 요일`
결과는 팝업(`st.dialog`)으로 뜬다.

> 폰에서 `📷 촬영` 탭은 HTTP라 브라우저가 카메라를 막는다.
> `🖼️ 사진 선택`을 쓰면 폰에서 "사진 촬영" 메뉴가 떠서 우회된다.
> HTTPS가 필요하면 `cloudflared tunnel --url http://localhost:8501`.

---

## 3. 기존 앱 — 코드 지도

```
configs/config.py        경로, 하이퍼파라미터, 임계값, 특수 클래스 목록
configs/classes.py       18종 클래스 정의 + 한글명
src/prepare_raw.py       AI-Hub 원본 → data/train,val,test 정리
src/train.py             학습 (--name, --unfreeze-layer4, --epochs, --lr)
src/evaluate.py          평가 → results/{name}_eval.json + confusion matrix
src/inference.py         WastePredictor — 추론 + confidence 판단 + 규칙 조회
src/model.py             ResNet18 구성 (ImageNet 사전학습 + fc 교체)
src/transforms.py        전처리/증강
rules/disposal_rules.json    18종 배출 방법 (verified: false 초안)
rules/collection_days.json   지역별 배출요일 (서울 4개 구)
app/app_pyqt6.py         PyQt6 데스크톱 UI
app/app.py               Streamlit 모바일 UI
docs/model_limitations.md    모델 한계·실패한 실험 기록 (중요)
```

### `configs/config.py`의 특수 목록 두 개

```python
LOW_DATA_CLASSES = {"electric_fry_pan", "electric_iron"}
# 데이터가 적어 오인식 위험 → confidence 높아도 mid로 낮추고 재질 확인 안내

SEARCH_RECOMMENDED_CLASSES = {"electric_fry_pan"}
# recall 0.20으로 사진 인식이 사실상 무의미 → 목록 직접 입력으로 유도
```

---

## 4. 데이터·모델 재생성 (새 컴퓨터에서 필요)

`data/`와 `results/checkpoints/`는 git에 없다.

```bash
# 1) AI-Hub 원본을 받아서 정리
python -m src.prepare_raw --src "<원본경로>" --limit 1200

# 2) 학습 (baseline 10분, improved 30분 정도 @ RTX 3080)
python -m src.train --name baseline --no-augment --epochs 10
python -m src.train --name improved --unfreeze-layer4 --epochs 15 --lr 1e-4

# 3) 평가
python -m src.evaluate --name improved
```

학습 결과 요약: baseline 85.6% → **improved 92.2%** (Macro F1 0.86).
best 체크포인트는 5~6 epoch에서 나오고 그 뒤로는 과적합(train 99.8% vs val 92%).

---

## 5. 이미 시도했다가 **기각한 것들** (다시 하지 말 것)

`docs/model_limitations.md`에 근거가 전부 정리돼 있다. 요약:

| 시도 | 결과 | 이유 |
|---|---|---|
| YOLO(COCO) 마스킹으로 배경 제거 | 기각 | 탐지율 43%, COCO 80종에 폐기물이 없어 엉뚱하게 잘림. 잘 되던 것까지 망가뜨림 |
| DINOv2 하이브리드 (저데이터 클래스 대체) | 기각 | recall은 오르나 precision 0.28~0.43. 멀쩡한 물건을 오인식 |
| DINOv2 이진분류기 + threshold 조정 | 기각 | precision 올리면 recall이 ResNet과 같아짐. test 표본이 10장뿐이라 근거 부족 |

**단, DINOv2 자체는 실사진에서 ResNet18보다 confidence가 안정적이었다.**
데이터가 더 모이면 재검토할 가치는 있다 (임베딩 재추출 + 분류기 재학습에
수 분이면 됨).

### 실사진 테스트 결과 (인터넷에서 구한 실제 사진 7장)

두 모델 모두 top1은 7/7 정답. 차이는 confidence뿐이었다.
ResNet18은 2건이 0.55 미만이라 "재촬영해주세요"가 떴다(정답인데도).

### 고친 실제 버그

- **EXIF 방향 미보정** (`src/inference.py`) — 폰 세로사진이 회전된 채로
  모델에 들어가던 문제. `ImageOps.exif_transpose()` 추가로 해결. 실사용
  오인식의 원인 중 하나였을 가능성이 높다.
- **Streamlit 마크다운 들여쓰기** (`app/app.py`) — 중첩 f-string의 들여쓰기가
  4칸 넘으면 CommonMark가 코드블록으로 오인식해 HTML이 그대로 노출됐다.
  한 줄 문자열로 조립해 해결.

---

## 6. 신규 — 선별장 벨트 QA 시스템 (`belt_qa/`)

### 요구사항

폐기물 처리장에서 사람이 수작업으로 분류하는 과정에서, 작업자가 **놓친
플라스틱**이 벨트를 타고 계속 흘러가는 것을 카메라가 잡아낸다.

- 카메라 위치: 작업 라인 **중간과 끝** 구간, 벨트를 **천장에서 수직으로 내려다봄**
- 미분류 폐기물 감지 시: 화면에 **라벨 박스** 표시 + **경고 알림음**
- 그 물체가 **버려질 때까지** 박스와 경고를 계속 유지
- 물체가 사라지면(치워짐) 해당 경고만 해제

### 현재 상태

`belt_qa/pipeline.py` — **골격 완성, 동작 검증됨**

```python
class BeltWorker(QThread):
    frame_ready    = pyqtSignal(object)      # 박스 그려진 프레임
    alert_started  = pyqtSignal(int, str)    # (track_id, label) 새 미분류 발견
    alert_cleared  = pyqtSignal(int)         # (track_id) 치워짐
    finished_source = pyqtSignal()
```

동작: `YOLO.track(persist=True)`로 프레임마다 탐지+추적 → 대상 클래스면
추적 ID별로 `TrackedAlert` 유지 → 안 보이는 프레임이 `MISSING_GRACE_FRAMES`
(기본 10) 넘으면 해제.

**임시 설정 (학습 후 교체 필요):**
```python
PLACEHOLDER_MODEL = "yolov8n.pt"                    # → 학습된 가중치로 교체
PLACEHOLDER_TARGET_CLASSES = {"bottle", "cup"}      # → 실제 클래스명으로 교체
```

### 검증 방법 (실제 벨트 영상이 없어서)

실제 사진을 벨트 배경 위로 흘려보내 **합성 영상**을 만들어 테스트했다.
스크립트: 스크래치패드의 `make_belt_video.py`, `test_belt_pipeline.py`
(임시 파일이므로 새 컴퓨터에는 없음 — 필요하면 재작성)

검증 결과: 240프레임 처리, 탐지·경고 시작·해제 모두 정상 동작.

### ⚠ 알려진 문제 — 추적 ID churn

같은 물체가 ID를 잃었다가 새 ID로 재등록되는 현상이 있다
(`track#3` → 해제 → 다시 `#3`, `#10`...). 실제로는 같은 쓰레기인데
경고가 반복해서 새로 울리게 된다.

원인 후보:
1. 합성 영상이 부자연스러워서 생긴 artifact일 가능성 (실제 영상으로 재확인 필요)
2. `conf` 임계값이 낮아 탐지가 깜빡임
3. `MISSING_GRACE_FRAMES`가 너무 짧음

**실제 벨트 영상을 확보한 뒤 먼저 재현되는지 확인할 것.**

---

## 7. 다음 할 일 (우선순위 순)

### 7-1. AI-Hub 데이터 확보 (병목 — 사람이 해야 함)

두 데이터셋 중 하나를 신청한다. 캔/플라스틱이 별도 클래스로 나뉘어 있고
bbox 라벨이 있어 바로 학습 가능하다.

| 데이터셋 | 특징 |
|---|---|
| [재활용품 분류 및 선별 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71362) | 선별영상 추출 29.9만장, bbox+segmentation, 서울·경기 실제 선별센터. 전체 2,876GB(너무 큼 — 캔/플라스틱만 골라 받을 것) |
| [생활폐기물 데이터 활용·환류](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71385) | 선별장 촬영분만 78GB, bbox, 29개 세부 클래스 |

절차: AI-Hub 회원가입 → 활용 신청 → 승인 → API 다운로드(분할 압축).

> 해외 공개 데이터셋(ZeroWaste, SpectralWaste)도 검토했으나 부적합.
> ZeroWaste는 65TB에 학술기관 이메일 승인 필요, 클래스에 캔이 없음.
> SpectralWaste는 클래스가 film/basket/cardboard 등이라 안 맞음.

### 7-2. YOLO 탐지 모델 학습

데이터가 들어오면 클래스를 **플라스틱 / 캔 2종**으로 정리해 학습한다.
데이터셋 폴더 구조는 YOLO 형식(`images/`, `labels/`)으로 변환 필요.

```bash
yolo detect train data=<데이터셋.yaml> model=yolov8m.pt epochs=50 imgsz=640
```

학습 후 `belt_qa/pipeline.py`의 `PLACEHOLDER_MODEL`,
`PLACEHOLDER_TARGET_CLASSES`를 교체한다.

### 7-3. 벨트 QA UI 만들기 (`belt_qa/app.py` — 아직 없음)

필요한 것:
- 영상 패널 (`frame_ready` 신호로 받은 프레임 표시)
- 경고 배너/팝업 (`alert_started`/`alert_cleared`로 켜고 끔)
- **경고음** — Windows면 `winsound.Beep()` 또는 `QSoundEffect` 반복 재생.
  경고가 하나라도 살아있는 동안 계속, 전부 해제되면 정지
- 영상 파일 열기 / 카메라 선택 UI

> 참고: `2601340028/CapturePlate - YOLO/lpr_system/app_pyqt6.py`에 같은 패턴이
> 있다 (VideoWorker + camera_signals + show_alert_popup). 구조를 그대로
> 가져다 쓸 수 있다. 기존 `app/app_pyqt6.py`도 이 프로젝트를 참고해 만들었다.

### 7-4. 데모 영상 확보

학습·검증과 별개로, 발표 시연용 실제 벨트 영상이 필요하다.
- 유튜브: [재활용컨베이어벨트 시운전](https://www.youtube.com/watch?v=RlQEv8MvjTw),
  [VR 플라스틱-컨베이어벨트](https://www.youtube.com/watch?v=kGEbHqxTdt8)
  → 팀 내부 검증용으로만. 앱에 넣어 배포하면 저작권 문제
- 또는 지자체 선별장에 촬영 협조 요청

---

## 7-5. 개선할 문제점 목록

### 기존 앱 (개인용 배출 안내)

| # | 문제 | 방향 | 우선순위 |
|---|---|---|---|
| A1 | **실사용 정확도가 검증되지 않음.** 92.2%는 AI-Hub 자체 test set 기준이고, 실사진 표본은 7장뿐이라 통계적 근거가 부족 | 실사진(직접 촬영 + 인터넷 이미지)을 클래스당 5~10장 모아 정식 평가. 인터넷 이미지도 실사용 시나리오이므로 정당한 테스트 데이터 | **높음** |
| A2 | **정답인데 "재촬영해주세요"가 뜬다.** 실사진 7장 중 2건이 confidence 0.55 미만 (McCol 캔 0.547, 우유팩 0.548) | `CONF_MID`(0.55)/`CONF_HIGH`(0.80) 임계값을 실사진 기준으로 재조정하거나, temperature scaling 등 캘리브레이션 적용 | **높음** |
| A3 | **여러 개를 같이 찍으면 오인식.** 캔 여러 개 → 뭉쳐서 각진 모양이 되어 `beverage_carton`으로 오답 | YOLO로 "몇 개 있는지"만 세서 2개 이상이면 "하나만 찍어주세요" 경고. 라벨이 틀려도 개수만 세면 되므로 마스킹 실험처럼 실패할 위험이 적다 | 중간 |
| A4 | `sealed_container` ↔ `pet_bottle` 혼동이 가장 많음 (confusion matrix 기준 108건 + 49건) | 둘 다 투명 플라스틱이라 시각적으로 실제 유사. 데이터 보강 외에 뾰족한 수가 없음 | 중간 |
| A5 | `electric_fry_pan` recall 0.20 — 원본이 41건뿐 | 현재 `SEARCH_RECOMMENDED_CLASSES`로 목록 유도해 우회 중. 데이터 확보 전까지는 이게 최선 | 낮음 (우회 완료) |
| A6 | **배출 규칙·배출요일이 `verified: false` 초안** | 환경부·지자체 공식 기준으로 검증하고 `verified`를 `true`로 변경. 서비스 배포 전 필수 | 중간 |
| A7 | 배출요일이 서울 4개 구(강남·관악·노원·종로)만 있음 | 필요한 지역이 생길 때마다 구청 공식 페이지 확인해 `rules/collection_days.json`에 추가. 종로구처럼 동별로 다른 곳은 `dong_specific: true` + 챗봇 링크로 처리 | 낮음 |
| A8 | PyQt는 사진 넣자마자 자동 인식 (Streamlit은 "결과 보기" 버튼으로 명시적 실행) | 데스크톱은 지연이 짧아 우선순위 낮음. 통일할지 결정 필요 | 낮음 |

### 신규 벨트 QA (`belt_qa/`)

| # | 문제 | 방향 | 우선순위 |
|---|---|---|---|
| B1 | **학습된 탐지 모델이 없음.** 지금은 COCO 사전학습 YOLOv8n을 임시로 얹은 상태 | AI-Hub 데이터 승인 → 플라스틱/캔 2종으로 정리 → `yolo detect train`. §7-1, §7-2 참고 | **최우선 (병목)** |
| B2 | **추적 ID churn** — 같은 물체가 ID를 잃고 새 ID로 재등록되어 경고가 반복 발생 | ① 실제 영상에서도 재현되는지 먼저 확인(합성 영상 artifact일 수 있음) ② `MISSING_GRACE_FRAMES` 상향 ③ `conf` 임계값 조정 ④ 필요하면 BoT-SORT로 트래커 교체(ReID 있어 더 안정적) | **높음** |
| B3 | **UI가 없음** (`belt_qa/app.py` 미작성) | 영상 패널 + 경고 배너 + 알림음. `lpr_system/app_pyqt6.py`의 VideoWorker/알림 패턴 재사용 | **높음** |
| B4 | **경고음 미구현** | 경고가 하나라도 살아있는 동안 반복 재생, 전부 해제되면 정지. Windows는 `winsound.Beep()` 또는 `QSoundEffect` | 중간 |
| B5 | **실제 벨트 영상 없음** — 지금은 합성 영상으로만 검증 | 유튜브 클립(내부 검증용) 또는 지자체 선별장 촬영 협조. §7-4 참고 | 중간 |
| B6 | 카메라가 "중간과 끝" 두 지점인데 지금은 단일 소스만 처리 | 다중 카메라 구조 설계 필요. `BeltWorker`를 카메라 수만큼 띄우고 경고를 한 화면에 모으는 방식이 단순 | 중간 |
| B7 | 경고 해제 조건이 "화면에서 사라짐"뿐 | 벨트 끝까지 안 치워지고 그냥 지나간 경우와, 작업자가 집어서 치운 경우를 구분하지 못함. 화면 하단(벨트 끝) 통과 여부로 나누면 "놓친 것"을 별도 집계 가능 | 낮음 |

### 공통

| # | 문제 | 방향 |
|---|---|---|
| C1 | `data/`, `results/checkpoints/`가 git에 없어 새 환경에서 재생성 필요 | 용량 때문에 의도한 것. §4 절차대로 재생성하거나 외장 매체로 복사 |
| C2 | Python 3.10/3.11 혼재로 `ModuleNotFoundError` 발생 | §0 참고. 가상환경(venv)으로 고정하는 것이 근본 해결 |

---

## 8. 발표 관련 메모

지수가 정리한 방향: **어떤 모델을 쓰느냐보다, 그 모델의 구조를 이해하고
직접 설명할 수 있는 것이 핵심.**

- 메인 모델은 ResNet18(`improved.pt`)로 유지. DINOv2는 앱에 넣지 않고
  "한계 원인 규명에 쓴 실험"으로만 소개
- 라이브 데모는 위험 — 실사진 인식률이 불안정하므로, AI-Hub 테스트 사진으로
  시연하고 실패 사례(McCol 캔 등)는 **발견한 한계로 정직하게 제시**
- "목록 직접 입력" 기능을 보여주면 "인식 실패해도 배출 안내는 확실히 된다"는
  안전장치 서사가 된다

설명해야 할 ResNet18 구조 포인트:
- ImageNet 사전학습에서 시작 (완전 처음부터가 아님)
- `layer4` + `fc`만 재학습 (`src/model.py`)
- 그래서 6만 장으로도 92%가 나왔지만, 그만큼 AI-Hub 스타일에 과하게 맞춰짐
- 이게 실사진에서 흔들리는 이유이고, DINOv2가 상대적으로 안 흔들린 이유

---

## 9. Git

```bash
git checkout jisu          # 작업 브랜치
git log --oneline -10      # 최근 이력
```

`master`는 안정 버전, `jisu`가 작업 브랜치, `ejae`는 협업자 브랜치.
작업은 `jisu`에서 하고 PR로 `master`에 합친다.

원격: https://github.com/seo941024/waste-lens

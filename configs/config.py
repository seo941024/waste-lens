from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

RESULTS_DIR = ROOT / "results"
CKPT_DIR = RESULTS_DIR / "checkpoints"
RULES_PATH = ROOT / "rules" / "disposal_rules.json"
COLLECTION_DAYS_PATH = ROOT / "rules" / "collection_days.json"

IMG_SIZE = 224
# RTX 3080 10GB 실측: batch 64 = VRAM 1.03GB(10%), 2033 img/s.
# 16은 VRAM 4%만 쓰고 속도는 1/3이라 낭비. 공유 메모리로 넘어갈 여지 없음.
BATCH_SIZE = 64  # 실행 시 --batch-size 로 변경 가능
NUM_WORKERS = 6   # JPEG 디코딩이 병목이라 GPU 학습에서는 필수. 문제 생기면 0으로
EPOCHS = 15
LR = 1e-3
FINETUNE_LR = 1e-4
WEIGHT_DECAY = 1e-4
SEED = 42

# ImageNet 정규화 통계
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Confidence 임계값 (계획서 7장 - Day 13에 실측으로 확정)
CONF_HIGH = 0.80
CONF_MID = 0.55

# AI-Hub 원본 자체가 적어 학습이 불충분한 클래스 (evaluate 결과 기준).
# electric_fry_pan: test 10장 중 2장만 정답(원본 41건뿐).
# electric_iron: test 55장 중 29장 정답, vacuum_cleaner와 혼동 다수(원본 182건).
# 이 클래스로 예측되면 confidence가 높아도 신뢰하지 말고 재질 확인을 안내한다.
LOW_DATA_CLASSES = {"electric_fry_pan", "electric_iron"}

# LOW_DATA_CLASSES 중에서도 recall이 너무 낮아(< 0.3) 사진 인식 자체가
# 사실상 의미 없는 클래스. YOLO(COCO) 크롭 실험(마스킹)도 해봤지만 COCO
# 80종에 이런 생활폐기물이 없어 엉뚱하게 잘려서 오히려 성능이 떨어졌다
# (실측: fry_pan 100%→69.5% mobile_phone으로 바뀌는 등). 데이터를
# 늘리거나 커스텀 탐지기를 새로 학습하기 전까지는, 사진 인식에 기대지
# 말고 목록에서 직접 찾도록 안내하는 게 낫다 (electric_fry_pan recall 0.20).
SEARCH_RECOMMENDED_CLASSES = {"electric_fry_pan"}

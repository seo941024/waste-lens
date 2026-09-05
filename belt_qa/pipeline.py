"""컨베이어벨트 위 미분류 폐기물을 추적하고 경고 상태를 관리한다.

카메라는 분류 작업 라인의 중간~끝 구간에서 벨트를 천장에서 수직으로
내려다보는 위치에 있다고 가정한다 (AI-Hub '재활용품 분류 및 선별 데이터'의
선별영상과 같은 구도).

동작:
    작업자가 놓친 대상 폐기물이 카메라 앞을 지나가면, 추적 ID를 붙여
    화면에 라벨 박스를 계속 띄우고 경고 신호를 올린다. 그 물체가 사라지면
    (집어서 버렸거나 벨트 끝으로 넘어감) 해당 경고만 내린다.

주의:
    아직 실제로 학습된 폐기물 탐지 모델이 없다 (AI-Hub 데이터 승인 대기).
    지금은 COCO 사전학습 YOLOv8n을 임시로 얹어 파이프라인 배선만 검증한다.
    학습이 끝나면 model_path와 target_classes만 바꾸면 그대로 동작한다.

ID churn 대응:
    YOLO.track()의 raw track_id는 가림 등으로 트래커가 같은 물체를 놓쳤다가
    다시 잡으면 새 id를 발급한다. 이걸 그대로 쓰면 같은 물체인데 경고가
    한 번 꺼졌다 켜지는 것처럼 보인다(중복 alert_started + 불필요한
    alert_cleared). TrackedAlert.track_id는 "최초 등록 시 raw id"로 고정된
    안정 id이고, _find_churned_match()가 새 raw id를 라벨+IoU로 최근
    놓친(grace 기간 안) 물체와 대조해 같은 물체면 흡수한다 — 안정 id는
    유지한 채 앞으로의 프레임만 새 raw id로 계속 추적한다.
"""
from dataclasses import dataclass

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO

# 학습 전 임시 설정 — 실제 모델이 나오면 이 두 개를 교체한다.
PLACEHOLDER_MODEL = "yolov8n.pt"
PLACEHOLDER_TARGET_CLASSES = {"bottle", "cup"}  # COCO에서 플라스틱류에 가장 가까운 것

# 몇 프레임 연속으로 안 보여야 "치워졌다"고 볼지. 순간적으로 가려지거나
# 탐지가 한두 프레임 튀는 경우에 경고가 깜빡이는 걸 막는다.
MISSING_GRACE_FRAMES = 10

# 새 raw id가 나타났을 때, 아직 grace 기간 안(missing_streak > 0)인 기존
# 물체와 이 값 이상 겹치면(IoU) 같은 물체로 보고 흡수한다(ID churn 방지).
# 실제 벨트 영상으로 캘리브레이션 전이라 잠정값 — 벨트 속도가 빨라 물체가
# grace 프레임 사이에 많이 이동하면 이 값을 낮춰야 할 수 있다.
REMATCH_IOU_THRESH = 0.3


def _iou(box_a, box_b):
    """두 (x1,y1,x2,y2) 박스의 IoU. 안 겹치면 0."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


@dataclass
class TrackedAlert:
    """아직 안 치워진 것으로 보는 물체 하나.

    track_id는 최초 등록 시 raw id로 고정되는 안정 id다. active_alerts의
    딕셔너리 키(현재 raw id)와는 ID churn 흡수 이후 달라질 수 있다.
    """
    track_id: int
    label: str
    box: tuple  # (x1, y1, x2, y2)
    first_seen_frame: int
    last_seen_frame: int
    missing_streak: int = 0


class BeltWorker(QThread):
    """영상/카메라를 읽어 탐지·추적하고 프레임과 경고를 신호로 내보낸다."""

    frame_ready = pyqtSignal(object)      # 박스가 그려진 BGR ndarray
    alert_started = pyqtSignal(int, str)  # (track_id, label) 새로 발견
    alert_cleared = pyqtSignal(int)       # (track_id) 치워짐
    finished_source = pyqtSignal()        # 영상 파일이 끝남(정상/비정상 공통)
    error = pyqtSignal(str)               # 모델 로드 실패, 소스 열기 실패 등

    def __init__(self, source, model_path=PLACEHOLDER_MODEL,
                 target_classes=None, conf=0.4, parent=None):
        super().__init__(parent)
        self.source = source
        self.model_path = model_path
        self.target_classes = set(target_classes or PLACEHOLDER_TARGET_CLASSES)
        self.conf = conf
        self.active_alerts: dict[int, TrackedAlert] = {}
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        # 모델 로드(첫 실행 시 PLACEHOLDER_MODEL 다운로드 포함)와 소스 열기가
        # 실패해도 QThread 안에서는 예외가 조용히 사라져 UI가 "실행 중"인
        # 채로 멈춘 것처럼 보인다. error 신호로 원인을 밖으로 알린다.
        try:
            model = YOLO(self.model_path)
        except Exception as exc:
            self.error.emit(f"모델을 불러오지 못했습니다: {exc}")
            self.finished_source.emit()
            return

        frame_idx = 0
        try:
            # persist=True 라야 프레임이 바뀌어도 같은 물체에 같은 ID가 유지된다.
            for result in model.track(source=self.source, stream=True, persist=True,
                                      conf=self.conf, verbose=False):
                if self._stop:
                    break
                frame_idx += 1
                frame = result.orig_img.copy()

                detections = []
                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    for xyxy, tid, cls_idx in zip(boxes.xyxy.cpu().numpy(),
                                                  boxes.id.cpu().numpy().astype(int),
                                                  boxes.cls.cpu().numpy().astype(int)):
                        label = model.names[int(cls_idx)]
                        if label not in self.target_classes:
                            continue
                        x1, y1, x2, y2 = xyxy.astype(int)
                        detections.append((int(tid), label, (int(x1), int(y1), int(x2), int(y2))))

                display_id = self._process_frame(detections, frame_idx)

                for raw_tid, label, box in detections:
                    x1, y1, x2, y2 = box
                    shown_id = display_id[raw_tid]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"UNSORTED #{shown_id} {label}",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

                self.frame_ready.emit(frame)
        except Exception as exc:
            self.error.emit(f"영상 처리 중 오류: {exc}")
        finally:
            self.finished_source.emit()

    def _process_frame(self, detections, frame_idx):
        """이번 프레임 탐지로 active_alerts를 갱신하고 필요한 신호를 낸다.

        cv2/YOLO 의존 없이 순수 자료구조만 다뤄서, 실제 벨트 영상 없이도
        합성 시나리오로 단위 테스트할 수 있다.

        detections: [(raw_tid, label, box), ...]
        반환값: {raw_tid: display_id} — 그리기에 쓸 안정 id. 트래커가 가림
        후 새 raw_tid를 발급해도(ID churn) 위치가 겹치면 원래 안정 id를
        그대로 돌려준다.
        """
        display_id = {}
        touched = set()

        for raw_tid, label, box in detections:
            if raw_tid in self.active_alerts:
                alert = self.active_alerts[raw_tid]
                alert.box = box
                alert.last_seen_frame = frame_idx
                alert.missing_streak = 0
            else:
                match_key = self._find_churned_match(raw_tid, label, box)
                if match_key is not None:
                    alert = self.active_alerts.pop(match_key)
                    alert.box = box
                    alert.last_seen_frame = frame_idx
                    alert.missing_streak = 0
                    self.active_alerts[raw_tid] = alert  # 앞으로는 새 raw id로 추적
                else:
                    alert = TrackedAlert(track_id=raw_tid, label=label, box=box,
                                          first_seen_frame=frame_idx, last_seen_frame=frame_idx)
                    self.active_alerts[raw_tid] = alert
                    self.alert_started.emit(raw_tid, label)

            display_id[raw_tid] = alert.track_id  # 항상 최초 등록 id로 표시
            touched.add(raw_tid)

        # 이번 프레임에 안 보인(=raw id로도, churn 매칭으로도 흡수 안 된) 물체는
        # 치워졌는지 카운트한다.
        for tid in list(self.active_alerts):
            if tid in touched:
                continue
            alert = self.active_alerts[tid]
            alert.missing_streak += 1
            if alert.missing_streak >= MISSING_GRACE_FRAMES:
                del self.active_alerts[tid]
                self.alert_cleared.emit(alert.track_id)  # 딕셔너리 키가 아니라 안정 id로 알림

        return display_id

    def _find_churned_match(self, new_tid, label, box):
        """놓친 지 얼마 안 된(grace 기간 안) 기존 물체 중 라벨이 같고 위치가
        많이 겹치는 게 있으면 그 raw id(active_alerts의 키)를 돌려준다."""
        best_tid, best_iou = None, REMATCH_IOU_THRESH
        for tid, alert in self.active_alerts.items():
            if tid == new_tid or alert.missing_streak == 0 or alert.label != label:
                continue
            iou = _iou(alert.box, box)
            if iou > best_iou:
                best_tid, best_iou = tid, iou
        return best_tid

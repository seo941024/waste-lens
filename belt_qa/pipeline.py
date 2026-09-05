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


@dataclass
class TrackedAlert:
    """아직 안 치워진 것으로 보는 물체 하나."""
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
    finished_source = pyqtSignal()        # 영상 파일이 끝남

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
        model = YOLO(self.model_path)
        frame_idx = 0

        # persist=True 라야 프레임이 바뀌어도 같은 물체에 같은 ID가 유지된다.
        for result in model.track(source=self.source, stream=True, persist=True,
                                  conf=self.conf, verbose=False):
            if self._stop:
                break
            frame_idx += 1
            frame = result.orig_img.copy()
            seen_now = set()

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                for xyxy, tid, cls_idx in zip(boxes.xyxy.cpu().numpy(),
                                              boxes.id.cpu().numpy().astype(int),
                                              boxes.cls.cpu().numpy().astype(int)):
                    label = model.names[int(cls_idx)]
                    if label not in self.target_classes:
                        continue

                    tid = int(tid)
                    seen_now.add(tid)
                    x1, y1, x2, y2 = xyxy.astype(int)

                    if tid in self.active_alerts:
                        alert = self.active_alerts[tid]
                        alert.box = (x1, y1, x2, y2)
                        alert.last_seen_frame = frame_idx
                        alert.missing_streak = 0
                    else:
                        self.active_alerts[tid] = TrackedAlert(
                            track_id=tid, label=label, box=(x1, y1, x2, y2),
                            first_seen_frame=frame_idx, last_seen_frame=frame_idx)
                        self.alert_started.emit(tid, label)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"UNSORTED #{tid} {label}",
                                (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 255), 2)

            # 이번 프레임에 안 보인 물체는 치워졌는지 카운트한다.
            for tid in list(self.active_alerts):
                if tid in seen_now:
                    continue
                alert = self.active_alerts[tid]
                alert.missing_streak += 1
                if alert.missing_streak >= MISSING_GRACE_FRAMES:
                    del self.active_alerts[tid]
                    self.alert_cleared.emit(tid)

            self.frame_ready.emit(frame)

        self.finished_source.emit()

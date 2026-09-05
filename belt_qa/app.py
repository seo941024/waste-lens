"""벨트 QA 데스크톱 UI.

작업자가 놓친 미분류 폐기물이 벨트를 지나가면 화면에 경고 박스를 띄우고
알림음을 울린다. 물체가 치워지면(BeltWorker가 감지) 해당 경고만 내린다.

실행:
    python belt_qa/app.py

주의:
    아직 학습된 폐기물 탐지 모델이 없다 (AI-Hub 데이터 승인 대기). 지금은
    COCO 사전학습 YOLOv8n을 임시로 얹어 파이프라인·UI 배선만 검증한다
    (belt_qa/pipeline.py의 PLACEHOLDER_MODEL 참고).
"""
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
                              QListWidget, QListWidgetItem, QMainWindow,
                              QMessageBox, QPushButton, QSizePolicy,
                              QVBoxLayout, QWidget)

from belt_qa.pipeline import (MISSING_GRACE_FRAMES, PLACEHOLDER_MODEL,
                               PLACEHOLDER_TARGET_CLASSES, BeltWorker)

try:
    import winsound
    HAS_WINSOUND = True
except ImportError:  # Windows 외 환경 (개발 중 확인용)
    HAS_WINSOUND = False

# 기존 app/app_pyqt6.py(친환경 녹색 테마)와 팔레트를 맞춘다.
BG = "#F4F8F5"
PANEL_BG = "#FFFFFF"
BORDER = "#DCE7E0"
TEXT = "#1B2A20"
MUTED = "#5F7267"
PRIMARY = "#167331"
PRIMARY_DARK = "#0D3B1E"
ACCENT = "#2E9E52"
DANGER_BG = "#FEE2E2"
DANGER_BORDER = "#DC2626"
DANGER_TEXT = "#991B1B"
WARN_BG = "#FEF3C7"
WARN_TEXT = "#92400E"

ALARM_INTERVAL_MS = 1500  # 경고가 살아있는 동안 이 주기로 짧게 삑
BEEP_FREQ_HZ = 1500
BEEP_MS = 150


class BeltQAWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("벨트 QA — 미분류 폐기물 감지 (검증용)")
        self.resize(1180, 720)
        self.setStyleSheet(f"QMainWindow {{ background: {BG}; }}")

        self.worker: BeltWorker | None = None
        self.source = None
        self.active_count = 0

        self._build_ui()

        self.alarm_timer = QTimer(self)
        self.alarm_timer.setInterval(ALARM_INTERVAL_MS)
        self.alarm_timer.timeout.connect(self._tick_alarm)
        self.alarm_timer.start()

    # ------------------------------------------------------------------ UI

    def _build_ui(self):
        central = QWidget()
        central.setStyleSheet(f"color: {TEXT}; font-size: 13px;")
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(12)

        root.addWidget(self._build_warning_banner())
        root.addLayout(self._build_toolbar())

        content = QHBoxLayout()
        content.setSpacing(12)
        content.addWidget(self._build_video_panel(), stretch=3)
        content.addWidget(self._build_alert_panel(), stretch=1)
        root.addLayout(content, stretch=1)

        self.status_label = QLabel("소스를 선택하고 시작을 누르세요.")
        self.status_label.setStyleSheet(f"color: {MUTED};")
        root.addWidget(self.status_label)

        self.setCentralWidget(central)

    def _build_warning_banner(self):
        classes = ", ".join(sorted(PLACEHOLDER_TARGET_CLASSES))
        banner = QLabel(
            f"⚠ 아직 학습된 폐기물 탐지 모델이 아닙니다 — COCO 사전학습 "
            f"{PLACEHOLDER_MODEL} 임시 사용 (대상 클래스: {classes}). "
            f"실제 배포 전 전용 데이터로 재학습이 필요합니다."
        )
        banner.setWordWrap(True)
        banner.setStyleSheet(
            f"background: {WARN_BG}; color: {WARN_TEXT}; padding: 8px 12px; "
            f"border-radius: 6px; font-weight: 600;"
        )
        return banner

    def _build_toolbar(self):
        row = QHBoxLayout()
        row.setSpacing(8)

        self.open_file_btn = QPushButton("📁 영상 파일 열기")
        self.open_camera_btn = QPushButton("🎥 카메라 열기")
        self.start_btn = QPushButton("▶ 시작")
        self.stop_btn = QPushButton("■ 정지")
        self.stop_btn.setEnabled(False)

        for btn in (self.open_file_btn, self.open_camera_btn):
            btn.setStyleSheet(self._button_style(PANEL_BG, TEXT, BORDER))
        self.start_btn.setStyleSheet(self._button_style(PRIMARY, "#FFFFFF", PRIMARY))
        self.stop_btn.setStyleSheet(self._button_style(DANGER_BG, DANGER_TEXT, DANGER_BORDER))

        self.open_file_btn.clicked.connect(self._choose_file)
        self.open_camera_btn.clicked.connect(self._choose_camera)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)

        self.source_label = QLabel("소스 없음")
        self.source_label.setStyleSheet(f"color: {MUTED};")

        row.addWidget(self.open_file_btn)
        row.addWidget(self.open_camera_btn)
        row.addWidget(self.source_label, stretch=1)
        row.addWidget(self.start_btn)
        row.addWidget(self.stop_btn)
        return row

    def _build_video_panel(self):
        self.video_label = QLabel("소스를 선택하세요")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Expanding)
        self.video_label.setStyleSheet(
            f"background: #10221A; color: #9BB8A5; border-radius: 8px; "
            f"border: 1px solid {BORDER};"
        )
        return self.video_label

    def _build_alert_panel(self):
        panel = QWidget()
        panel.setStyleSheet(
            f"background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 8px;"
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("⚠ 미분류 경고")
        title.setStyleSheet(f"font-weight: 700; font-size: 15px; color: {PRIMARY_DARK};")
        layout.addWidget(title)

        self.summary_label = QLabel("현재 경고 0건")
        self.summary_label.setStyleSheet(self._summary_style(active=False))
        layout.addWidget(self.summary_label)

        self.alert_list = QListWidget()
        self.alert_list.setStyleSheet(
            f"QListWidget {{ border: none; }} "
            f"QListWidget::item {{ background: {DANGER_BG}; color: {DANGER_TEXT}; "
            f"border: 1px solid {DANGER_BORDER}; border-radius: 6px; "
            f"padding: 6px; margin-bottom: 4px; }}"
        )
        layout.addWidget(self.alert_list, stretch=1)

        hint = QLabel(
            f"경고는 같은 물체가 최대 {MISSING_GRACE_FRAMES}프레임 안 보여도 "
            f"유지됩니다(순간적으로 가려짐 대응). 그보다 오래 사라지면 "
            f"'치워짐'으로 보고 해제됩니다."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {MUTED}; font-size: 11px;")
        layout.addWidget(hint)

        return panel

    @staticmethod
    def _button_style(bg, fg, border):
        return (
            f"QPushButton {{ background: {bg}; color: {fg}; border: 1px solid {border}; "
            f"border-radius: 6px; padding: 6px 14px; font-weight: 600; }}"
            f"QPushButton:disabled {{ opacity: 0.5; color: #9BB8A5; }}"
        )

    @staticmethod
    def _summary_style(active):
        if active:
            return f"background: {DANGER_BG}; color: {DANGER_TEXT}; padding: 6px 10px; border-radius: 6px; font-weight: 700;"
        return f"background: {BG}; color: {MUTED}; padding: 6px 10px; border-radius: 6px;"

    # ------------------------------------------------------------- 소스 선택

    def _choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "영상 파일 선택", "",
            "동영상 파일 (*.mp4 *.avi *.mov *.mkv);;모든 파일 (*)")
        if path:
            self.source = path
            self.source_label.setText(f"파일: {Path(path).name}")

    def _choose_camera(self):
        # 카메라 인덱스 선택 UI는 최소화하고 기본 카메라(0)만 지원한다.
        # 여러 대를 붙여 써야 하면(§B6) 인덱스 입력 다이얼로그를 추가하면 된다.
        self.source = 0
        self.source_label.setText("카메라: 기본 장치 (0)")

    # --------------------------------------------------------------- 실행

    def _start(self):
        if self.source is None:
            QMessageBox.warning(self, "소스 없음", "먼저 영상 파일이나 카메라를 선택하세요.")
            return
        if self.worker is not None:
            return

        self.alert_list.clear()
        self.active_count = 0
        self._update_summary()

        self.worker = BeltWorker(source=self.source)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.alert_started.connect(self._on_alert_started)
        self.worker.alert_cleared.connect(self._on_alert_cleared)
        self.worker.finished_source.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.open_file_btn.setEnabled(False)
        self.open_camera_btn.setEnabled(False)
        self.status_label.setText("감지 중...")

    def _stop(self):
        if self.worker is None:
            return
        self.worker.stop()
        self.worker.wait(3000)
        self._reset_after_stop("정지했습니다.")

    def _reset_after_stop(self, message):
        self.worker = None
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.open_file_btn.setEnabled(True)
        self.open_camera_btn.setEnabled(True)
        self.status_label.setText(message)

    # ------------------------------------------------------------- 신호 처리

    def _on_frame(self, frame):
        h, w, ch = frame.shape
        qimg = QImage(frame.data, w, h, ch * w, QImage.Format.Format_BGR888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _on_alert_started(self, track_id, label):
        self.active_count += 1
        item = QListWidgetItem(f"#{track_id}  UNSORTED  {label}")
        item.setData(Qt.ItemDataRole.UserRole, track_id)
        self.alert_list.addItem(item)
        self._update_summary()

    def _on_alert_cleared(self, track_id):
        self.active_count = max(0, self.active_count - 1)
        for i in range(self.alert_list.count()):
            item = self.alert_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == track_id:
                self.alert_list.takeItem(i)
                break
        self._update_summary()

    def _on_finished(self):
        # 영상 파일이 끝났거나(카메라라면 거의 발생 안 함) 에러로 중단된 경우.
        # 남은 경고는 마지막 확인된 상태로 그대로 두고, 조작 버튼만 되돌린다.
        if self.worker is not None:
            self._reset_after_stop("소스가 끝났습니다.")

    def _on_error(self, message):
        QMessageBox.critical(self, "오류", message)
        self._reset_after_stop(f"오류로 중단됨: {message}")

    def _update_summary(self):
        self.summary_label.setText(f"현재 경고 {self.active_count}건")
        self.summary_label.setStyleSheet(self._summary_style(active=self.active_count > 0))

    # --------------------------------------------------------------- 알림음

    def _tick_alarm(self):
        # 실제 삑 소리는 GUI 스레드를 막지 않도록 짧은 별도 스레드에서 낸다.
        # 경고가 하나라도 남아있는 동안 계속, 전부 해제되면 자동으로 멈춘다.
        if self.active_count > 0 and HAS_WINSOUND:
            threading.Thread(target=winsound.Beep, args=(BEEP_FREQ_HZ, BEEP_MS),
                              daemon=True).start()

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = BeltQAWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""app_pyqt6.py - PyQt6 기반 폐기물 배출 안내 UI

사진을 넣으면 물품을 인식하고, 신뢰도를 판단해 배출 방법을 안내한다.

실행:
    python app/app_pyqt6.py
"""
import json
import os
import sys
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (QApplication, QComboBox, QFileDialog, QFrame,
                             QGraphicsDropShadowEffect, QHBoxLayout,
                             QHeaderView, QLabel, QMainWindow, QProgressBar,
                             QPushButton, QScrollArea, QSizePolicy,
                             QStackedWidget, QTableWidget, QTableWidgetItem,
                             QToolButton, QVBoxLayout, QWidget)

import qtawesome as qta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.classes import CLASSES, CLASS_KOR_NAME
from configs.config import (CKPT_DIR, CONF_HIGH, CONF_MID, LOW_DATA_CLASSES,
                            RESULTS_DIR, RULES_PATH)
from src.inference import WastePredictor

# ── 색상 팔레트 (친환경 녹색 컨셉, 기준색 #167331) ──────────────────────────────────
C_ACCENT      = "#167331"   # 브랜드 기준색 — 주요 버튼·태그
C_ACCENT_DARK = "#115C27"   # hover
C_ACCENT_TINT = "#E8F3EC"   # 옅은 배경 강조
C_GREEN       = "#2E9E52"   # 성공·신뢰도 높음 (기준색보다 밝게 두어 위계 구분)
C_SIDEBAR      = "#0D3B1E"  # 사이드바 — 기준색을 어둡게
C_SIDEBAR_TEXT = "#9BB8A5"  # 사이드바 위 비활성 글자·아이콘
C_SIDEBAR_HOVER = "#CFE3D6" # 사이드바 hover
C_BG          = "#F4F8F5"   # 페이지 배경 — 녹색기 도는 밝은 회색
C_SURFACE     = "#FFFFFF"
C_RED         = "#DC2626"
C_YELLOW      = "#D97706"
C_TEXT        = "#1B2A20"
C_MUTED       = "#5F7267"
C_BORDER      = "#DCE7E0"

IMG_EXTS = "이미지 (*.jpg *.jpeg *.png *.bmp *.webp);;모든 파일 (*)"

STYLE_SHEET = f"""
QMainWindow, QWidget#MainBG {{ background-color: {C_BG}; }}
QWidget {{ background-color: transparent; }}
QFrame#Card {{
    background-color: {C_SURFACE};
    border: none;
    border-radius: 10px;
}}
QFrame#DropZone {{
    background-color: {C_SURFACE};
    border: 2px dashed {C_BORDER};
    border-radius: 10px;
}}
QFrame#DropZoneActive {{
    background-color: {C_ACCENT_TINT};
    border: 2px dashed {C_ACCENT};
    border-radius: 10px;
}}
QLabel {{ color: {C_TEXT}; font-size: 15px; }}
QLabel#HeaderTitle {{ font-size: 22px; font-weight: bold; color: {C_TEXT}; }}
QLabel#SectionTitle {{ font-size: 24px; font-weight: bold; color: {C_TEXT}; }}
QLabel#Muted {{ color: {C_MUTED}; font-size: 14px; }}
QLabel#StatValue {{ font-size: 34px; font-weight: bold; color: {C_TEXT}; }}
QLabel#ResultName {{ font-size: 32px; font-weight: bold; color: {C_TEXT}; }}
QLabel#RuleBody {{ font-size: 15px; color: {C_TEXT}; }}
QPushButton {{
    border-radius: 6px;
    padding: 9px 18px;
    font-weight: bold;
    font-size: 15px;
    border: none;
}}
QPushButton#Primary {{ background-color: {C_ACCENT}; color: white; }}
QPushButton#Primary:hover {{ background-color: {C_ACCENT_DARK}; }}
QPushButton#Primary:disabled {{ background-color: {C_MUTED}; }}
QPushButton#Outline {{ background-color: {C_ACCENT_DARK}; color: white; }}
QPushButton#Outline:hover {{ background-color: {C_SIDEBAR}; }}
QPushButton#Muted {{ background-color: {C_MUTED}; color: white; }}
QPushButton#SectionTab {{
    background-color: {C_BG}; color: {C_MUTED};
    padding: 7px 14px; font-size: 13px; border-radius: 8px;
}}
QPushButton#SectionTab:checked {{ background-color: {C_ACCENT}; color: white; }}
QComboBox {{
    padding: 8px 12px;
    border: none;
    border-radius: 6px;
    background-color: {C_BG};
    color: {C_TEXT};
    font-size: 15px;
}}
QComboBox::drop-down {{ border: none; padding-right: 8px; }}
QScrollArea {{ border: none; background-color: transparent; }}
QScrollBar:vertical {{
    background: {C_BG}; width: 8px; border-radius: 4px; margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {C_MUTED}55; border-radius: 4px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {C_MUTED}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QTableWidget {{
    border: none;
    border-radius: 8px;
    background-color: {C_SURFACE};
    gridline-color: {C_BG};
    font-size: 14px;
}}
QHeaderView::section {{
    background-color: {C_SIDEBAR};
    color: {C_SIDEBAR_HOVER};
    border: none;
    font-weight: bold;
    font-size: 14px;
    padding: 9px 12px;
}}
QTableWidget::item {{ padding: 8px 12px; color: {C_TEXT}; }}
QProgressBar {{
    border: none;
    border-radius: 4px;
    background-color: {C_BG};
    height: 8px;
    text-align: center;
}}
QProgressBar::chunk {{ background-color: {C_ACCENT}; border-radius: 4px; }}
QStatusBar {{ background-color: {C_SURFACE}; border: none; font-size: 14px; }}
QFrame#Sidebar {{ background-color: {C_SIDEBAR}; border: none; }}
QToolButton {{
    background-color: transparent;
    color: {C_SIDEBAR_TEXT};
    border: none;
    border-radius: 8px;
    font-size: 12px;
    padding: 6px 0px;
}}
QToolButton:hover {{ background-color: rgba(255,255,255,0.10); color: {C_SIDEBAR_HOVER}; }}
"""


# ── 공통 헬퍼 ──────────────────────────────────────────────────────────────────────
def create_shadow(blur=8, opacity=18, dy=2):
    s = QGraphicsDropShadowEffect()
    s.setBlurRadius(blur)
    s.setColor(QColor(0, 0, 0, opacity))
    s.setOffset(0, dy)
    return s


def level_color(level):
    return {"high": C_GREEN, "mid": C_YELLOW, "low": C_RED}.get(level, C_MUTED)


def level_text(level):
    return {"high": "신뢰도 높음", "mid": "확인 필요", "low": "인식 실패"}.get(level, "-")


def make_tag(text, color):
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background-color: {color}; color: white; border-radius: 4px; "
        f"padding: 4px 12px; font-weight: bold; font-size: 12px;"
    )
    lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lbl.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
    return lbl


def clear_layout(layout):
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().setParent(None)
        elif item.layout():
            clear_layout(item.layout())


def available_models():
    """results/checkpoints 안의 .pt 파일 이름 목록. improved를 앞에 둔다."""
    if not CKPT_DIR.is_dir():
        return []
    names = sorted(p.stem for p in CKPT_DIR.glob("*.pt"))
    return sorted(names, key=lambda n: (n != "improved", n))


# ── 백그라운드 작업 ─────────────────────────────────────────────────────────────────
class ModelLoader(QThread):
    """체크포인트 로딩은 수 초 걸리므로 UI를 막지 않도록 별도 스레드에서 한다."""
    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, name):
        super().__init__()
        self.name = name

    def run(self):
        try:
            self.ready.emit(WastePredictor(CKPT_DIR / f"{self.name}.pt"))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class PredictWorker(QThread):
    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, predictor, image_path):
        super().__init__()
        self.predictor = predictor
        self.image_path = image_path

    def run(self):
        try:
            self.done.emit(self.predictor.predict(self.image_path))
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


# ── 드래그&드롭 이미지 영역 ─────────────────────────────────────────────────────────
class DropZone(QFrame):
    dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setMinimumSize(420, 340)

        lo = QVBoxLayout(self)
        lo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lo.setSpacing(10)

        self.icon = QLabel()
        self.icon.setPixmap(qta.icon("fa5s.image", color=C_MUTED).pixmap(46, 46))
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lo.addWidget(self.icon)

        self.hint = QLabel("이미지를 여기로 끌어다 놓거나\n아래 버튼으로 선택하세요")
        self.hint.setObjectName("Muted")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lo.addWidget(self.hint)

        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setVisible(False)
        lo.addWidget(self.preview)

    def show_image(self, path):
        pm = QPixmap(path)
        if pm.isNull():
            return False
        self.icon.setVisible(False)
        self.hint.setVisible(False)
        self.preview.setVisible(True)
        self.preview.setPixmap(pm.scaled(
            400, 300,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))
        return True

    def _is_image(self, url):
        return url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls()
        if urls and self._is_image(urls[0]):
            self.setObjectName("DropZoneActive")
            self.setStyleSheet(STYLE_SHEET)
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.setObjectName("DropZone")
        self.setStyleSheet(STYLE_SHEET)

    def dropEvent(self, event):
        self.setObjectName("DropZone")
        self.setStyleSheet(STYLE_SHEET)
        urls = event.mimeData().urls()
        if urls and self._is_image(urls[0]):
            self.dropped.emit(urls[0].toLocalFile())


# ── 메인 윈도우 ────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("어디버려? — 폐기물 배출 안내")
        self.setMinimumSize(1180, 780)
        self.setStyleSheet(STYLE_SHEET)

        self.predictor = None
        self.loader = None
        self.worker = None
        self.current_image = None
        self.model_name = (available_models() or ["improved"])[0]
        self.rules = self._load_rules()

        main_widget = QWidget()
        main_widget.setObjectName("MainBG")
        self.setCentralWidget(main_widget)
        root = QHBoxLayout(main_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(96)
        self.sidebar_lo = QVBoxLayout(self.sidebar)
        self.sidebar_lo.setContentsMargins(6, 12, 6, 12)
        self.sidebar_lo.setSpacing(4)
        self.sidebar_lo.setAlignment(Qt.AlignmentFlag.AlignTop)
        root.addWidget(self.sidebar)

        right = QWidget()
        right_lo = QVBoxLayout(right)
        right_lo.setContentsMargins(0, 0, 0, 0)
        right_lo.setSpacing(0)

        header = QFrame()
        header.setStyleSheet(
            f"background-color: {C_SURFACE}; border-bottom: 1px solid {C_BORDER};")
        header.setFixedHeight(52)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.setSpacing(10)
        icon = QLabel()
        icon.setPixmap(qta.icon("fa5s.recycle", color=C_GREEN).pixmap(24, 24))
        hl.addWidget(icon)
        title = QLabel("어디버려?")
        title.setObjectName("HeaderTitle")
        hl.addWidget(title)
        sub = QLabel("생활용품 이미지 인식 기반 배출 안내")
        sub.setObjectName("Muted")
        hl.addWidget(sub)
        hl.addStretch()
        self.header_model = QLabel()
        self.header_model.setObjectName("Muted")
        hl.addWidget(self.header_model)
        right_lo.addWidget(header)

        self.stacked = QStackedWidget()
        right_lo.addWidget(self.stacked, 1)
        root.addWidget(right, 1)

        self.init_pages()
        self.init_sidebar()
        self._update_header_model()

        QTimer.singleShot(100, self.load_model)

    # ── 공통 ───────────────────────────────────────────────────────────────────────
    def _load_rules(self):
        try:
            return json.loads(RULES_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def show_toast(self, msg, is_error=False):
        bar = self.statusBar()
        bar.showMessage(msg, 4000)
        bar.setStyleSheet(
            f"color: {C_RED if is_error else C_GREEN}; font-weight: bold; padding: 4px 12px;")

    def _section_header(self, title, right_widget=None):
        row = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setObjectName("SectionTitle")
        row.addWidget(lbl)
        row.addStretch()
        if right_widget:
            row.addWidget(right_widget)
        return row

    def _scroll_area(self):
        sa = QScrollArea(widgetResizable=True)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        w = QWidget()
        w.setStyleSheet("background-color: transparent;")
        vb = QVBoxLayout(w)
        vb.setAlignment(Qt.AlignmentFlag.AlignTop)
        vb.setSpacing(8)
        sa.setWidget(w)
        return sa, vb

    def _update_header_model(self):
        state = "로딩 중..." if self.predictor is None else "준비됨"
        self.header_model.setText(f"모델: {self.model_name}  ·  {state}")

    # ── 사이드바 ───────────────────────────────────────────────────────────────────
    def init_sidebar(self):
        menus = [
            ("배출 안내", "fa5s.camera"),
            ("배출 규칙", "fa5s.book"),
            ("모델 성능", "fa5s.chart-bar"),
            ("설정", "fa5s.sliders-h"),
        ]
        self.nav_btns = []
        for i, (text, icon_name) in enumerate(menus):
            btn = QToolButton()
            btn.setText(text)
            btn.setIcon(qta.icon(icon_name, color=C_SIDEBAR_TEXT))
            btn.setIconSize(QSize(22, 22))
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setFixedSize(84, 68)
            btn.setProperty("icon_name", icon_name)
            btn.clicked.connect(lambda _, idx=i: self.switch_page(idx))
            self.sidebar_lo.addWidget(btn)
            self.nav_btns.append(btn)
        self.switch_page(0)

    def switch_page(self, idx):
        self.stacked.setCurrentIndex(idx)
        for i, btn in enumerate(self.nav_btns):
            name = btn.property("icon_name")
            if i == idx:
                btn.setStyleSheet(
                    f"QToolButton {{ background-color: {C_ACCENT}; color: white; "
                    "border-radius: 8px; font-weight: bold; font-size: 12px; padding: 6px 0px; }")
                btn.setIcon(qta.icon(name, color="white"))
            else:
                btn.setStyleSheet(
                    f"QToolButton {{ background-color: transparent; color: {C_SIDEBAR_TEXT}; "
                    "border: none; font-size: 12px; padding: 6px 0px; }")
                btn.setIcon(qta.icon(name, color=C_SIDEBAR_TEXT))

        if idx == 1:
            self.rules_refresh()
        elif idx == 2:
            self.stats_refresh()

    def init_pages(self):
        for attr, builder in [
            ("page_predict", self.build_predict),
            ("page_rules", self.build_rules),
            ("page_stats", self.build_stats),
            ("page_settings", self.build_settings),
        ]:
            page = QWidget()
            setattr(self, attr, page)
            self.stacked.addWidget(page)
            builder(page)

    # ════════════════════════════════════════════════════════════
    # 1. 배출 안내
    # ════════════════════════════════════════════════════════════
    def build_predict(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16)
        lo.setSpacing(12)
        lo.addLayout(self._section_header("배출 안내"))

        hint = QLabel("버리려는 물건 하나가 잘 보이도록 촬영한 사진을 넣어 주세요.")
        hint.setObjectName("Muted")
        lo.addWidget(hint)

        split = QHBoxLayout()
        split.setSpacing(14)

        # 왼쪽: 이미지 입력
        left = QVBoxLayout()
        left.setSpacing(10)
        self.drop = DropZone()
        self.drop.dropped.connect(self.run_predict)
        left.addWidget(self.drop, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_pick = QPushButton("이미지 선택")
        self.btn_pick.setObjectName("Primary")
        self.btn_pick.setIcon(qta.icon("fa5s.folder-open", color="white"))
        self.btn_pick.clicked.connect(self.pick_image)
        self.btn_again = QPushButton("다시 인식")
        self.btn_again.setObjectName("Muted")
        self.btn_again.setEnabled(False)
        self.btn_again.clicked.connect(
            lambda: self.current_image and self.run_predict(self.current_image))
        btn_row.addWidget(self.btn_pick, 1)
        btn_row.addWidget(self.btn_again)
        left.addLayout(btn_row)
        left.addWidget(self._manual_card())
        split.addLayout(left, 1)

        # 오른쪽: 결과 — 섹션 탭 + 스택. 한 번에 한 섹션만 보여줘야
        # 배출 안내 본문이 길어도 화면 아래로 밀려서 잘리지 않는다.
        right = QVBoxLayout()
        right.setSpacing(8)
        self.section_row = QHBoxLayout()
        self.section_row.setSpacing(6)
        right.addLayout(self.section_row)
        self.section_stack = QStackedWidget()
        right.addWidget(self.section_stack, 1)
        split.addLayout(right, 1)
        self._section_btns = []

        lo.addLayout(split, 1)
        self._show_placeholder()

    def _set_sections(self, sections, default_index=0):
        """sections: [(탭 이름, 내용 QWidget)]. 하나만 보이고 탭으로 전환한다."""
        while self.section_row.count():
            item = self.section_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        while self.section_stack.count():
            w = self.section_stack.widget(0)
            self.section_stack.removeWidget(w)
            w.deleteLater()

        self._section_btns = []
        show_tabs = len(sections) > 1
        for i, (label, widget) in enumerate(sections):
            if show_tabs:
                btn = QPushButton(label)
                btn.setObjectName("SectionTab")
                btn.setCheckable(True)
                btn.clicked.connect(lambda _, idx=i: self._switch_section(idx))
                self.section_row.addWidget(btn)
                self._section_btns.append(btn)

            page_scroll, page_box = self._scroll_area()
            page_box.addWidget(widget)
            self.section_stack.addWidget(page_scroll)

        if show_tabs:
            self.section_row.addStretch()
        self._switch_section(default_index)

    def _switch_section(self, idx):
        self.section_stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._section_btns):
            btn.setChecked(i == idx)

    def _manual_card(self):
        """사진 없이 목록에서 직접 고르기. 오인식했거나 그냥 찾아보고 싶을 때 쓴다."""
        card = QFrame()
        card.setObjectName("Card")
        card.setGraphicsEffect(create_shadow())
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 12, 16, 12)
        cl.setSpacing(8)

        head = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(qta.icon("fa5s.list-ul", color=C_ACCENT).pixmap(15, 15))
        head.addWidget(ic)
        t = QLabel("목록에서 직접 찾기")
        t.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {C_TEXT};")
        head.addWidget(t)
        head.addStretch()
        cl.addLayout(head)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.manual_cat = QComboBox()
        cats = sorted({(self.rules.get(s, {}).get("disposal_category") or "기타")
                       for s in CLASSES})
        self.manual_cat.addItems(["전체"] + cats)
        self.manual_cat.currentTextChanged.connect(self._manual_fill_items)
        row.addWidget(self.manual_cat, 1)

        self.manual_item = QComboBox()
        row.addWidget(self.manual_item, 2)
        cl.addLayout(row)

        btn = QPushButton("배출 방법 보기")
        btn.setObjectName("Outline")
        btn.clicked.connect(self._show_manual_rule)
        cl.addWidget(btn)

        self._manual_fill_items("전체")
        return card

    def _manual_fill_items(self, category):
        """분류를 고르면 그 분류의 품목만 남긴다."""
        self.manual_item.clear()
        for slug in CLASSES:
            cat = self.rules.get(slug, {}).get("disposal_category") or "기타"
            if category in ("전체", cat):
                self.manual_item.addItem(CLASS_KOR_NAME.get(slug, slug), slug)

    def _show_manual_rule(self):
        slug = self.manual_item.currentData()
        if not slug:
            return
        rule = self.rules.get(slug)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)

        head = QFrame()
        head.setStyleSheet(
            f"background-color: {C_ACCENT_TINT}; border: none; border-radius: 10px;")
        hl = QVBoxLayout(head)
        hl.setContentsMargins(18, 14, 18, 14)
        hl.setSpacing(4)
        row = QHBoxLayout()
        row.addWidget(make_tag("직접 선택", C_ACCENT))
        row.addStretch()
        hl.addLayout(row)
        name = QLabel(CLASS_KOR_NAME.get(slug, slug))
        name.setObjectName("ResultName")
        hl.addWidget(name)
        note = QLabel("목록에서 고른 품목입니다. 사진 인식 결과가 아닙니다.")
        note.setObjectName("Muted")
        hl.addWidget(note)
        bl.addWidget(head)

        if rule:
            bl.addWidget(self._rule_card(rule))
        if slug in LOW_DATA_CLASSES:
            bl.addWidget(self._warn_card(
                "학습 데이터가 적은 품목입니다",
                "사진으로 인식할 때 오인식 가능성이 높은 품목입니다. "
                "제품에 표시된 재질을 함께 확인하세요."))
        bl.addStretch()

        self._set_sections([("배출 안내", body)])

    def _show_placeholder(self):
        card = QFrame()
        card.setObjectName("Card")
        card.setGraphicsEffect(create_shadow())
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 40, 20, 40)
        cl.setSpacing(10)
        cl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ic = QLabel()
        ic.setPixmap(qta.icon("fa5s.search", color=C_BORDER).pixmap(40, 40))
        ic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(ic)
        msg = QLabel("이미지를 넣으면 결과가 여기에 표시됩니다")
        msg.setObjectName("Muted")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(msg)
        self._set_sections([("안내", card)])

    def pick_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "이미지 선택", "", IMG_EXTS)
        if path:
            self.run_predict(path)

    def run_predict(self, path):
        if self.predictor is None:
            self.show_toast("모델을 아직 불러오는 중입니다. 잠시 후 다시 시도하세요.", True)
            return
        if not self.drop.show_image(path):
            self.show_toast("이미지를 열 수 없습니다.", True)
            return

        self.current_image = path
        self.btn_pick.setEnabled(False)
        self.btn_again.setEnabled(False)

        busy = QLabel("인식 중...")
        busy.setObjectName("Muted")
        busy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._set_sections([("안내", busy)])

        self.worker = PredictWorker(self.predictor, path)
        self.worker.done.connect(self.show_result)
        self.worker.failed.connect(self._predict_failed)
        self.worker.start()

    def _predict_failed(self, msg):
        self.btn_pick.setEnabled(True)
        self.btn_again.setEnabled(True)
        self._show_placeholder()
        self.show_toast(f"인식 실패: {msg}", True)

    def show_result(self, result):
        self.btn_pick.setEnabled(True)
        self.btn_again.setEnabled(True)

        top = result["top1"]
        level = result["confidence_level"]
        color = level_color(level)

        # 결과 카드
        card = QFrame()
        card.setObjectName("Card")
        card.setGraphicsEffect(create_shadow())
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 18, 20, 18)
        cl.setSpacing(10)

        tag_row = QHBoxLayout()
        tag_row.addWidget(make_tag(level_text(level), color))
        tag_row.addStretch()
        conf = QLabel(f"신뢰도 {top['confidence']:.1%}")
        conf.setObjectName("Muted")
        tag_row.addWidget(conf)
        cl.addLayout(tag_row)

        if level == "low":
            name = QLabel("인식하지 못했습니다")
            name.setStyleSheet(f"font-size: 26px; font-weight: bold; color: {C_RED};")
        else:
            name = QLabel(top["name_kor"])
            name.setObjectName("ResultName")
        cl.addWidget(name)

        msg = QLabel(result["message"])
        msg.setWordWrap(True)
        msg.setObjectName("Muted")
        cl.addWidget(msg)

        summary = QWidget()
        sl = QVBoxLayout(summary)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(10)
        sl.addWidget(card)
        if result.get("low_data_warning"):
            sl.addWidget(self._warn_card(
                "학습 데이터가 적은 품목입니다",
                "이 종류는 원본 데이터가 부족해 오인식 가능성이 높습니다. "
                "제품에 표시된 재질을 반드시 함께 확인하세요."))
        sl.addStretch()

        sections = [("요약", summary)]
        rule = result.get("rule")
        default_idx = 0
        if rule:
            sections.append(("배출 안내", self._rule_card(rule)))
            default_idx = 1  # 원하는 건 결과가 아니라 배출 방법이므로 바로 그 탭으로
        sections.append(("다른 후보", self._candidates_card(result["candidates"])))
        self._set_sections(sections, default_index=default_idx)

    def _warn_card(self, title, body):
        card = QFrame()
        card.setStyleSheet("background-color: #FEF3C7; border: none; border-radius: 10px;")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(6)
        row = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(qta.icon("fa5s.exclamation-triangle", color="#B45309").pixmap(16, 16))
        row.addWidget(ic)
        t = QLabel(title)
        t.setStyleSheet("font-size: 15px; font-weight: bold; color: #92400E;")
        row.addWidget(t)
        row.addStretch()
        cl.addLayout(row)
        b = QLabel(body)
        b.setWordWrap(True)
        b.setStyleSheet("font-size: 14px; color: #92400E;")
        cl.addWidget(b)
        return card

    def _rule_card(self, rule):
        card = QFrame()
        card.setObjectName("Card")
        card.setGraphicsEffect(create_shadow())
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 18, 20, 18)
        cl.setSpacing(10)

        head = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(qta.icon("fa5s.trash-alt", color=C_ACCENT).pixmap(18, 18))
        head.addWidget(ic)
        t = QLabel("배출 안내")
        t.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {C_TEXT};")
        head.addWidget(t)
        head.addStretch()
        head.addWidget(make_tag(rule.get("disposal_category") or "조사 중", C_ACCENT))
        cl.addLayout(head)

        body = QLabel(rule.get("instruction") or "배출 방법 데이터를 준비 중입니다.")
        body.setObjectName("RuleBody")
        body.setWordWrap(True)
        cl.addWidget(body)

        notes = []
        if rule.get("needs_material_check"):
            notes.append("제품에 표시된 재질을 함께 확인하세요.")
        if rule.get("local_rule_required"):
            notes.append("배출 기준은 지역별로 다를 수 있습니다. 거주 지자체 안내를 확인하세요.")
        if not rule.get("verified"):
            notes.append("공식 기준 검증 전인 초안입니다.")
        for text in notes:
            row = QHBoxLayout()
            row.setSpacing(6)
            dot = QLabel()
            dot.setPixmap(qta.icon("fa5s.info-circle", color=C_MUTED).pixmap(13, 13))
            row.addWidget(dot)
            lbl = QLabel(text)
            lbl.setObjectName("Muted")
            lbl.setWordWrap(True)
            row.addWidget(lbl, 1)
            cl.addLayout(row)

        if rule.get("source"):
            src = QLabel(f"출처: {rule['source']}")
            src.setObjectName("Muted")
            src.setWordWrap(True)
            cl.addWidget(src)
        return card

    def _candidates_card(self, candidates):
        card = QFrame()
        card.setObjectName("Card")
        card.setGraphicsEffect(create_shadow())
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 16, 20, 16)
        cl.setSpacing(8)
        t = QLabel("다른 후보")
        t.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {C_TEXT};")
        cl.addWidget(t)

        for i, cand in enumerate(candidates):
            row = QHBoxLayout()
            row.setSpacing(10)
            name = QLabel(cand["name_kor"])
            name.setFixedWidth(130)
            if i == 0:
                name.setStyleSheet(f"font-weight: bold; color: {C_TEXT};")
            row.addWidget(name)
            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(int(cand["confidence"] * 1000))
            bar.setTextVisible(False)
            bar.setFixedHeight(8)
            if i > 0:
                bar.setStyleSheet(
                    f"QProgressBar::chunk {{ background-color: {C_MUTED}88; border-radius: 4px; }}")
            row.addWidget(bar, 1)
            pct = QLabel(f"{cand['confidence']:.1%}")
            pct.setObjectName("Muted")
            pct.setFixedWidth(60)
            pct.setAlignment(Qt.AlignmentFlag.AlignRight)
            row.addWidget(pct)
            cl.addLayout(row)
        return card

    # ════════════════════════════════════════════════════════════
    # 2. 배출 규칙
    # ════════════════════════════════════════════════════════════
    def build_rules(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16)
        lo.setSpacing(12)
        lo.addLayout(self._section_header("배출 규칙"))

        desc = QLabel(f"학습 대상 {len(CLASSES)}종의 배출 분류와 안내 문구입니다.")
        desc.setObjectName("Muted")
        lo.addWidget(desc)

        self.rules_scroll, self.rules_box = self._scroll_area()
        lo.addWidget(self.rules_scroll, 1)

    def rules_refresh(self):
        clear_layout(self.rules_box)
        self.rules = self._load_rules()
        if not self.rules:
            lbl = QLabel("rules/disposal_rules.json 을 읽을 수 없습니다.")
            lbl.setObjectName("Muted")
            self.rules_box.addWidget(lbl)
            return

        for slug in CLASSES:
            rule = self.rules.get(slug)
            if not rule:
                continue
            card = QFrame()
            card.setObjectName("Card")
            card.setGraphicsEffect(create_shadow())
            cl = QVBoxLayout(card)
            cl.setContentsMargins(18, 14, 18, 14)
            cl.setSpacing(7)

            head = QHBoxLayout()
            name = QLabel(rule.get("item_name") or CLASS_KOR_NAME.get(slug, slug))
            name.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {C_TEXT};")
            head.addWidget(name)
            head.addStretch()
            if slug in LOW_DATA_CLASSES:
                head.addWidget(make_tag("데이터 부족", C_RED))
            head.addWidget(make_tag(rule.get("disposal_category") or "미분류", C_ACCENT))
            cl.addLayout(head)

            divider = QFrame()
            divider.setFixedHeight(1)
            divider.setStyleSheet(f"background-color: {C_BORDER}; border: none;")
            cl.addWidget(divider)

            body = QLabel(rule.get("instruction") or "배출 방법 데이터를 준비 중입니다.")
            body.setWordWrap(True)
            body.setObjectName("RuleBody")
            cl.addWidget(body)

            flags = []
            if rule.get("needs_material_check"):
                flags.append("재질 확인 필요")
            if rule.get("local_rule_required"):
                flags.append("지자체 기준 확인")
            flags.append("검증 완료" if rule.get("verified") else "검증 전 초안")
            f = QLabel("  ·  ".join(flags))
            f.setObjectName("Muted")
            cl.addWidget(f)

            self.rules_box.addWidget(card)

    # ════════════════════════════════════════════════════════════
    # 3. 모델 성능
    # ════════════════════════════════════════════════════════════
    def build_stats(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16)
        lo.setSpacing(12)

        self.stats_combo = QComboBox()
        self.stats_combo.setFixedWidth(160)
        self.stats_combo.currentTextChanged.connect(lambda _: self.stats_refresh())
        lo.addLayout(self._section_header("모델 성능", self.stats_combo))

        self.stats_scroll, self.stats_box = self._scroll_area()
        lo.addWidget(self.stats_scroll, 1)

    def _stat_card(self, parent_layout, title, value, icon_name, icon_color):
        frame = QFrame()
        frame.setObjectName("Card")
        frame.setGraphicsEffect(create_shadow())
        lo = QVBoxLayout(frame)
        lo.setContentsMargins(16, 14, 16, 14)
        lo.setSpacing(6)
        top = QHBoxLayout()
        ic = QLabel()
        ic.setPixmap(qta.icon(icon_name, color=icon_color).pixmap(20, 20))
        top.addWidget(ic)
        t = QLabel(title)
        t.setObjectName("Muted")
        top.addWidget(t)
        top.addStretch()
        lo.addLayout(top)
        v = QLabel(value)
        v.setObjectName("StatValue")
        v.setAlignment(Qt.AlignmentFlag.AlignRight)
        lo.addWidget(v)
        parent_layout.addWidget(frame)

    def stats_refresh(self):
        # 콤보 채우기 (평가 리포트가 있는 모델만)
        reports = sorted(p.stem.replace("_eval", "") for p in RESULTS_DIR.glob("*_eval.json"))
        reports = sorted(reports, key=lambda n: (n != "improved", n))
        if [self.stats_combo.itemText(i) for i in range(self.stats_combo.count())] != reports:
            self.stats_combo.blockSignals(True)
            self.stats_combo.clear()
            self.stats_combo.addItems(reports)
            self.stats_combo.blockSignals(False)

        clear_layout(self.stats_box)
        if not reports:
            lbl = QLabel("평가 리포트가 없습니다. python -m src.evaluate --name improved 를 먼저 실행하세요.")
            lbl.setObjectName("Muted")
            self.stats_box.addWidget(lbl)
            return

        name = self.stats_combo.currentText() or reports[0]
        try:
            data = json.loads((RESULTS_DIR / f"{name}_eval.json").read_text(encoding="utf-8"))
        except Exception as exc:
            lbl = QLabel(f"리포트를 읽을 수 없습니다: {exc}")
            lbl.setObjectName("Muted")
            self.stats_box.addWidget(lbl)
            return

        report = data["classification_report"]
        macro = report.get("macro avg", {})

        row = QHBoxLayout()
        row.setSpacing(12)
        self._stat_card(row, "Accuracy", f"{data['accuracy']:.1%}", "fa5s.bullseye", C_GREEN)
        self._stat_card(row, "Macro F1", f"{macro.get('f1-score', 0):.3f}", "fa5s.balance-scale", C_ACCENT)
        self._stat_card(row, "클래스", str(len(CLASSES)), "fa5s.tags", C_MUTED)
        wrap = QWidget()
        wrap.setLayout(row)
        self.stats_box.addWidget(wrap)

        # 클래스별 성능
        self.stats_box.addWidget(self._subtitle("클래스별 성능"))
        table = QTableWidget(len(CLASSES), 5)
        table.setHorizontalHeaderLabels(["클래스", "Precision", "Recall", "F1", "장수"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for i, slug in enumerate(CLASSES):
            r = report.get(slug, {})
            recall = r.get("recall", 0)
            cells = [
                CLASS_KOR_NAME.get(slug, slug),
                f"{r.get('precision', 0):.3f}",
                f"{recall:.3f}",
                f"{r.get('f1-score', 0):.3f}",
                f"{int(r.get('support', 0)):,}",
            ]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if j > 0:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # recall이 낮은 클래스는 눈에 띄게 (전기프라이팬 등)
                if recall < 0.7:
                    item.setForeground(QColor(C_RED))
                table.setItem(i, j, item)
        table.setFixedHeight(min(len(CLASSES), 19) * 37 + 40)
        self.stats_box.addWidget(table)

        # 임계값별 커버리지
        self.stats_box.addWidget(self._subtitle("신뢰도 임계값별 커버리지"))
        note = QLabel(
            f"현재 설정: 높음 {CONF_HIGH:.2f} / 보통 {CONF_MID:.2f}  —  "
            "임계값을 올리면 답변 비율(커버리지)은 줄지만 정확도는 올라갑니다.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        self.stats_box.addWidget(note)

        rows = [r for r in data["threshold_analysis"] if r["threshold"] >= 0.3]
        th_table = QTableWidget(len(rows), 4)
        th_table.setHorizontalHeaderLabels(["임계값", "커버리지", "채택 정확도", "답변 수"])
        th_table.verticalHeader().setVisible(False)
        th_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        th_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i, r in enumerate(rows):
            cells = [f"{r['threshold']:.2f}", f"{r['coverage']:.1%}",
                     f"{r['accepted_accuracy']:.1%}", f"{r['n_accepted']:,}"]
            for j, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if abs(r["threshold"] - CONF_HIGH) < 1e-6 or abs(r["threshold"] - CONF_MID) < 1e-6:
                    item.setForeground(QColor(C_ACCENT))
                th_table.setItem(i, j, item)
        th_table.setFixedHeight(len(rows) * 37 + 40)
        self.stats_box.addWidget(th_table)

        # Confusion Matrix 이미지
        cm_path = RESULTS_DIR / f"{name}_confusion_matrix.png"
        if cm_path.exists():
            self.stats_box.addWidget(self._subtitle("Confusion Matrix"))
            card = QFrame()
            card.setObjectName("Card")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(14, 14, 14, 14)
            img = QLabel()
            img.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img.setPixmap(QPixmap(str(cm_path)).scaled(
                820, 820, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
            cl.addWidget(img)
            self.stats_box.addWidget(card)

    def _subtitle(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {C_TEXT}; margin-top: 6px;")
        return lbl

    # ════════════════════════════════════════════════════════════
    # 4. 설정
    # ════════════════════════════════════════════════════════════
    def build_settings(self, page):
        lo = QVBoxLayout(page)
        lo.setContentsMargins(20, 16, 20, 16)
        lo.setSpacing(12)
        lo.addLayout(self._section_header("설정"))

        # 모델 선택
        card = QFrame()
        card.setObjectName("Card")
        card.setGraphicsEffect(create_shadow())
        cl = QVBoxLayout(card)
        cl.setContentsMargins(20, 16, 20, 16)
        cl.setSpacing(10)
        cl.addWidget(self._subtitle("모델"))

        row = QHBoxLayout()
        row.setSpacing(10)
        self.model_combo = QComboBox()
        self.model_combo.setFixedWidth(200)
        models = available_models()
        self.model_combo.addItems(models or ["(체크포인트 없음)"])
        if self.model_name in models:
            self.model_combo.setCurrentText(self.model_name)
        row.addWidget(self.model_combo)
        btn = QPushButton("적용")
        btn.setObjectName("Primary")
        btn.clicked.connect(self.apply_model)
        row.addWidget(btn)
        row.addStretch()
        cl.addLayout(row)

        self.model_status = QLabel()
        self.model_status.setObjectName("Muted")
        cl.addWidget(self.model_status)

        if not models:
            warn = QLabel("results/checkpoints 에 .pt 파일이 없습니다. "
                          "python -m src.train --name improved ... 를 먼저 실행하세요.")
            warn.setObjectName("Muted")
            warn.setWordWrap(True)
            cl.addWidget(warn)
        lo.addWidget(card)

        # 신뢰도 임계값
        card2 = QFrame()
        card2.setObjectName("Card")
        card2.setGraphicsEffect(create_shadow())
        c2 = QVBoxLayout(card2)
        c2.setContentsMargins(20, 16, 20, 16)
        c2.setSpacing(8)
        c2.addWidget(self._subtitle("신뢰도 임계값"))
        for label, value, color, desc in [
            ("높음", CONF_HIGH, C_GREEN, "이 이상이면 배출 방법을 바로 안내합니다."),
            ("보통", CONF_MID, C_YELLOW, "이 사이면 재질 확인을 함께 안내합니다."),
            ("낮음", 0.0, C_RED, "보통 미만이면 다시 촬영하도록 안내합니다."),
        ]:
            r = QHBoxLayout()
            r.setSpacing(10)
            r.addWidget(make_tag(label, color))
            v = QLabel(f"{value:.2f}" if value else f"< {CONF_MID:.2f}")
            v.setStyleSheet(f"font-weight: bold; color: {C_TEXT};")
            v.setFixedWidth(60)
            r.addWidget(v)
            d = QLabel(desc)
            d.setObjectName("Muted")
            r.addWidget(d, 1)
            c2.addLayout(r)
        note = QLabel("값을 바꾸려면 configs/config.py 의 CONF_HIGH · CONF_MID 를 수정하세요.")
        note.setObjectName("Muted")
        c2.addWidget(note)
        lo.addWidget(card2)

        # 알려진 한계
        card3 = QFrame()
        card3.setObjectName("Card")
        card3.setGraphicsEffect(create_shadow())
        c3 = QVBoxLayout(card3)
        c3.setContentsMargins(20, 16, 20, 16)
        c3.setSpacing(8)
        c3.addWidget(self._subtitle("알려진 한계"))
        body = QLabel(
            "아래 품목은 AI-Hub 원본 데이터가 적어 오인식 가능성이 높습니다. "
            "이 품목으로 예측되면 신뢰도가 높아도 '확인 필요'로 낮추고 재질 확인을 안내합니다.\n\n"
            "다만 전기프라이팬은 원본이 41건뿐이라 밀폐용기 등으로 아예 다르게 "
            "분류되는 경우가 많고, 그때는 이 안전장치가 걸리지 않습니다.")
        body.setWordWrap(True)
        body.setObjectName("RuleBody")
        c3.addWidget(body)
        tags = QHBoxLayout()
        for slug in sorted(LOW_DATA_CLASSES):
            tags.addWidget(make_tag(CLASS_KOR_NAME.get(slug, slug), C_RED))
        tags.addStretch()
        c3.addLayout(tags)
        lo.addWidget(card3)

        lo.addStretch()

    def apply_model(self):
        name = self.model_combo.currentText()
        if name not in available_models():
            self.show_toast("사용할 수 있는 체크포인트가 없습니다.", True)
            return
        self.model_name = name
        self.predictor = None
        self.load_model()

    # ── 모델 로딩 ──────────────────────────────────────────────────────────────────
    def load_model(self):
        if not available_models():
            self._update_header_model()
            if hasattr(self, "model_status"):
                self.model_status.setText("체크포인트가 없습니다.")
            self.show_toast("학습된 모델이 없습니다. 먼저 학습을 실행하세요.", True)
            return

        self._update_header_model()
        if hasattr(self, "model_status"):
            self.model_status.setText(f"'{self.model_name}' 불러오는 중...")

        self.loader = ModelLoader(self.model_name)
        self.loader.ready.connect(self._model_ready)
        self.loader.failed.connect(self._model_failed)
        self.loader.start()

    def _model_ready(self, predictor):
        self.predictor = predictor
        self._update_header_model()
        meta = getattr(predictor, "meta", {}) or {}
        val_acc = meta.get("val_acc")
        detail = f" (val acc {val_acc:.4f})" if isinstance(val_acc, float) else ""
        if hasattr(self, "model_status"):
            self.model_status.setText(f"'{self.model_name}' 준비됨{detail}")
        self.show_toast(f"모델 '{self.model_name}' 준비 완료{detail}")

    def _model_failed(self, msg):
        self._update_header_model()
        if hasattr(self, "model_status"):
            self.model_status.setText(f"불러오기 실패: {msg}")
        self.show_toast(f"모델을 불러오지 못했습니다: {msg}", True)


def main():
    # Windows 고DPI 환경에서 아이콘/글자가 흐려지지 않도록
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

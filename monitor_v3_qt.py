#/== dev by palianitsia ==\

#|ogni|utente|e|consapevole|di|eventuale|violazione|di|tos|durante|uso|del|software|si|prega|di|usare|software|solo|in|ambiente|di|test|

import sys
import asyncio
import threading
import aiohttp
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QTextEdit, QSplitter, QFrame,
    QRubberBand, QSizePolicy
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineProfile, QWebEnginePage, QWebEngineCookieStore
from PyQt6.QtCore import (
    Qt, QUrl, QTimer, QThread, pyqtSignal, QObject, QRect, QSize, QPoint
)
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter, QPen, QFont, QIcon

DOMAIN = "it"
HOME_URL = f"https://{DOMAIN}.bidoo.com"
USER_AGENT = "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"

CYAN_LOWER = np.array([85, 180, 150])
CYAN_UPPER = np.array([95, 255, 255])
RED_LOWER  = np.array([0, 150, 150])
RED_UPPER  = np.array([10, 255, 255])


def build_url(id_asta):
    return f"https://{DOMAIN}.bidoo.com/bid.php?AID={id_asta}&sup=0&shock=0"


def get_headers(dess):
    return {
        "Cookie": f"dess={dess};",
        "User-Agent": USER_AGENT,
        "Referer": f"https://{DOMAIN}.bidoo.com/",
        "Accept": "*/*",
    }


def qpixmap_to_bgr(pixmap):
    img = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(h * w * 3)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3)).copy()
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def count_red_pixels(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.countNonZero(cv2.inRange(hsv, RED_LOWER, RED_UPPER))


def detect_cyan(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return cv2.countNonZero(cv2.inRange(hsv, CYAN_LOWER, CYAN_UPPER)) > 50


class AreaSelector(QWidget):
    area_selected = pyqtSignal(int, int, int, int)

    def __init__(self, pixmap, label):
        super().__init__()
        self.pixmap = pixmap
        self.label = label
        self.origin = QPoint()
        self.selection = QRect()
        self.drawing = False
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()
        self.setCursor(Qt.CursorShape.CrossCursor)

    def paintEvent(self, e):
        p = QPainter(self)
        p.drawPixmap(0, 0, self.pixmap)
        overlay = QColor(0, 0, 0, 100)
        p.fillRect(self.rect(), overlay)
        if not self.selection.isNull():
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            p.fillRect(self.selection, QColor(0, 0, 0, 0))
            p.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            pen = QPen(QColor("#e74c3c"), 2)
            p.setPen(pen)
            p.drawRect(self.selection)
        p.setPen(QColor("yellow"))
        p.setFont(QFont("Courier New", 14, QFont.Weight.Bold))
        p.drawText(self.rect(), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter, self.label)

    def mousePressEvent(self, e):
        self.origin = e.pos()
        self.selection = QRect(self.origin, QSize())
        self.drawing = True

    def mouseMoveEvent(self, e):
        if self.drawing:
            self.selection = QRect(self.origin, e.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, e):
        self.drawing = False

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self.selection.width() > 5 and self.selection.height() > 5:
                self.area_selected.emit(self.selection.x(), self.selection.y(), self.selection.width(), self.selection.height())
            self.close()
        elif e.key() == Qt.Key.Key_Escape:
            self.area_selected.emit(0, 0, 0, 0)
            self.close()


class MonitorWorker(QObject):
    log_signal    = pyqtSignal(str)
    status_signal = pyqtSignal(int, bool, bool)
    done_signal   = pyqtSignal()
    grab_signal   = pyqtSignal()

    def __init__(self, browser, id_asta, dess, soglia_min, soglia_max, rect_timer, rect_vincendo):
        super().__init__()
        self.browser       = browser
        self.id_asta       = id_asta
        self.dess          = dess
        self.soglia_min    = soglia_min
        self.soglia_max    = soglia_max
        self.rect_timer    = rect_timer
        self.rect_vincendo = rect_vincendo
        self.running       = False
        self._frame        = None
        self._frame_event  = threading.Event()

    def receive_frame(self, bgr):
        self._frame = bgr
        self._frame_event.set()

    def crop(self, bgr, rect):
        if rect is None or rect.isNull():
            return None
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        bh, bw = bgr.shape[:2]
        x1 = max(0, min(x, bw - 1))
        y1 = max(0, min(y, bh - 1))
        x2 = max(0, min(x + w, bw))
        y2 = max(0, min(y + h, bh))
        return bgr[y1:y2, x1:x2]

    def request_frame(self):
        self._frame_event.clear()
        self._frame = None
        self.grab_signal.emit()
        self._frame_event.wait(timeout=2.0)
        return self._frame

    async def fetch(self):
        url = build_url(self.id_asta)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=get_headers(self.dess),
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    body = await resp.text()
                    self.log_signal.emit(f"Fetch → {resp.status} | {body[:80]}")
        except Exception as ex:
            self.log_signal.emit(f"Errore fetch: {ex}")

    async def loop(self):
        already_sent = False
        self.running = True
        while self.running:
            bgr = await asyncio.get_event_loop().run_in_executor(None, self.request_frame)
            if bgr is None:
                await asyncio.sleep(0.05)
                continue

            crop_t = self.crop(bgr, self.rect_timer)
            red = count_red_pixels(crop_t) if crop_t is not None else 0

            vincendo = False
            if self.rect_vincendo and not self.rect_vincendo.isNull():
                crop_v = self.crop(bgr, self.rect_vincendo)
                if crop_v is not None:
                    vincendo = detect_cyan(crop_v)

            in_range = self.soglia_min <= red <= self.soglia_max
            self.status_signal.emit(red, in_range, vincendo)

            if in_range and not vincendo:
                if not already_sent:
                    self.log_signal.emit(f"Timer in range ({red} px) → fetch!")
                    await self.fetch()
                    already_sent = True
            elif in_range and vincendo:
                if not already_sent:
                    self.log_signal.emit(f"In range ma stai vincendo — skip.")
                    already_sent = True
            else:
                already_sent = False

            await asyncio.sleep(0.05)

        self.done_signal.emit()

    def start_loop(self):
        asyncio.run(self.loop())

    def stop(self):
        self.running = False


STYLE = """
QMainWindow, QWidget#root {
    background: #0d1b2a;
}
QWidget {
    background: #0d1b2a;
    color: #eaeaea;
    font-family: 'Courier New';
    font-size: 11px;
}
QLineEdit {
    background: #1a2d42;
    border: none;
    border-radius: 4px;
    padding: 4px 8px;
    color: #eaeaea;
    font-family: 'Courier New';
    font-size: 11px;
}
QPushButton {
    border: none;
    border-radius: 4px;
    padding: 5px 12px;
    font-family: 'Courier New';
    font-size: 11px;
    font-weight: bold;
}
QPushButton#btn_go {
    background: #27ae60;
    color: white;
    min-width: 80px;
}
QPushButton#btn_go:hover { background: #2ecc71; }
QPushButton#btn_go:disabled { background: #1a4a2e; color: #555; }
QPushButton#btn_stop {
    background: #e74c3c;
    color: white;
    min-width: 80px;
}
QPushButton#btn_stop:hover { background: #ff6b5b; }
QPushButton#btn_stop:disabled { background: #4a1a1a; color: #555; }
QPushButton#btn_save {
    background: #e74c3c;
    color: white;
    min-width: 28px;
    padding: 5px 8px;
}
QPushButton#btn_save:hover { background: #ff6b5b; }
QPushButton#btn_sel {
    background: #2980b9;
    color: white;
    font-size: 10px;
}
QPushButton#btn_sel:hover { background: #3498db; }
QTextEdit {
    background: #060f18;
    color: #7fb3d3;
    border: none;
    font-family: 'Courier New';
    font-size: 10px;
    padding: 4px;
}
QLabel#status_bar {
    background: #0a1520;
    color: #7fb3d3;
    padding: 3px 10px;
    font-size: 10px;
}
QLabel#lbl_area {
    color: #3498db;
    font-size: 10px;
}
QSplitter::handle {
    background: #1a2d42;
    height: 4px;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bidoo Timer Monitor")
        self.resize(1100, 820)
        self.setStyleSheet(STYLE)

        self.rect_timer    = None
        self.rect_vincendo = None
        self.worker        = None
        self.worker_thread = None
        self._pending_selection = None

        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        main_layout = QVBoxLayout(root)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # === Top control bar ===
        bar = QWidget()
        bar.setFixedHeight(90)
        bar_layout = QVBoxLayout(bar)
        bar_layout.setContentsMargins(10, 6, 10, 6)
        bar_layout.setSpacing(4)

        # Row 1: ID + DESS
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row1.addWidget(QLabel("ID Asta:"))
        self.id_input = QLineEdit()
        self.id_input.setPlaceholderText("es. 123456")
        self.id_input.setFixedWidth(90)
        row1.addWidget(self.id_input)
        row1.addSpacing(10)
        row1.addWidget(QLabel("DESS:"))
        self.dess_input = QLineEdit()
        self.dess_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.dess_input.setPlaceholderText("cookie dess")
        self.dess_input.setFixedWidth(200)
        row1.addWidget(self.dess_input)
        btn_save = QPushButton("✓")
        btn_save.setObjectName("btn_save")
        btn_save.clicked.connect(self.on_save)
        row1.addWidget(btn_save)
        row1.addSpacing(16)
        row1.addWidget(QLabel("Min px:"))
        self.min_input = QLineEdit("660")
        self.min_input.setFixedWidth(48)
        row1.addWidget(self.min_input)
        row1.addWidget(QLabel("Max px:"))
        self.max_input = QLineEdit("770")
        self.max_input.setFixedWidth(48)
        row1.addWidget(self.max_input)
        row1.addStretch()
        bar_layout.addLayout(row1)

        # Row 2: area selectors + go/stop
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        btn_sel_timer = QPushButton("⊡ Seleziona area TIMER")
        btn_sel_timer.setObjectName("btn_sel")
        btn_sel_timer.clicked.connect(lambda: self.start_selection("timer"))
        row2.addWidget(btn_sel_timer)
        self.lbl_timer = QLabel("non selezionata")
        self.lbl_timer.setObjectName("lbl_area")
        row2.addWidget(self.lbl_timer)
        row2.addSpacing(10)
        btn_sel_vinc = QPushButton("⊡ Seleziona area STAI VINCENDO")
        btn_sel_vinc.setObjectName("btn_sel")
        btn_sel_vinc.clicked.connect(lambda: self.start_selection("vincendo"))
        row2.addWidget(btn_sel_vinc)
        self.lbl_vinc = QLabel("non selezionata")
        self.lbl_vinc.setObjectName("lbl_area")
        row2.addWidget(self.lbl_vinc)
        row2.addStretch()
        self.btn_go = QPushButton("▶  GO")
        self.btn_go.setObjectName("btn_go")
        self.btn_go.clicked.connect(self.on_go)
        row2.addWidget(self.btn_go)
        self.btn_stop = QPushButton("■  STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop)
        row2.addWidget(self.btn_stop)
        bar_layout.addLayout(row2)

        main_layout.addWidget(bar)

        # === Splitter: browser + log ===
        splitter = QSplitter(Qt.Orientation.Vertical)

        self.browser = QWebEngineView()
        self.profile = QWebEngineProfile.defaultProfile()
        self.profile.setHttpUserAgent(USER_AGENT)
        self.browser.setUrl(QUrl(HOME_URL))
        self.browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.browser.urlChanged.connect(self.on_url_changed)
        self.profile.cookieStore().cookieAdded.connect(self.on_cookie_added)
        splitter.addWidget(self.browser)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self.status_bar = QLabel("In attesa...")
        self.status_bar.setObjectName("status_bar")
        bottom_layout.addWidget(self.status_bar)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(110)
        bottom_layout.addWidget(self.log_box)
        splitter.addWidget(bottom)

        splitter.setStretchFactor(0, 1)
        splitter.setSizes([600, 160])
        main_layout.addWidget(splitter)

        self.dess = ""

    def on_cookie_added(self, cookie):
        name = cookie.name().data().decode("utf-8", errors="ignore")
        if name == "dess":
            value = cookie.value().data().decode("utf-8", errors="ignore")
            self.dess = value
            self.dess_input.setText(value)
            self.log(f"Cookie DESS rilevato: {value[:12]}...")

    def on_url_changed(self, url):
        url_str = url.toString()
        if "auction.php?a=" in url_str:
            last = url_str.split("_")[-1]
            digits = ''.join(filter(str.isdigit, last))
            id_candidate = digits[-8:] if len(digits) >= 8 else digits
            if id_candidate:
                self.id_input.setText(id_candidate)
                self.log(f"ID asta rilevato dall'URL: {id_candidate}")

    def on_save(self):
        d = self.dess_input.text().strip()
        if d:
            self.dess = d
            self.log(f"DESS salvato: {d[:12]}...")
        else:
            self.log("DESS vuoto!")

    def start_selection(self, which):
        self._pending_selection = which
        pixmap = self.browser.grab()
        label = (
            "AREA TIMER  |  Trascina → ENTER conferma  |  ESC annulla"
            if which == "timer" else
            "AREA 'STAI VINCENDO'  |  Trascina → ENTER conferma  |  ESC salta"
        )
        self._selector = AreaSelector(pixmap, label)
        self._selector.area_selected.connect(self.on_area_selected)

    def on_area_selected(self, x, y, w, h):
        which = self._pending_selection
        rect = QRect(x, y, w, h)
        if rect.isNull() or (w == 0 and h == 0):
            self.log(f"Selezione {which} annullata.")
            return
        if which == "timer":
            self.rect_timer = rect
            self.lbl_timer.setText(f"{rect.x()},{rect.y()} {rect.width()}×{rect.height()}")
            self.log(f"Area timer impostata: {rect.x()},{rect.y()} {rect.width()}×{rect.height()}")
        else:
            self.rect_vincendo = rect
            self.lbl_vinc.setText(f"{rect.x()},{rect.y()} {rect.width()}×{rect.height()}")
            self.log(f"Area vincendo impostata: {rect.x()},{rect.y()} {rect.width()}×{rect.height()}")

    def on_go(self):
        id_asta = self.id_input.text().strip()
        if not id_asta:
            self.log("Inserisci ID asta!")
            return
        if not self.dess:
            self.log("Salva il cookie DESS prima!")
            return
        if not self.rect_timer:
            self.log("Seleziona l'area del timer!")
            return
        try:
            soglia_min = int(self.min_input.text())
            soglia_max = int(self.max_input.text())
        except ValueError:
            self.log("Soglia min/max devono essere interi!")
            return
        if soglia_min >= soglia_max:
            self.log("Soglia min deve essere < soglia max!")
            return

        self.btn_go.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log(f"Avvio → ID: {id_asta} | URL: {build_url(id_asta)}")
        self.log(f"Soglia: {soglia_min}–{soglia_max} px rossi")

        self.worker = MonitorWorker(
            self.browser, id_asta, self.dess,
            soglia_min, soglia_max,
            self.rect_timer, self.rect_vincendo
        )
        self.worker.log_signal.connect(self.log)
        self.worker.status_signal.connect(self.update_status)
        self.worker.done_signal.connect(self.on_worker_done)
        self.worker.grab_signal.connect(self.do_grab)

        self.worker_thread = threading.Thread(target=self.worker.start_loop, daemon=True)
        self.worker_thread.start()

    def on_stop(self):
        if self.worker:
            self.worker.stop()
        self.btn_go.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.log("Monitoraggio fermato.")

    def do_grab(self):
        QTimer.singleShot(0, self._perform_grab)

    def _perform_grab(self):
        try:
            pixmap = self.browser.grab()
            bgr = qpixmap_to_bgr(pixmap)
            if self.worker:
                self.worker.receive_frame(bgr)
        except Exception as e:
            if self.worker:
                self.worker.receive_frame(None)

    def on_worker_done(self):
        self.btn_go.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def log(self, msg):
        self.log_box.append(msg)

    def update_status(self, red, in_range, vincendo):
        r = "✓ IN RANGE" if in_range else "○ fuori"
        v = "  |  🏆 VINCENDO" if vincendo else ""
        self.status_bar.setText(f"Red px: {red}  |  {r}{v}")


import os
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-features=AttributionReporting"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

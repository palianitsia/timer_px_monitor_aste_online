import os
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-features=AttributionReporting --disable-logging --log-level=3"

import sys
import asyncio
import threading
import time
import aiohttp
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QTextEdit, QSplitter, QFrame,
    QRubberBand, QSizePolicy, QSlider, QRadioButton, QButtonGroup
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


def build_url(id_asta, domain="it"):
    return f"https://{domain}.bidoo.com/bid.php?AID={id_asta}&sup=0&shock=0"


def get_headers(dess, domain="it"):
    return {
        "Cookie": f"dess={dess};",
        "User-Agent": USER_AGENT,
        "Referer": f"https://{domain}.bidoo.com/",
        "Accept": "*/*",
        "Accept-Language": "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "x-requested-with": "XMLHttpRequest",
    }



def parse_data_response(raw):
    try:
        import re, time as _time
        timestamps = re.findall(r'\d{10}', raw)
        if len(timestamps) < 2:
            return None
        time_server  = int(timestamps[0])
        scadenza     = int(timestamps[1])
        receipt_time = _time.time()
        diff_int     = scadenza - time_server

        parts = raw.split("*")
        if len(parts) < 2:
            return None
        block = parts[1]
        m = re.search(r"\[(.+)\]", block)
        if not m:
            return None
        inner = m.group(1)
        dett = inner.split(",")[0].split(";")
        if len(dett) < 5:
            return None
        stato       = dett[1] if len(dett) > 1 else "?"
        prezzo_cent = int(dett[3]) if len(dett) > 3 and dett[3].lstrip("-").isdigit() else 0
        vincitore   = dett[4] if len(dett) > 4 else "?"
        type_off    = dett[5] if len(dett) > 5 else "0"
        return {
            "time_server":  time_server,
            "stato":        stato,
            "scadenza":     scadenza,
            "diff_int":     diff_int,
            "receipt_time": receipt_time,
            "prezzo":       prezzo_cent / 100,
            "vincitore":    vincitore,
            "type_off":     type_off,
        }
    except Exception:
        return None


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
    log_signal     = pyqtSignal(str)
    status_signal  = pyqtSignal(int, bool, bool)
    done_signal    = pyqtSignal()
    grab_signal    = pyqtSignal()
    data_signal    = pyqtSignal(str)
    auction_signal = pyqtSignal(dict)

    def __init__(self, browser, id_asta, dess, soglia_min, soglia_max, rect_timer, rect_vincendo, mode='pixel', domain='it'):
        super().__init__()
        self.browser       = browser
        self.id_asta       = id_asta
        self.dess          = dess
        self.domain        = domain
        self.soglia_min    = soglia_min
        self.soglia_max    = soglia_max
        self.rect_timer    = rect_timer
        self.rect_vincendo = rect_vincendo
        self.mode          = mode
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
        url = build_url(self.id_asta, self.domain)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    headers=get_headers(self.dess, self.domain),
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    body = await resp.text()
                    self.log_signal.emit(f"Fetch → {resp.status} | {body[:80]}")
        except Exception as ex:
            self.log_signal.emit(f"Errore fetch: {ex}")

    async def fetch_data(self):
        url = f"https://{self.domain}.bidoo.com/data.php?ALL={self.id_asta}&LISTID=0"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=get_headers(self.dess),
                    timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    return await resp.text()
        except Exception:
            return None

    async def loop(self):
        already_sent = False
        self.running = True
        while self.running:
            if self.mode == "data":
                await self.loop_data_mode()
                break
            else:
                await self.loop_pixel_tick()
        self.done_signal.emit()

    async def loop_pixel_tick(self):
        already_sent = False
        while self.running and self.mode == "pixel":
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
                    self.log_signal.emit(f"[PIXEL] Timer in range ({red} px) → fetch!")
                    await self.fetch()
                    already_sent = True
            elif in_range and vincendo:
                if not already_sent:
                    self.log_signal.emit(f"[PIXEL] In range ma già vincendo — skip.")
                    already_sent = True
            else:
                already_sent = False

            await asyncio.sleep(0.05)

    async def loop_data_mode(self):
        already_sent = False
        fire_scheduled = False
        while self.running and self.mode == "data":
            t_request = time.time()
            raw = await self.fetch_data()
            t_response = time.time()

            if raw:
                if "nosess" in raw:
                    self.log_signal.emit("[DATA] Sessione scaduta — rifare login.")
                    await asyncio.sleep(2)
                    continue

                parsed = parse_data_response(raw)
                if parsed:
                    self.auction_signal.emit(parsed)

                    # Network round-trip latency (ms)
                    latency_ms = (t_response - t_request) * 1000

                    # Real-time timer: scadenza unix - ora locale (compensato da latency)
                    now_ms = time.time() * 1000
                    scadenza_ms = parsed["scadenza"] * 1000
                    timer_ms = scadenza_ms - now_ms

                    vincendo = bool(parsed["vincitore"] and parsed["vincitore"] != "nessun offerente")
                    self.status_signal.emit(int(max(timer_ms, 0)), timer_ms > 0, vincendo)

                    if parsed["stato"] == "ON" and not already_sent and not fire_scheduled:
                        if vincendo:
                            # reset se qualcuno ha puntato dopo di noi
                            already_sent = False
                        elif timer_ms > 500:
                            # Calcola attesa precisa: vogliamo inviare a esattamente 500ms
                            # Compensa latency stimata per il prossimo fetch
                            wait_ms = timer_ms - 500 - latency_ms
                            if wait_ms > 0:
                                fire_scheduled = True
                                self.log_signal.emit(f"[DATA] Timer {timer_ms:.0f}ms — attendo {wait_ms:.0f}ms poi fire")
                                await asyncio.sleep(wait_ms / 1000)
                                fire_scheduled = False
                                if not self.running:
                                    break
                                # Ricontrolla stato prima di inviare
                                raw2 = await self.fetch_data()
                                if raw2 and "nosess" not in raw2:
                                    parsed2 = parse_data_response(raw2)
                                    if parsed2:
                                        self.auction_signal.emit(parsed2)
                                        vincendo2 = bool(parsed2["vincitore"] and parsed2["vincitore"] != "nessun offerente")
                                        now_ms2 = time.time() * 1000
                                        timer_ms2 = parsed2["scadenza"] * 1000 - now_ms2
                                        self.status_signal.emit(int(max(timer_ms2, 0)), timer_ms2 > 0, vincendo2)
                                        if parsed2["stato"] == "ON" and not vincendo2:
                                            self.log_signal.emit(f"[DATA] Fire! Timer {timer_ms2:.0f}ms → fetch!")
                                            await self.fetch()
                                            already_sent = True
                                        else:
                                            self.log_signal.emit(f"[DATA] Già vincendo al momento del fire — skip.")
                                continue
                        elif 0 < timer_ms <= 500:
                            # Già dentro la finestra, invia subito
                            if not vincendo:
                                self.log_signal.emit(f"[DATA] Timer {timer_ms:.0f}ms (già in finestra) → fetch!")
                                await self.fetch()
                                already_sent = True
                    elif parsed["stato"] != "ON" or timer_ms <= 0:
                        already_sent = False
                        fire_scheduled = False

            await asyncio.sleep(0.5)

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
    width: 4px;
}
QLabel#lbl_section {
    background: #0a1520;
    color: #3498db;
    font-size: 10px;
    padding: 2px 6px;
    font-weight: bold;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bidoo Timer Monitor")
        self.resize(1100, 820)
        self.setStyleSheet(STYLE)

        self.rect_timer         = None
        self.rect_vincendo      = None
        self.worker             = None
        self.worker_thread      = None
        self._pending_selection = None
        self._fire_handle       = None
        self._already_sent      = False
        self._active_id         = None
        self._last_scadenza     = 0
        self._prev_active_id    = None
        self.domain             = "it"
        self.username           = ""

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
        bar.setFixedHeight(118)
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
        self.btn_sel_vinc = QPushButton("⊡ Seleziona area STAI VINCENDO")
        btn_sel_vinc = self.btn_sel_vinc
        btn_sel_vinc.setObjectName("btn_sel")
        btn_sel_vinc.clicked.connect(lambda: self.start_selection("vincendo"))
        row2.addWidget(btn_sel_vinc)
        self.lbl_vinc = QLabel("non selezionata")
        self.lbl_vinc.setObjectName("lbl_area")
        row2.addWidget(self.lbl_vinc)
        row2.addStretch()
        self.radio_pixel = QRadioButton("Modalità PIXEL")
        self.radio_data  = QRadioButton("Modalità DATA.PHP")
        self.radio_pixel.setChecked(True)
        self._mode_group = QButtonGroup()
        self._mode_group.addButton(self.radio_pixel, 0)
        self._mode_group.addButton(self.radio_data, 1)
        row2.addWidget(self.radio_pixel)
        row2.addWidget(self.radio_data)
        row2.addSpacing(10)
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
        self.browser.loadFinished.connect(self.on_load_finished)
        self.profile.cookieStore().cookieAdded.connect(self.on_cookie_added)
        splitter.addWidget(self.browser)

        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(0)

        self.status_bar = QLabel("In attesa...")
        self.status_bar.setObjectName("status_bar")
        bottom_layout.addWidget(self.status_bar)

        # === Auction info panel ===
        info_w = QWidget()
        info_w.setFixedHeight(36)
        info_l = QHBoxLayout(info_w)
        info_l.setContentsMargins(10, 2, 10, 2)
        info_l.setSpacing(20)

        def make_info(label, val_width=60):
            lbl = QLabel(label)
            lbl.setStyleSheet("color:#3498db; font-size:10px; font-family:'Courier New';")
            lbl.setFixedWidth(len(label) * 7 + 4)
            val = QLabel("—")
            val.setFixedWidth(val_width)
            val.setStyleSheet("color:#eaeaea; font-size:11px; font-family:'Courier New'; font-weight:bold;")
            info_l.addWidget(lbl)
            info_l.addWidget(val)
            return val

        self.val_timer   = make_info("TIMER:",      80)
        self.val_stato   = make_info("STATO:",       40)
        self.val_prezzo  = make_info("PREZZO:",      60)
        self.val_vinc    = make_info("VINCITORE:",  110)
        self.val_pagate  = make_info("SALDO BIDS:",  50)
        self.val_gratis  = make_info("GRATIS:",      40)
        self.val_manuali = make_info("MANUALI:",     40)
        self.val_auto    = make_info("AUTO:",         40)
        # Init bid labels to 0 so int() never fails
        self.val_pagate.setText("0")
        self.val_gratis.setText("0")
        self.val_manuali.setText("0")
        self.val_auto.setText("0")
        info_l.addSpacing(16)
        lbl_sec = QLabel("Sec:")
        lbl_sec.setStyleSheet("color:#3498db; font-size:10px; font-family:'Courier New';")
        info_l.addWidget(lbl_sec)
        self.fire_slider = QSlider(Qt.Orientation.Horizontal)
        self.fire_slider.setMinimum(0)
        self.fire_slider.setMaximum(24)
        self.fire_slider.setValue(1)
        self.fire_slider.setFixedWidth(120)
        self.fire_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.fire_slider.setTickInterval(2)
        self.fire_slider.valueChanged.connect(self._on_slider_change)
        info_l.addWidget(self.fire_slider)
        self.lbl_fire_val = QLabel("0.5s")
        self.lbl_fire_val.setFixedWidth(32)
        self.lbl_fire_val.setStyleSheet("color:#e74c3c; font-weight:bold; font-family:'Courier New'; font-size:11px;")
        info_l.addWidget(self.lbl_fire_val)
        info_l.addStretch()
        bottom_layout.addWidget(info_w)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)

        left_w = QWidget()
        left_l = QVBoxLayout(left_w)
        left_l.setContentsMargins(0, 2, 0, 0)
        left_l.setSpacing(1)
        lbl_log = QLabel("  LOG")
        lbl_log.setObjectName("lbl_section")
        left_l.addWidget(lbl_log)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        left_l.addWidget(self.log_box)
        h_splitter.addWidget(left_w)

        right_w = QWidget()
        right_l = QVBoxLayout(right_w)
        right_l.setContentsMargins(0, 2, 0, 0)
        right_l.setSpacing(1)
        lbl_data = QLabel("  TRAFFICO data.php")
        lbl_data.setObjectName("lbl_section")
        right_l.addWidget(lbl_data)
        self.data_box = QTextEdit()
        self.data_box.setReadOnly(True)
        right_l.addWidget(self.data_box)
        h_splitter.addWidget(right_w)

        h_splitter.setSizes([400, 400])
        bottom_layout.addWidget(h_splitter)
        splitter.addWidget(bottom)

        splitter.setStretchFactor(0, 1)
        splitter.setSizes([600, 160])
        main_layout.addWidget(splitter)

        self.dess = ""
        self._data_timer = QTimer()
        self._data_timer.setInterval(300)
        self._data_timer.timeout.connect(self._poll_data)
        self._data_timer.start()

    @property
    def winning_label(self):
        labels = {
            "it": "STAI VINCENDO",
            "es": "ESTÁS GANANDO",
            "pt": "ESTÁS A GANHAR",
            "fr": "VOUS GAGNEZ",
            "de": "SIE GEWINNEN",
            "pl": "WYGRYWASZ",
        }
        return labels.get(self.domain, "STAI VINCENDO")

    def on_cookie_added(self, cookie):
        name = cookie.name().data().decode("utf-8", errors="ignore")
        if name == "dess":
            value = cookie.value().data().decode("utf-8", errors="ignore")
            self.dess = value
            self.dess_input.setText(value)
            # Extract domain from cookie: e.g. ".it.bidoo.com" → "it"
            raw_domain = cookie.domain().lower()
            import re
            m = re.search(r'\.([a-z]{2,5})\.bidoo\.com', raw_domain)
            if m:
                self.domain = m.group(1)
                self.log(f"Cookie DESS rilevato: {value[:12]}... | dominio: {self.domain}.bidoo.com")
                self.btn_sel_vinc.setText(f"⊡ Seleziona area {self.winning_label}")
            else:
                self.log(f"Cookie DESS rilevato: {value[:12]}...")
            # Fetch username after login
            threading.Thread(target=self._fetch_username, daemon=True).start()

    def _fetch_username(self):
        import asyncio
        async def _get():
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"https://{self.domain}.bidoo.com/ajax/get_logged_user.php",
                        headers={"Cookie": f"dess={self.dess};", "User-Agent": USER_AGENT},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.username = data.get("username", "")
                            self.log(f"Username rilevato: {self.username}")
                        else:
                            self.log(f"get_logged_user → status {resp.status}")
            except Exception as ex:
                self.log(f"Errore get username: {ex}")
        asyncio.run(_get())

    def on_url_changed(self, url):
        url_str = url.toString()
        if "auction.php?a=" in url_str:
            last = url_str.split("_")[-1]
            digits = ''.join(filter(str.isdigit, last))
            id_candidate = digits[-8:] if len(digits) >= 8 else digits
            if id_candidate:
                self.id_input.setText(id_candidate)
                self.log(f"ID asta rilevato dall'URL: {id_candidate}")

    def _poll_data(self):
        js = """
        (function() {
            var q = window.__qt_data_queue;
            if (!q || q.length === 0) return '';
            var out = q.join('|||');
            window.__qt_data_queue = [];
            return out;
        })()
        """
        self.browser.page().runJavaScript(js, self._handle_data_poll)

    def _handle_data_poll(self, result):
        if not result:
            return
        for entry in result.split('|||'):
            entry = entry.strip()
            if entry:
                self.on_data_received(entry)

    def on_load_finished(self, ok):
        if ok:
            self.inject_data_monitor()

    def inject_data_monitor(self):
        js = """
        (function() {
            if (window.__dataMonitorInjected) return;
            window.__dataMonitorInjected = true;
            window.__qt_data_queue = [];
            const origOpen = XMLHttpRequest.prototype.open;
            const origSend = XMLHttpRequest.prototype.send;
            XMLHttpRequest.prototype.open = function(method, url, ...args) {
                this.__url = url;
                return origOpen.apply(this, [method, url, ...args]);
            };
            XMLHttpRequest.prototype.send = function(...args) {
                this.addEventListener('load', function() {
                    if (this.__url && this.__url.includes('data.php')) {
                        window.__qt_data_queue.push(this.__url + ' | ' + this.responseText.substring(0, 400));
                    }
                });
                return origSend.apply(this, args);
            };
            const origFetch = window.fetch;
            window.fetch = function(input, init) {
                const url = typeof input === 'string' ? input : (input && input.url ? input.url : '');
                return origFetch.apply(this, arguments).then(function(resp) {
                    if (url && url.includes('data.php')) {
                        resp.clone().text().then(function(text) {
                            window.__qt_data_queue.push(url + ' | ' + text.substring(0, 400));
                        });
                    }
                    return resp;
                });
            };
        })();
        """
        self.browser.page().runJavaScript(js)

    def on_data_received(self, msg):
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        if " | " in msg:
            raw = msg.split(" | ", 1)[1]
            parsed = parse_data_response(raw)
            if parsed:
                self.update_auction_info(parsed)
                if self.radio_data.isChecked() and self._active_id and parsed.get("stato") == "ON":
                    self._handle_fire_logic(parsed)
        # Keep only last data.php entry in the box
        self.data_box.setPlainText(f"[{ts}] {msg}")
        self.data_box.verticalScrollBar().setValue(
            self.data_box.verticalScrollBar().maximum()
        )

    def _handle_fire_logic(self, parsed):
        import time as _t
        current_winner = parsed.get("vincitore", "") or ""
        vincendo = bool(self.username and current_winner.strip().lower() == self.username.strip().lower())
        # elapsed ms since server response was received
        elapsed   = _t.time() - parsed["receipt_time"]
        timer_sec = parsed["diff_int"] - elapsed
        offset_sec = self.fire_slider.value() * 0.5

        # Nuovo ciclo asta: scadenza cambiata → reset solo fire state
        scadenza = parsed.get("scadenza", 0)
        if scadenza != self._last_scadenza:
            self._last_scadenza = scadenza
            self._already_sent = False
            if self._fire_handle:
                self._fire_handle.stop()
                self._fire_handle = None
        # Nuovo id_asta → reset anche valori bid GUI
        if self._active_id != self._prev_active_id:
            self._prev_active_id = self._active_id
            self.val_pagate.setText("0")
            self.val_gratis.setText("0")
            self.val_manuali.setText("0")
            self.val_auto.setText("0")

        # Asta scaduta
        if timer_sec <= 0:
            self._already_sent = False
            return

        if vincendo:
            if self._fire_handle:
                self._fire_handle.stop()
                self._fire_handle = None
                self.log(f"[DATA] {self.winning_label} — fire annullato.")
            self._already_sent = False
            return

        if self._already_sent:
            return

        if self._fire_handle:
            return

        wait_ms = int((timer_sec - offset_sec) * 1000)
        if wait_ms <= 0:
            self.log(f"[DATA] Timer {timer_sec*1000:.0f}ms — fire immediato! (offset {offset_sec}s)")
            self._do_fire()
        else:
            self.log(f"[DATA] Timer {timer_sec*1000:.0f}ms — fire tra {wait_ms}ms (offset {offset_sec}s)")
            self._fire_handle = QTimer()
            self._fire_handle.setSingleShot(True)
            self._fire_handle.setInterval(wait_ms)
            self._fire_handle.timeout.connect(self._on_fire_timeout)
            self._fire_handle.start()

    def _on_fire_timeout(self):
        self._fire_handle = None
        if not self._already_sent and self._active_id:
            self.log(f"[DATA] Fire! → fetch bid")
            self._do_fire()

    def _do_fire(self):
        self._already_sent = True
        id_asta = self._active_id
        # Build URL using domain from current browser URL — always correct, no default fallback
        current_url = self.browser.url().toString()
        import re as _re
        m = _re.search(r'https?://([a-z]{2,5})\.bidoo\.com', current_url)
        domain = m.group(1) if m else self.domain
        url = build_url(id_asta, domain)
        js = f"""
        (function() {{
            var bidUrl = '{url}';
            fetch(bidUrl, {{
                method: 'POST',
                credentials: 'include',
                headers: {{
                    'Accept': '*/*',
                    'Referer': window.location.origin + '/'
                }}
            }})
            .then(r => r.text())
            .then(t => {{
                window.__bid_result = t;
                console.log('[BID]', t);
            }})
            .catch(e => {{
                window.__bid_result = 'ERROR: ' + e;
                console.log('[BID ERROR]', e);
            }});
        }})();
        """
        self.browser.page().runJavaScript(js)
        QTimer.singleShot(1500, self._check_bid_result)

    def _check_bid_result(self):
        self.browser.page().runJavaScript(
            "(function(){ var r = window.__bid_result; window.__bid_result = null; return r || ''; })()",
            self._on_bid_result
        )

    def _on_bid_result(self, result):
        if not result:
            self.log("[DATA] Bid inviato (no response)")
            return

        result = result.strip()
        self.log(f"[DATA] Bid response raw → {result[:120]}")

        if not result:
            self.log("[DATA] ❌ Risposta vuota dal server")
            return

        parts = result.split("|")

        if len(parts) < 7:
            if "nosess" in result.lower():
                self.log("[DATA] ❌ Sessione scaduta — rifare login")
            elif result.startswith("NOBB"):
                self.log("[DATA] ❌ Nessuna puntata disponibile (NOBB)")
            else:
                self.log(f"[DATA] ⚠️ Risposta malformata: {result[:80]}")
            return

        status       = parts[0].lower()
        saldo        = int(parts[1]) if parts[1].isdigit() else 0
        puntate_usate = int(parts[4]) if parts[4].isdigit() else 0
        id_asta_srv  = parts[6] if len(parts) > 6 else "?"

        if status == "ok":
            saldo_totale = int(parts[1]) if parts[1].isdigit() else 0
            free_bids    = int(parts[2]) if parts[2].isdigit() else 0
            spent_bids   = (int(parts[3]) if parts[3].isdigit() else 0) + (int(parts[4]) if parts[4].isdigit() else 0)
            self.log(f"[DATA] ✅ Bid OK — saldo bids: {saldo_totale} | gratis: {free_bids} | spese: {spent_bids}")
            self.val_pagate.setText(str(saldo_totale))
            self.val_gratis.setText(str(free_bids))
            cur = self.val_manuali.text()
            self.val_manuali.setText(str((int(cur) if cur.isdigit() else 0) + 1))

        elif status == "no":
            err_type    = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0
            err_msg     = parts[1] if len(parts) > 1 else "motivo sconosciuto"
            self.log(f"[DATA] ❌ Bid rifiutato — {err_msg} (tipo: {err_type})")

        else:
            self.log(f"[DATA] ⚠️ Status sconosciuto: {status} | {result[:80]}")

    def _on_slider_change(self, val):
        secs = val * 0.5
        self.lbl_fire_val.setText(f"{secs:g}s")

    @property
    def fire_offset_ms(self):
        return self.fire_slider.value() * 500

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
            f"AREA '{self.winning_label}'  |  Trascina → ENTER conferma  |  ESC salta"
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
        if not self.rect_timer and not self.radio_data.isChecked():
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
        mode = "data" if self.radio_data.isChecked() else "pixel"
        self.log(f"Avvio → ID: {id_asta} | Modalità: {mode.upper()}")

        if mode == "data":
            self._active_id = id_asta
            self._already_sent = False
            self._fire_handle = None
            self._last_scadenza = 0
            # Reset bid labels for new auction
            self.val_pagate.setText("0")
            self.val_gratis.setText("0")
            self.val_manuali.setText("0")
            self.val_auto.setText("0")
            self.val_timer.setText("—")
            self.val_stato.setText("—")
            self.val_prezzo.setText("—")
            self.val_vinc.setText("—")
            self.log(f"[DATA] Monitoraggio via data.php attivo — fire a 500ms")
        else:
            self.log(f"Soglia pixel: {soglia_min}–{soglia_max}")
            self.worker = MonitorWorker(
                self.browser, id_asta, self.dess,
                soglia_min, soglia_max,
                self.rect_timer, self.rect_vincendo,
                mode=mode
            )
            self.worker.log_signal.connect(self.log)
            self.worker.status_signal.connect(self.update_status)
            self.worker.done_signal.connect(self.on_worker_done)
            self.worker.grab_signal.connect(self.do_grab)
            self.worker.auction_signal.connect(self.update_auction_info)
            self.worker_thread = threading.Thread(target=self.worker.start_loop, daemon=True)
            self.worker_thread.start()

    def on_stop(self):
        self._active_id = None
        self._already_sent = False
        if self._fire_handle:
            self._fire_handle.stop()
            self._fire_handle = None
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

    def update_status(self, val, in_range, vincendo):
        r = "✓ IN RANGE" if in_range else "○ fuori"
        v = "  |  🏆 VINCENDO" if vincendo else ""
        if self.worker and self.worker.mode == "data":
            self.status_bar.setText(f"Timer: {val}ms  |  {r}{v}")
        else:
            self.status_bar.setText(f"Red px: {val}  |  {r}{v}")

    def update_auction_info(self, d):
        import time as _t
        elapsed   = _t.time() - d.get("receipt_time", _t.time())
        timer_sec = d.get("diff_int", 0) - elapsed
        if timer_sec > 0:
            mins  = int(timer_sec // 60)
            secs  = timer_sec % 60
            t_str = f"{mins:02d}:{secs:06.3f}"
        else:
            t_str = "SCADUTO"
        stato = d.get("stato", "?")
        color = "#27ae60" if stato == "ON" else "#e74c3c"
        self.val_stato.setText(stato)
        self.val_stato.setStyleSheet(f"color:{color}; font-size:11px; font-family:'Courier New'; font-weight:bold;")
        self.val_timer.setText(t_str)
        self.val_prezzo.setText(f"€ {d.get('prezzo', 0):.2f}")
        self.val_vinc.setText(d.get("vincitore", "—") or "—")
        # Puntate/saldo NON aggiornati qui — solo da bid.php response


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
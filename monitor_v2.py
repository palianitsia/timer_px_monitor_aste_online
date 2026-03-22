#/== dev by palianitsia ==\

#|ogni|utente|e|consapevole|di|eventuale|violazione|di|tos|durante|uso|del|software|si|prega|di|usare|software|solo|in|ambiente|di|test|

import tkinter as tk
from PIL import Image, ImageTk
import threading
import asyncio
import aiohttp
import mss
import numpy as np
import cv2

RED_LOWER = np.array([0, 150, 150])
RED_UPPER = np.array([10, 255, 255])

# Ciano "STAI VINCENDO": #00bcd4 → HSV ~H:88, S:255, V:212
CYAN_LOWER = np.array([85, 180, 150])
CYAN_UPPER = np.array([95, 255, 255])

DOMAIN = "it"
USER_AGENT = "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
DESS = ""

def build_url(id_asta):
    return f"https://{DOMAIN}.bidoo.com/bid.php?AID={id_asta}&sup=0&shock=0"

def get_headers():
    return {
        "Cookie": f"dess={DESS};",
        "User-Agent": USER_AGENT,
        "Referer": f"https://{DOMAIN}.bidoo.com/",
        "Accept": "*/*",
    }

state = {
    "running": False,
    "url": "",
    "area_timer": None,
    "area_vincendo": None,
    "thread": None,
    "loop": None,
}


def select_area_tk(label_text):
    result = {}
    with mss.mss() as sct:
        full = np.array(sct.grab(sct.monitors[1]))
    full_rgb = cv2.cvtColor(full, cv2.COLOR_BGRA2RGB)
    sh, sw = full_rgb.shape[:2]

    sel_win = tk.Toplevel()
    sel_win.attributes("-fullscreen", True)
    sel_win.attributes("-topmost", True)
    sel_win.configure(cursor="crosshair")

    pil_img = Image.fromarray(full_rgb)
    tk_img = ImageTk.PhotoImage(pil_img)

    canvas = tk.Canvas(sel_win, width=sw, height=sh, highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.create_text(sw // 2, 30, text=label_text,
                       fill="yellow", font=("Courier New", 14, "bold"))

    drag = {"x0": 0, "y0": 0, "x1": 0, "y1": 0, "rect": None, "dragging": False}

    def on_press(e):
        drag["x0"], drag["y0"] = e.x, e.y
        drag["dragging"] = True
        if drag["rect"]:
            canvas.delete(drag["rect"])

    def on_drag(e):
        if drag["dragging"]:
            if drag["rect"]:
                canvas.delete(drag["rect"])
            drag["x1"], drag["y1"] = e.x, e.y
            drag["rect"] = canvas.create_rectangle(
                drag["x0"], drag["y0"], e.x, e.y,
                outline="#e74c3c", width=2
            )

    def on_release(e):
        drag["x1"], drag["y1"] = e.x, e.y
        drag["dragging"] = False

    def on_confirm(e=None):
        x0, y0 = min(drag["x0"], drag["x1"]), min(drag["y0"], drag["y1"])
        x1, y1 = max(drag["x0"], drag["x1"]), max(drag["y0"], drag["y1"])
        if x1 - x0 > 5 and y1 - y0 > 5:
            result["area"] = {"top": y0, "left": x0, "width": x1 - x0, "height": y1 - y0}
        sel_win.destroy()

    def on_cancel(e=None):
        sel_win.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    sel_win.bind("<Return>", on_confirm)
    sel_win.bind("<Escape>", on_cancel)

    sel_win.wait_window()
    return result.get("area", None)


def select_area(label_text):
    area = [None]
    done = threading.Event()

    def run():
        area[0] = select_area_tk(label_text)
        done.set()

    root.after(0, run)
    done.wait()
    return area[0]


def capture(area, sct):
    frame = np.array(sct.grab(area))
    return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)


def count_red_pixels(bgr_frame):
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LOWER, RED_UPPER)
    return cv2.countNonZero(mask)


def is_vincendo(bgr_frame):
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, CYAN_LOWER, CYAN_UPPER)
    cyan_px = cv2.countNonZero(mask)
    return cyan_px > 50


async def send_fetch(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=get_headers(),
                timeout=aiohttp.ClientTimeout(total=2)
            ) as resp:
                body = await resp.text()
                log(f"Fetch inviato → status {resp.status} | {body[:80]}")
    except Exception as e:
        log(f"Errore fetch: {e}")


async def monitor_loop(area_timer, area_vincendo, soglia_min, soglia_max, url):
    already_sent = False
    with mss.mss() as sct:
        while state["running"]:
            frame_timer = capture(area_timer, sct)
            red_count = count_red_pixels(frame_timer)

            vincendo = False
            if area_vincendo:
                frame_v = capture(area_vincendo, sct)
                vincendo = is_vincendo(frame_v)

            log_status(red_count, soglia_min, soglia_max, vincendo)

            in_range = soglia_min <= red_count <= soglia_max

            if in_range and not vincendo:
                if not already_sent:
                    log(f"Timer in range ({red_count} px) → fetch!")
                    await send_fetch(url)
                    already_sent = True
            elif in_range and vincendo:
                if not already_sent:
                    log(f"Timer in range ma stai già vincendo — skip fetch.")
                    already_sent = True
            else:
                already_sent = False

            await asyncio.sleep(0.02)


def run_async_loop(area_timer, area_vincendo, soglia_min, soglia_max, url):
    loop = asyncio.new_event_loop()
    state["loop"] = loop
    asyncio.set_event_loop(loop)
    loop.run_until_complete(monitor_loop(area_timer, area_vincendo, soglia_min, soglia_max, url))
    loop.close()


def on_save():
    global DESS
    d = dess_var.get().strip()
    if d:
        DESS = d
        log(f"DESS salvato: {DESS[:12]}...")
    else:
        log("DESS vuoto!")


def on_go():
    if state["running"]:
        return
    id_asta = id_var.get().strip()
    if not id_asta:
        log("Inserisci un ID asta valido!")
        return
    if not DESS:
        log("Inserisci e salva il cookie DESS!")
        return
    try:
        soglia_min = int(soglia_min_var.get())
        soglia_max = int(soglia_max_var.get())
    except ValueError:
        log("Soglia min/max devono essere numeri interi!")
        return
    if soglia_min >= soglia_max:
        log("Soglia min deve essere minore di soglia max!")
        return

    url = build_url(id_asta)
    state["url"] = url
    state["running"] = True
    btn_go.config(state="disabled")
    btn_stop.config(state="normal")
    log(f"ID asta: {id_asta} | URL: {url}")
    log(f"Soglia px rossi: {soglia_min} — {soglia_max}")

    def start():
        log("1/2 — Seleziona area TIMER...")
        area_timer = select_area("AREA TIMER  |  Trascina sul timer  |  ENTER = conferma  |  ESC = annulla")
        if not area_timer:
            log("Selezione timer annullata.")
            state["running"] = False
            root.after(0, lambda: btn_go.config(state="normal"))
            root.after(0, lambda: btn_stop.config(state="disabled"))
            return
        state["area_timer"] = area_timer
        log(f"Area timer: {area_timer}")

        log("2/2 — Seleziona area STAI VINCENDO...")
        area_vincendo = select_area("AREA 'STAI VINCENDO'  |  Trascina sul bottone  |  ENTER = conferma  |  ESC = salta")
        state["area_vincendo"] = area_vincendo
        if area_vincendo:
            log(f"Area vincendo: {area_vincendo}")
        else:
            log("Area vincendo non selezionata — controllo disabilitato.")

        log("Monitoraggio avviato...")
        run_async_loop(area_timer, area_vincendo, soglia_min, soglia_max, url)

    t = threading.Thread(target=start, daemon=True)
    state["thread"] = t
    t.start()


def on_stop():
    state["running"] = False
    btn_go.config(state="normal")
    btn_stop.config(state="disabled")
    log("Monitoraggio fermato.")


def log(msg):
    log_text.config(state="normal")
    log_text.insert("end", msg + "\n")
    log_text.see("end")
    log_text.config(state="disabled")


def log_status(red, soglia_min, soglia_max, vincendo):
    in_range = soglia_min <= red <= soglia_max
    v_str = "  |  🏆 VINCENDO" if vincendo else ""
    r_str = "✓ IN RANGE" if in_range else "○ fuori"
    status_var.set(f"Red px: {red}  |  {soglia_min}–{soglia_max}  |  {r_str}{v_str}")


root = tk.Tk()
root.title("Timer Monitor")
root.resizable(False, False)
root.attributes("-topmost", True)

BG = "#0d1b2a"
ACCENT = "#e74c3c"
FG = "#eaeaea"
BTN_BG = "#1a2d42"
FONT = ("Courier New", 10)

root.configure(bg=BG)

# === Riga 1: ID Asta + DESS ===
frame_row1 = tk.Frame(root, bg=BG, padx=10, pady=8)
frame_row1.pack(fill="x")

tk.Label(frame_row1, text="ID Asta:", bg=BG, fg=FG, font=FONT).pack(side="left")
id_var = tk.StringVar()
tk.Entry(frame_row1, textvariable=id_var, width=12, bg=BTN_BG, fg=FG,
         insertbackground=FG, relief="flat", font=FONT).pack(side="left", padx=(4, 10))

tk.Label(frame_row1, text="DESS:", bg=BG, fg=FG, font=FONT).pack(side="left")
dess_var = tk.StringVar()
tk.Entry(frame_row1, textvariable=dess_var, width=22, bg=BTN_BG, fg=FG,
         insertbackground=FG, relief="flat", font=FONT, show="*").pack(side="left", padx=(4, 4))
tk.Button(frame_row1, text="✓", command=on_save,
          bg=ACCENT, fg="white", relief="flat", font=FONT, padx=6).pack(side="left")

# === Riga 2: Soglia min / max ===
frame_row2 = tk.Frame(root, bg=BG, padx=10, pady=0)
frame_row2.pack(fill="x")

tk.Label(frame_row2, text="Soglia min px:", bg=BG, fg=FG, font=FONT).pack(side="left")
soglia_min_var = tk.StringVar(value="660")
tk.Entry(frame_row2, textvariable=soglia_min_var, width=6, bg=BTN_BG, fg=FG,
         insertbackground=FG, relief="flat", font=FONT).pack(side="left", padx=(4, 14))

tk.Label(frame_row2, text="Soglia max px:", bg=BG, fg=FG, font=FONT).pack(side="left")
soglia_max_var = tk.StringVar(value="770")
tk.Entry(frame_row2, textvariable=soglia_max_var, width=6, bg=BTN_BG, fg=FG,
         insertbackground=FG, relief="flat", font=FONT).pack(side="left", padx=(4, 0))

# === Riga 3: GO / STOP ===
frame_btns = tk.Frame(root, bg=BG, padx=10, pady=8)
frame_btns.pack(fill="x")

btn_go = tk.Button(frame_btns, text="▶  GO", command=on_go, width=12,
                   bg="#27ae60", fg="white", relief="flat", font=("Courier New", 11, "bold"), pady=4)
btn_go.pack(side="left", padx=(0, 8))

btn_stop = tk.Button(frame_btns, text="■  STOP", command=on_stop, width=12,
                     bg=ACCENT, fg="white", relief="flat", font=("Courier New", 11, "bold"),
                     pady=4, state="disabled")
btn_stop.pack(side="left")

# === Status bar ===
status_var = tk.StringVar(value="In attesa...")
tk.Label(root, textvariable=status_var, bg="#0a1520", fg="#7fb3d3",
         font=("Courier New", 9), anchor="w", padx=10, pady=3).pack(fill="x")

# === Log ===
log_text = tk.Text(root, height=8, bg="#060f18", fg="#7fb3d3", font=("Courier New", 9),
                   relief="flat", state="disabled", padx=6, pady=4)
log_text.pack(fill="both", padx=10, pady=(4, 10))

root.mainloop()

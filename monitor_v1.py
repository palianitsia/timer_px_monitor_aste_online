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

DOMAIN = "it"
USER_AGENT = "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
DESS = "incolla qui il tuo dess"

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
    "area": None,
    "threshold": None,
    "thread": None,
    "loop": None,
}


def select_area_tk():
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
    canvas.create_text(sw // 2, 30, text="Trascina per selezionare area timer  |  ENTER = conferma  |  ESC = annulla",
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


def select_area():
    area = [None]
    done = threading.Event()

    def run():
        area[0] = select_area_tk()
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


def calibrate_tk(area):
    result = {}
    done = threading.Event()

    def show():
        with mss.mss() as sct:
            frame = capture(area, sct)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        scale = min(400 / w, 300 / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        frame_rgb = cv2.resize(frame_rgb, (nw, nh))

        cal_win = tk.Toplevel()
        cal_win.title("Calibrazione")
        cal_win.attributes("-topmost", True)
        cal_win.resizable(False, False)

        pil_img = Image.fromarray(frame_rgb)
        tk_img = ImageTk.PhotoImage(pil_img)
        lbl_img = tk.Label(cal_win, image=tk_img)
        lbl_img.image = tk_img
        lbl_img.pack()

        red = count_red_pixels(capture(area, mss.mss()))
        lbl_info = tk.Label(cal_win, text=f"Red px rilevati: {red}\nPremi CONFERMA quando il timer mostra 2",
                            font=("Courier New", 10), pady=8)
        lbl_info.pack()

        def confirm():
            with mss.mss() as sct:
                f = capture(area, sct)
            r = count_red_pixels(f)
            result["threshold"] = int(r * 0.80)
            cal_win.destroy()
            done.set()

        tk.Button(cal_win, text="✓ CONFERMA (timer mostra 2)", command=confirm,
                  bg="#27ae60", fg="white", font=("Courier New", 10, "bold"),
                  relief="flat", pady=6, padx=10).pack(pady=(0, 10))

        cal_win.wait_window()

    root.after(0, show)
    done.wait()
    return result.get("threshold", 50)


async def send_fetch(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=get_headers(),
                data=DESS,
                timeout=aiohttp.ClientTimeout(total=2)
            ) as resp:
                body = await resp.text()
                log(f"Fetch inviato → status {resp.status} | {body[:80]}")
    except Exception as e:
        log(f"Errore fetch: {e}")


async def monitor_loop(area, threshold, url):
    already_sent = False
    with mss.mss() as sct:
        while state["running"]:
            frame = capture(area, sct)
            red_count = count_red_pixels(frame)
            log_status(red_count, threshold)

            if red_count <= threshold:
                if not already_sent:
                    log(f"Timer a 1! ({red_count} px) → fetch!")
                    await send_fetch(url)
                    already_sent = True
            else:
                already_sent = False

            await asyncio.sleep(0.02)


def run_async_loop(area, threshold, url):
    loop = asyncio.new_event_loop()
    state["loop"] = loop
    asyncio.set_event_loop(loop)
    loop.run_until_complete(monitor_loop(area, threshold, url))
    loop.close()


def on_go():
    if state["running"]:
        return
    id_asta = id_var.get().strip()
    if not id_asta:
        log("Inserisci un ID asta valido!")
        return
    url = build_url(id_asta)
    state["url"] = url
    state["running"] = True
    btn_go.config(state="disabled")
    btn_stop.config(state="normal")
    log(f"ID asta: {id_asta}")
    log(f"URL: {url}")
    log("Selezione area — trascina sul timer...")

    def start():
        area = select_area()
        if not area:
            log("Selezione annullata.")
            state["running"] = False
            root.after(0, lambda: btn_go.config(state="normal"))
            root.after(0, lambda: btn_stop.config(state="disabled"))
            return
        state["area"] = area
        log(f"Area: {area}")
        log("Calibrazione — posiziona il timer su 2 e premi CONFERMA...")
        threshold = calibrate_tk(area)
        state["threshold"] = threshold
        log(f"Soglia: {threshold} px rossi")
        log("Monitoraggio avviato...")
        run_async_loop(area, threshold, url)

    t = threading.Thread(target=start, daemon=True)
    state["thread"] = t
    t.start()


def on_stop():
    state["running"] = False
    btn_go.config(state="normal")
    btn_stop.config(state="disabled")
    log("Monitoraggio fermato.")


def on_confirm_id():
    id_asta = id_var.get().strip()
    if id_asta:
        url = build_url(id_asta)
        state["url"] = url
        log(f"ID confermato: {id_asta}")
        log(f"URL: {url}")
    else:
        log("ID asta vuoto!")


def log(msg):
    log_text.config(state="normal")
    log_text.insert("end", msg + "\n")
    log_text.see("end")
    log_text.config(state="disabled")


def log_status(red, threshold):
    status_var.set(f"Red px: {red}  |  Soglia: {threshold}")


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

frame_id = tk.Frame(root, bg=BG, padx=10, pady=8)
frame_id.pack(fill="x")

tk.Label(frame_id, text="ID Asta:", bg=BG, fg=FG, font=FONT).pack(side="left")
id_var = tk.StringVar()
id_entry = tk.Entry(frame_id, textvariable=id_var, width=20, bg=BTN_BG, fg=FG,
                    insertbackground=FG, relief="flat", font=FONT)
id_entry.pack(side="left", padx=(6, 4))
btn_confirm = tk.Button(frame_id, text="✓", command=on_confirm_id,
                        bg=ACCENT, fg="white", relief="flat", font=FONT, padx=6)
btn_confirm.pack(side="left")

frame_btns = tk.Frame(root, bg=BG, padx=10, pady=4)
frame_btns.pack(fill="x")

btn_go = tk.Button(frame_btns, text="▶  GO", command=on_go, width=12,
                   bg="#27ae60", fg="white", relief="flat", font=("Courier New", 11, "bold"), pady=4)
btn_go.pack(side="left", padx=(0, 8))

btn_stop = tk.Button(frame_btns, text="■  STOP", command=on_stop, width=12,
                     bg=ACCENT, fg="white", relief="flat", font=("Courier New", 11, "bold"),
                     pady=4, state="disabled")
btn_stop.pack(side="left")

status_var = tk.StringVar(value="In attesa...")
status_bar = tk.Label(root, textvariable=status_var, bg="#0a1520", fg="#7fb3d3",
                      font=("Courier New", 9), anchor="w", padx=10, pady=3)
status_bar.pack(fill="x")

log_text = tk.Text(root, height=8, bg="#060f18", fg="#7fb3d3", font=("Courier New", 9),
                   relief="flat", state="disabled", padx=6, pady=4)
log_text.pack(fill="both", padx=10, pady=(4, 10))

root.mainloop()
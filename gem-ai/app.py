import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import serial
import serial.tools.list_ports

# Optional libs (only needed for background image)
from PIL import Image, ImageTk


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DIR = os.path.join(BASE_DIR, "raw")
COMP_DIR = os.path.join(BASE_DIR, "composites")

ANGLES = [0, 90, 180, 270]


def ensure_dirs():
    for lab in ["real", "synthetic"]:
        os.makedirs(os.path.join(RAW_DIR, lab), exist_ok=True)
        os.makedirs(os.path.join(COMP_DIR, lab), exist_ok=True)


def make_composite_for_stone(label, stone_id, tile=256):
    """Create ONE composite for raw/<label>/<stone_id>"""
    from PIL import Image

    stone_dir = os.path.join(RAW_DIR, label, stone_id)
    needed = [os.path.join(stone_dir, f"{a}.jpg") for a in ANGLES]
    for p in needed:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    imgs = [Image.open(p).convert("RGB").resize((tile, tile)) for p in needed]

    grid = Image.new("RGB", (tile * 2, tile * 2))
    grid.paste(imgs[0], (0, 0))          # 0
    grid.paste(imgs[1], (tile, 0))       # 90
    grid.paste(imgs[2], (0, tile))       # 180
    grid.paste(imgs[3], (tile, tile))    # 270

    out_path = os.path.join(COMP_DIR, label, f"{stone_id}.jpg")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    grid.save(out_path, quality=95)
    return out_path


class App:
    def __init__(self, root):
        ensure_dirs()

        self.root = root
        self.root.title("Gem Polarizer Controller")
        self.root.state("zoomed")

        self.ser = None
        self.current_angle = 0

        # ---- Layout root ----
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Canvas for background (optional)
        self.canvas = tk.Canvas(self.root, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.configure(bg="#0b1220")

        self.bg_img = None
        self.bg_photo = None
        self.bg_id = None

        # set background image path (optional)
        self.bg_path = os.path.join(BASE_DIR, "assets", "bg.jpg")

        # Content frame
        self.content = tk.Frame(self.canvas, bg="white")
        self.content_id = self.canvas.create_window(0, 0, anchor="center", window=self.content)

        self.canvas.bind("<Configure>", self.on_resize)

        # ---- Styles ----
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("TLabel", background="white", foreground="black")
        style.configure("Heading.TLabel", font=("Segoe UI", 26, "bold"), background="white", foreground="black")
        style.configure("Status.TLabel", font=("Segoe UI", 11), background="white", foreground="black")
        style.configure("Angle.TLabel", font=("Segoe UI", 16, "bold"), background="white", foreground="black")

        style.configure("Primary.TButton", font=("Segoe UI", 12, "bold"), padding=12)
        style.map("Primary.TButton",
                  background=[("active", "#2563eb"), ("!active", "#1d4ed8")],
                  foreground=[("!disabled", "white")])

        style.configure("Danger.TButton", font=("Segoe UI", 12, "bold"), padding=12)
        style.map("Danger.TButton",
                  background=[("active", "#dc2626"), ("!active", "#b91c1c")],
                  foreground=[("!disabled", "white")])

        style.configure("Ghost.TButton", font=("Segoe UI", 11, "bold"), padding=10)
        style.map("Ghost.TButton",
                  background=[("active", "#e5e7eb"), ("!active", "#f3f4f6")],
                  foreground=[("!disabled", "black")])

        # ---- Build UI ----
        self.build_ui()

        # poll serial
        self.root.after(80, self.read_serial)

    def on_resize(self, event):
        w, h = event.width, event.height
        self.canvas.coords(self.content_id, w // 2, h // 2)

        # Background image cover (optional)
        try:
            if os.path.exists(self.bg_path):
                if self.bg_img is None:
                    self.bg_img = Image.open(self.bg_path)

                iw, ih = self.bg_img.size
                scale = max(w / iw, h / ih)
                nw, nh = int(iw * scale), int(ih * scale)
                resized = self.bg_img.resize((nw, nh), Image.LANCZOS)

                left = (nw - w) // 2
                top = (nh - h) // 2
                cropped = resized.crop((left, top, left + w, top + h))

                self.bg_photo = ImageTk.PhotoImage(cropped)
                if self.bg_id is None:
                    self.bg_id = self.canvas.create_image(0, 0, anchor="nw", image=self.bg_photo)
                    self.canvas.tag_lower(self.bg_id)
                else:
                    self.canvas.itemconfig(self.bg_id, image=self.bg_photo)
            else:
                self.canvas.configure(bg="#0b1220")
        except:
            self.canvas.configure(bg="#0b1220")

    def build_ui(self):
        card = tk.Frame(self.content, bg="white", highlightthickness=1, highlightbackground="#d1d5db")
        card.grid(row=0, column=0, padx=40, pady=40)
        card.columnconfigure(0, weight=1)

        # Heading
        ttk.Label(card, text="Gem Polarizer Controller", style="Heading.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))
        ttk.Label(card, text="Manual capture workflow + auto file saving (0/90/180/270).", style="Status.TLabel").grid(row=1, column=0, sticky="w", pady=(0, 18))

        # Top: COM + Connect
        top = tk.Frame(card, bg="white")
        top.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="COM Port:", style="TLabel").grid(row=0, column=0, sticky="w")

        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_var = tk.StringVar(value=ports[0] if ports else "")
        self.port_combo = ttk.Combobox(top, values=ports, textvariable=self.port_var, width=18, state="readonly")
        self.port_combo.grid(row=0, column=1, sticky="w", padx=10)

        ttk.Button(top, text="Refresh", style="Ghost.TButton", command=self.refresh_ports).grid(row=0, column=2, padx=6)
        ttk.Button(top, text="Connect", style="Primary.TButton", command=self.connect).grid(row=0, column=3, padx=6)
        ttk.Button(top, text="Disconnect", style="Ghost.TButton", command=self.disconnect).grid(row=0, column=4, padx=6)

        # Mid: angle + status
        mid = tk.Frame(card, bg="white")
        mid.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)

        self.angle_lbl = ttk.Label(mid, text="Angle: 0°", style="Angle.TLabel")
        self.angle_lbl.grid(row=0, column=0, sticky="w")

        self.status_lbl = ttk.Label(mid, text="Status: Not connected", style="Status.TLabel")
        self.status_lbl.grid(row=0, column=1, sticky="e")

        # Control buttons
        btns = tk.Frame(card, bg="white")
        btns.grid(row=4, column=0, sticky="ew", pady=(0, 16))
        for c in range(4):
            btns.columnconfigure(c, weight=1)

        ttk.Button(btns, text="LED ON", style="Primary.TButton", command=lambda: self.send("LED_ON")).grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        ttk.Button(btns, text="LED OFF", style="Ghost.TButton", command=lambda: self.send("LED_OFF")).grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        ttk.Button(btns, text="START", style="Primary.TButton", command=lambda: self.send("START")).grid(row=0, column=2, padx=6, pady=6, sticky="ew")
        ttk.Button(btns, text="STOP", style="Danger.TButton", command=lambda: self.send("STOP")).grid(row=0, column=3, padx=6, pady=6, sticky="ew")

        # Dataset section
        ds = tk.LabelFrame(card, text="Dataset (Save your photos correctly)", bg="white", fg="black", padx=10, pady=10)
        ds.grid(row=5, column=0, sticky="ew")
        ds.columnconfigure(1, weight=1)

        ttk.Label(ds, text="Stone ID:", style="TLabel").grid(row=0, column=0, sticky="w", pady=4)
        self.stone_var = tk.StringVar(value="stone001")
        tk.Entry(ds, textvariable=self.stone_var).grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Label(ds, text="Label:", style="TLabel").grid(row=1, column=0, sticky="w", pady=4)
        self.label_var = tk.StringVar(value="real")
        ttk.Combobox(ds, values=["real", "synthetic"], textvariable=self.label_var, state="readonly", width=14).grid(row=1, column=1, sticky="w", padx=8)

        self.save_btn = ttk.Button(ds, text="Import Photo for Current Angle", style="Primary.TButton", command=self.import_current_angle_photo)
        self.save_btn.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 6))

        self.open_raw_btn = ttk.Button(ds, text="Open Raw Folder", style="Ghost.TButton", command=self.open_raw_folder)
        self.open_raw_btn.grid(row=3, column=0, sticky="ew", pady=4)

        self.make_comp_btn = ttk.Button(ds, text="Make Composite (This Stone)", style="Ghost.TButton", command=self.make_composite_one)
        self.make_comp_btn.grid(row=3, column=1, sticky="ew", pady=4)

        self.comp_status = ttk.Label(ds, text="Composite: Not created", style="Status.TLabel")
        self.comp_status.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(card, text="Tip: Rotate to each angle → capture photo on phone → Import photo here. After 4 angles, click Make Composite.",
                  style="Status.TLabel").grid(row=6, column=0, sticky="w", pady=(12, 0))

    # ---------- dataset actions ----------
    def stone_dir(self):
        stone = self.stone_var.get().strip()
        if not stone:
            raise ValueError("Stone ID is empty")
        label = self.label_var.get().strip()
        return label, stone, os.path.join(RAW_DIR, label, stone)

    def import_current_angle_photo(self):
        try:
            label, stone, folder = self.stone_dir()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        os.makedirs(folder, exist_ok=True)
        angle = self.current_angle

        # choose file
        path = filedialog.askopenfilename(
            title=f"Select photo for angle {angle}°",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.JPG;*.JPEG;*.PNG")]
        )
        if not path:
            return

        dest = os.path.join(folder, f"{angle}.jpg")
        try:
            shutil.copy2(path, dest)
            self.status_lbl.config(text=f"Saved: raw/{label}/{stone}/{angle}.jpg")
        except Exception as e:
            messagebox.showerror("Save failed", str(e))

    def make_composite_one(self):
        try:
            label, stone, _ = self.stone_dir()
            out = make_composite_for_stone(label, stone)
            self.comp_status.config(text=f"Composite created: composites/{label}/{stone}.jpg")
            self.status_lbl.config(text="Composite created successfully.")
        except Exception as e:
            messagebox.showerror("Composite error", str(e))

    def open_raw_folder(self):
        try:
            label, stone, folder = self.stone_dir()
            os.makedirs(folder, exist_ok=True)
            os.startfile(folder)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------- serial ----------
    def refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        if ports:
            self.port_var.set(ports[0])

    def connect(self):
        port = self.port_var.get().strip()
        if not port:
            messagebox.showerror("Error", "No COM port selected.")
            return
        try:
            self.ser = serial.Serial(port, 115200, timeout=0.1)
            self.ser.reset_input_buffer()
            self.status_lbl.config(text=f"Status: Connected ({port})")
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.ser = None

    def disconnect(self):
        if self.ser:
            try:
                self.ser.close()
            except:
                pass
        self.ser = None
        self.status_lbl.config(text="Status: Not connected")

    def send(self, cmd):
        if not self.ser:
            messagebox.showwarning("Warning", "Please Connect first!")
            return
        self.ser.write((cmd + "\n").encode())

    def read_serial(self):
        if self.ser:
            try:
                while self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors="ignore").strip()
                    if not line:
                        continue

                    # Example: ANGLE=90
                    if line.startswith("ANGLE="):
                        ang = int(line.split("=", 1)[1])
                        self.current_angle = ang
                        self.angle_lbl.config(text=f"Angle: {ang}°")
                        self.status_lbl.config(text=f"Rotate stopped at {ang}°. Capture photo → Import.")
                    else:
                        self.status_lbl.config(text=f"{line}")
            except:
                pass

        self.root.after(80, self.read_serial)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()

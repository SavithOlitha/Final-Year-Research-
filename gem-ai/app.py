import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import serial
import serial.tools.list_ports
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
    stone_dir = os.path.join(RAW_DIR, label, stone_id)
    needed = [os.path.join(stone_dir, f"{a}.jpg") for a in ANGLES]
    for p in needed:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing: {p}")

    imgs = [Image.open(p).convert("RGB").resize((tile, tile)) for p in needed]

    grid = Image.new("RGB", (tile * 2, tile * 2))
    grid.paste(imgs[0], (0, 0))
    grid.paste(imgs[1], (tile, 0))
    grid.paste(imgs[2], (0, tile))
    grid.paste(imgs[3], (tile, tile))

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

        # Layout root
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Main container (full window)
        self.content = tk.Frame(self.root, bg="white")
        self.content.grid(row=0, column=0, sticky="nsew")
        self.root.configure(bg="white")
        self.root.configure(bg="#f5f7fb")   # light background
        self.content = tk.Frame(self.root, bg="#f5f7fb")
        self.content.grid(row=0, column=0, sticky="nsew")



        # Styles
        self.setup_styles()

        # Build UI
        self.build_ui()

        # Start serial polling
        self.root.after(80, self.read_serial)

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

    # Base colors
        self.APP_BG = "#f3f4f6"   # light gray
        self.CARD_BG = "#ffffff"
        self.BORDER = "#e5e7eb"
        self.TEXT = "#111827"
        self.MUTED = "#6b7280"

    # Labels
        style.configure("TLabel", background=self.CARD_BG, foreground=self.TEXT)
        style.configure("Heading.TLabel", font=("Segoe UI", 28, "bold"),
                    background=self.CARD_BG, foreground=self.TEXT)
        style.configure("Sub.TLabel", font=("Segoe UI", 11),
                    background=self.CARD_BG, foreground=self.MUTED)
        style.configure("Section.TLabel", font=("Segoe UI", 14, "bold"),
                    background=self.CARD_BG, foreground=self.TEXT)
        style.configure("Status.TLabel", font=("Segoe UI", 10),
                    background=self.CARD_BG, foreground=self.MUTED)
        style.configure("AngleBig.TLabel", font=("Segoe UI", 26, "bold"),
                    background=self.CARD_BG, foreground=self.TEXT)

    # Buttons
        style.configure("Primary.TButton", font=("Segoe UI", 12, "bold"), padding=(18, 14))
        style.map("Primary.TButton",
              background=[("active", "#2563eb"), ("!active", "#1d4ed8")],
              foreground=[("!disabled", "white")])

        style.configure("Danger.TButton", font=("Segoe UI", 12, "bold"), padding=(18, 14))
        style.map("Danger.TButton",
              background=[("active", "#dc2626"), ("!active", "#b91c1c")],
              foreground=[("!disabled", "white")])

        style.configure("Ghost.TButton", font=("Segoe UI", 11, "bold"), padding=(16, 12))
        style.map("Ghost.TButton",
              background=[("active", "#e5e7eb"), ("!active", "#f9fafb")],
              foreground=[("!disabled", self.TEXT)])

    # Tabs (page switch buttons)
        style.configure("Tab.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 10))
        style.map("Tab.TButton",
              background=[("active", "#e5e7eb"), ("!active", "#f9fafb")],
              foreground=[("!disabled", self.TEXT)])

        style.configure("TabActive.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 10))
        style.map("TabActive.TButton",
              background=[("active", "#dbeafe"), ("!active", "#bfdbfe")],
              foreground=[("!disabled", "#1e3a8a")])


    def on_resize(self, event):
        w, h = event.width, event.height
        self.canvas.coords(self.content_id, w // 2, h // 2)

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

    # ---------------- UI ----------------
    def build_ui(self):
    # --- Root / Content stretch ---
        self.root.configure(bg=self.APP_BG)
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.content.configure(bg=self.APP_BG)
        self.content.grid(row=0, column=0, sticky="nsew")
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

    # --- Main Card ---
        self.card = tk.Frame(self.content, bg=self.CARD_BG, bd=1, relief="solid")
        self.card.grid(row=0, column=0, padx=28, pady=24, sticky="nsew")
        self.card.columnconfigure(0, weight=1)

    # rows: 0=header, 1=controls, 2=separator, 3=pages (grow)
        self.card.rowconfigure(3, weight=1)
    # ---------- Header (Title + Tabs) ----------
        header = tk.Frame(self.card, bg=self.CARD_BG)
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(18, 6))

        header.columnconfigure(0, weight=1)
        header.columnconfigure(1, weight=2)
        header.columnconfigure(2, weight=1)

        ttk.Label(header, text="Device Controller", style="Heading.TLabel").grid(
        row=0, column=1, sticky="n"
    )

        nav = tk.Frame(header, bg=self.CARD_BG)
        nav.grid(row=0, column=2, sticky="e")

        self.btn_device_tab = ttk.Button(
        nav, text="Device Control Page", style="TabActive.TButton",
        command=lambda: self.show_page("device")
    )
        self.btn_device_tab.grid(row=0, column=0, padx=6)

        self.btn_dataset_tab = ttk.Button(
        nav, text="Dataset & Model Page", style="Tab.TButton",
        command=lambda: self.show_page("dataset")
    )
        self.btn_dataset_tab.grid(row=0, column=1, padx=6)

    # ---------- Controls Area ----------
        controls_wrap = tk.Frame(self.card, bg=self.CARD_BG)
        controls_wrap.grid(row=1, column=0, sticky="ew", padx=20, pady=(18, 10))
        controls_wrap.columnconfigure(0, weight=1)

        self.conn_bar = self.build_connection_bar(controls_wrap)
        self.conn_bar.grid(row=0, column=0, sticky="ew")

        self.status_bar = self.build_status_bar(controls_wrap)
        self.status_bar.grid(row=1, column=0, sticky="ew", pady=(8, 0))
    # ---------- Separator ----------
        ttk.Separator(self.card, orient="horizontal").grid(
        row=2, column=0, sticky="ew", padx=20, pady=(6, 14)
    )

    # ---------- Pages (this should grow) ----------
        self.pages = tk.Frame(self.card, bg=self.CARD_BG)
        self.pages.grid(row=3, column=0, sticky="nsew", padx=20, pady=(0, 20))
        self.pages.rowconfigure(0, weight=1)
        self.pages.columnconfigure(0, weight=1)

        self.device_page = tk.Frame(self.pages, bg=self.CARD_BG)
        self.dataset_page = tk.Frame(self.pages, bg=self.CARD_BG)

        for p in (self.device_page, self.dataset_page):
            p.grid(row=0, column=0, sticky="nsew")
        self.build_device_page(self.device_page)
        self.build_dataset_page(self.dataset_page)

        self.show_page("device")

    def show_page(self, which):
        if which == "device":
            self.device_page.tkraise()
            self.status_lbl.config(text="Status: Device Control page")

        # Hide "Device Control Page" button on this page
            self.btn_device_tab.grid_remove()
            self.btn_dataset_tab.grid()

        # styles
            self.btn_dataset_tab.config(style="Tab.TButton")

        else:
            self.dataset_page.tkraise()
            self.status_lbl.config(text="Status: Dataset & Model page")

        # Hide "Dataset & Model Page" button on this page
            self.btn_dataset_tab.grid_remove()
            self.btn_device_tab.grid()

        # styles
            self.btn_device_tab.config(style="Tab.TButton")

 # ---------------- Common sections ----------------
    def build_connection_bar(self, parent):
        wrap = tk.Frame(parent, bg="white")
        wrap.columnconfigure(0, weight=1)
        wrap.columnconfigure(2, weight=1)

        top = tk.Frame(wrap, bg="white")
        top.grid(row=0, column=1)  # center
        ttk.Label(top, text="COM Port:", style="TLabel").grid(row=0, column=0, sticky="w")
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_var = tk.StringVar(value=ports[0] if ports else "")
        self.port_combo = ttk.Combobox(top, values=ports, textvariable=self.port_var, width=22, state="readonly")
        self.port_combo.grid(row=0, column=1, padx=10)

        ttk.Button(top, text="Refresh", style="Ghost.TButton", command=self.refresh_ports).grid(row=0, column=2, padx=8)
        ttk.Button(top, text="Connect", style="Primary.TButton", command=self.connect).grid(row=0, column=3, padx=8)
        ttk.Button(top, text="Disconnect", style="Ghost.TButton", command=self.disconnect).grid(row=0, column=4, padx=8)

        return wrap

    
    def build_status_bar(self, parent):
        wrap = tk.Frame(parent, bg="white")
        wrap.columnconfigure(0, weight=1)
        wrap.columnconfigure(2, weight=1)

        mid = tk.Frame(wrap, bg="white")
        mid.grid(row=0, column=1, sticky="ew")  # center

        self.status_lbl = ttk.Label(mid, text="Status: Not connected", style="Status.TLabel")
        self.status_lbl.grid(row=0, column=0, sticky="e")

        return wrap

    # ---------------- Page 1: Device ----------------
    def build_device_page(self, parent):
        parent.configure(bg="white")
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

    # Center container (full page center)
        center = tk.Frame(parent, bg="white")
        center.grid(row=0, column=0, sticky="nsew")
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=1)
        center.rowconfigure(2, weight=1)
    # Controls card (center box)
        box = tk.Frame(center, bg="white", highlightthickness=1, highlightbackground="#e5e7eb")
        box.grid(row=1, column=0, padx=80, pady=40, sticky="n")
        box.columnconfigure(0, weight=1)

        ttk.Label(box, text="DEVICE CONTROL", style="Angle.TLabel").grid(row=0, column=0, pady=(18, 10), padx=18, sticky="w")

        btns = tk.Frame(box, bg="white")
        btns.grid(row=1, column=0, padx=18, pady=(10, 18), sticky="ew")
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)

    # Bigger buttons (2x2 grid)
        ttk.Button(btns, text="LED ON", style="Primary.TButton",
               command=lambda: self.send("LED_ON")).grid(row=0, column=0, padx=12, pady=12, sticky="ew", ipady=10)

        ttk.Button(btns, text="LED OFF", style="Ghost.TButton",
               command=lambda: self.send("LED_OFF")).grid(row=0, column=1, padx=12, pady=12, sticky="ew", ipady=10)

        ttk.Button(btns, text="START ROTATE", style="Primary.TButton",
               command=lambda: self.send("START")).grid(row=1, column=0, padx=12, pady=12, sticky="ew", ipady=10)

        ttk.Button(btns, text="STOP ROTATE", style="Danger.TButton",
               command=lambda: self.send("STOP")).grid(row=1, column=1, padx=12, pady=12, sticky="ew", ipady=10)



     # ---------------- Page 2: Dataset ----------------
    def build_dataset_page(self, parent):
        parent.columnconfigure(0, weight=1)

        ttk.Label(parent, text="DATASET & MODEL INPUT", style="Angle.TLabel").grid(row=0, column=0, sticky="w", pady=(10, 10))

        ds = tk.LabelFrame(parent, text="Dataset (Save photos correctly)", bg="white", fg="black", padx=12, pady=12)
        ds.grid(row=1, column=0, sticky="ew", pady=10)
        ds.columnconfigure(1, weight=1)

        ttk.Label(ds, text="Stone ID:", style="TLabel").grid(row=0, column=0, sticky="w", pady=4)
        self.stone_var = tk.StringVar(value="stone001")
        tk.Entry(ds, textvariable=self.stone_var).grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Label(ds, text="Label:", style="TLabel").grid(row=1, column=0, sticky="w", pady=4)
        self.label_var = tk.StringVar(value="real")
        ttk.Combobox(ds, values=["real", "synthetic"], textvariable=self.label_var, state="readonly", width=14).grid(row=1, column=1, sticky="w", padx=8)

        ttk.Button(ds, text="Import Photo for Current Angle", style="Primary.TButton",
                   command=self.import_current_angle_photo).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10, 6))

        row2 = tk.Frame(ds, bg="white")
        row2.grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)

        ttk.Button(row2, text="Open Raw Folder", style="Ghost.TButton", command=self.open_raw_folder).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(row2, text="Make Composite (This Stone)", style="Ghost.TButton", command=self.make_composite_one).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        self.comp_status = ttk.Label(ds, text="Composite: Not created", style="Status.TLabel")
        self.comp_status.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(parent,
                  text="Workflow: Rotate → capture photo → Import → repeat for 0/90/180/270 → Make Composite.",
                  style="Status.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 0))

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
            make_composite_for_stone(label, stone)
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

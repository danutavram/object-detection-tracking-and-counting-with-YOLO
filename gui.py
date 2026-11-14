# gui.py

import threading
import os

import tkinter as tk

import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

from main_tracker import real_time_detection_with_callback
from calibration import (AUTO_PRESET, ROAD_PRESETS, analyze_video, calibrate_auto, calibrate_manual, draw_grid,
                         load_calibration, save_calibration)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PREVIEW_W = 720
PREVIEW_H = 405


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Vehicle Detection & Tracking System")
        self.geometry("1100x760")
        self.resizable(False, False)

        self._latest_update = None   # (frame, stats) - ultima pereche disponibila
        self._outcome = None         # ("done", rezultat) / ("error", mesaj) - lasat de thread-ul de procesare
        self._current_image = None   # referinta pentru a preveni garbage collection
        self.stop_event = threading.Event()
        self.processing_thread = None

        self._build_sidebar()
        self._build_main_panel()

    # ------------------------------------------------------------------ UI --

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="Vehicle\nDetection System",
            font=ctk.CTkFont(size=20, weight="bold"),
        ).pack(pady=(30, 10))

        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=8)

        # --- Video input ---
        ctk.CTkLabel(sidebar, text="Video Input", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=20, pady=(12, 4)
        )
        self.video_entry = ctk.CTkEntry(sidebar, placeholder_text="Selecteaza fisierul video...", width=240)
        self.video_entry.pack(padx=20, pady=(0, 6))
        ctk.CTkButton(sidebar, text="Browse", command=self._browse_video, width=240).pack(padx=20)

        # --- Video output ---
        ctk.CTkLabel(sidebar, text="Video Output", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=20, pady=(14, 4)
        )
        self.output_entry = ctk.CTkEntry(sidebar, placeholder_text="Fisier de iesire...", width=240)
        self.output_entry.pack(padx=20, pady=(0, 6))
        ctk.CTkButton(sidebar, text="Browse", command=self._browse_output, width=240).pack(padx=20)

        # --- Calibrarea vitezei ---
        ctk.CTkLabel(sidebar, text="Calibrare viteza", font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=20, pady=(14, 2)
        )
        self.calib_label = ctk.CTkLabel(
            sidebar, text="Niciun videoclip selectat", text_color="gray60", font=ctk.CTkFont(size=11),
            wraplength=240, justify="left"
        )
        self.calib_label.pack(anchor="w", padx=20, pady=(0, 6))
        ctk.CTkButton(sidebar, text="Calibrare...", command=self._open_calibration, width=240).pack(padx=20)

        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=16)

        # --- Butoane ---
        self.btn_start = ctk.CTkButton(
            sidebar,
            text="START",
            command=self._start,
            fg_color="#1a7a1a",
            hover_color="#145214",
            width=240,
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        self.btn_start.pack(padx=20, pady=(0, 10))

        self.btn_stop = ctk.CTkButton(
            sidebar,
            text="STOP",
            command=self._stop,
            fg_color="#8b0000",
            hover_color="#5c0000",
            width=240,
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            state="disabled",
        )
        self.btn_stop.pack(padx=20)

        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=20)

        # --- Status + progres ---
        self.status_label = ctk.CTkLabel(
            sidebar, text="Gata", text_color="gray60", font=ctk.CTkFont(size=11), wraplength=240
        )
        self.status_label.pack(pady=(0, 8))

        self.progress_bar = ctk.CTkProgressBar(sidebar, width=240)
        self.progress_bar.pack(padx=20)
        self.progress_bar.set(0)

    def _build_main_panel(self):
        main = ctk.CTkFrame(self)
        main.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # --- Preview ---
        self.preview_label = ctk.CTkLabel(main, text="", width=PREVIEW_W, height=PREVIEW_H)
        self.preview_label.pack(pady=(10, 8))
        self._show_placeholder()

        # --- Statistici ---
        stats_outer = ctk.CTkFrame(main)
        stats_outer.pack(fill="x", padx=10, pady=(0, 10))

        stat_defs = [
            ("Se apropie", "entry_count", "#00bcd4"),
            ("Se departeaza", "exit_count", "#ff9800"),
            ("Vehicule Rosii", "red_vehicle_count", "#f44336"),
            ("Viteza Max", "max_speed", "#ffeb3b"),
            ("Numere citite", "plates_read", "#00c8ff"),
            ("Frame", "frame_info", "#e0e0e0"),
        ]

        self.stat_labels = {}
        for col, (title, key, color) in enumerate(stat_defs):
            cell = ctk.CTkFrame(stats_outer)
            cell.grid(row=0, column=col, padx=6, pady=8, sticky="nsew")
            stats_outer.grid_columnconfigure(col, weight=1)

            ctk.CTkLabel(cell, text=title, font=ctk.CTkFont(size=10), text_color="gray55").pack(
                pady=(8, 2)
            )
            val = ctk.CTkLabel(
                cell, text="0", font=ctk.CTkFont(size=22, weight="bold"), text_color=color
            )
            val.pack(pady=(0, 8))
            self.stat_labels[key] = val

        # --- Numerele de inmatriculare citite ---
        ctk.CTkLabel(main, text="Numere de inmatriculare citite", font=ctk.CTkFont(size=12, weight="bold")).pack(
            anchor="w", padx=16)
        self.plates_box = ctk.CTkTextbox(main, height=150, font=ctk.CTkFont(family="Consolas", size=14),
                                         state="disabled", wrap="word")
        self.plates_box.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        self._plates_shown = None

    # --------------------------------------------------------------- Actions --

    def _browse_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
        )
        if not path:
            return
        self.video_entry.delete(0, "end")
        self.video_entry.insert(0, path)
        if not self.output_entry.get():
            base, _ = os.path.splitext(path)
            self.output_entry.insert(0, base + "_output.mp4")
        self._refresh_calibration_status()

    def _refresh_calibration_status(self):
        """Afiseaza daca videoclipul selectat are o calibrare salvata (fara calibrare nu se masoara viteze)."""
        video_path = self.video_entry.get().strip()
        calib = load_calibration(video_path) if video_path else None
        if calib is None:
            self.calib_label.configure(text="Necalibrat: fara viteze", text_color="#ff9800")
        else:
            self.calib_label.configure(text=calib.describe(), text_color="#4caf50")

    def _open_calibration(self):
        video_path = self.video_entry.get().strip()
        if not video_path or not os.path.exists(video_path):
            self._set_status("Selecteaza mai intai un fisier video!", "red")
            return
        CalibrationWindow(self, video_path, on_saved=self._refresh_calibration_status)

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi")],
        )
        if not path:
            return
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, path)

    def _start(self):
        video_path = self.video_entry.get().strip()
        output_path = self.output_entry.get().strip()

        if not video_path:
            self._set_status("Selecteaza un fisier video!", "red")
            return
        if not os.path.exists(video_path):
            self._set_status("Fisierul video nu exista!", "red")
            return
        if not output_path:
            self._set_status("Selecteaza fisierul de iesire!", "red")
            return

        self.stop_event.clear()
        self._latest_update = None
        self._outcome = None
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress_bar.set(0)
        self._set_status("Se pregateste (model, stabilizare)...", "#ffeb3b")

        self.processing_thread = threading.Thread(
            target=self._run_processing,
            args=(video_path, output_path),
            daemon=True,
        )
        self.processing_thread.start()
        self._poll()

    def _stop(self):
        # START redevine activ abia dupa ce thread-ul de procesare s-a oprit efectiv (vezi _finish)
        self.stop_event.set()
        self.btn_stop.configure(state="disabled")
        self._set_status("Se opreste...", "#ff9800")

    # ---------------------------------------------------------- Threading --

    def _run_processing(self, video_path, output_path):
        # Ruleaza in thread-ul de procesare: nu atinge interfata, doar lasa rezultatul pentru _poll.
        try:
            result = real_time_detection_with_callback(
                video_path,
                output_path,
                frame_callback=self._on_frame,
                stop_event=self.stop_event,
            )
            if result is None:
                self._outcome = ("error", "videoclipul nu a putut fi deschis")
            else:
                self._outcome = ("done", result)
        except Exception as exc:
            self._outcome = ("error", str(exc))

    def _on_frame(self, frame, stats):
        # Scriem din thread-ul de procesare; GIL garanteaza atomicitatea
        self._latest_update = (frame, stats)

    def _poll(self):
        alive = self.processing_thread is not None and self.processing_thread.is_alive()
        update = self._latest_update
        if update is not None:
            self._latest_update = None
            frame, stats = update
            self._render_frame(frame)
            self._update_stats(stats)

        if alive:
            self.after(40, self._poll)  # ~25 fps GUI refresh
        else:
            self._finish()

    def _finish(self):
        """Procesarea s-a incheiat (normal, oprita de utilizator sau cu eroare)."""
        kind, value = self._outcome or ("error", "procesarea s-a oprit neasteptat")
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if kind == "error":
            self._set_status(f"Eroare: {value[:150]}", "#f44336")
        elif self.stop_event.is_set():
            self._set_status("Oprit de utilizator.", "#ff9800")
        else:
            self.progress_bar.set(1)
            self._set_status("Procesare finalizata!", "#4caf50")

    # ------------------------------------------------------------- Rendering --

    def _show_placeholder(self):
        img = Image.new("RGB", (PREVIEW_W, PREVIEW_H), color=(18, 18, 18))
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(PREVIEW_W, PREVIEW_H))
        self._current_image = ctk_img
        self.preview_label.configure(image=ctk_img)

    def _render_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb).resize((PREVIEW_W, PREVIEW_H), Image.LANCZOS)
        ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(PREVIEW_W, PREVIEW_H))
        self._current_image = ctk_img
        self.preview_label.configure(image=ctk_img)

    def _update_stats(self, stats):
        self.stat_labels["entry_count"].configure(text=str(stats.get("entry_count", 0)))
        self.stat_labels["exit_count"].configure(text=str(stats.get("exit_count", 0)))
        self.stat_labels["red_vehicle_count"].configure(text=str(stats.get("red_vehicle_count", 0)))
        if stats.get("calibrated", True):
            self.stat_labels["max_speed"].configure(text=f"{stats.get('max_speed', 0):.0f} km/h")
        else:
            self.stat_labels["max_speed"].configure(text="necalibrat")

        self._show_plates(stats.get("plates", []))
        self.stat_labels["plates_read"].configure(text=str(len(stats.get("plates", []))))

        fn = stats.get("frame_number", 0)
        total = stats.get("total_frames", 0)
        self.stat_labels["frame_info"].configure(text=f"{fn}/{total}")

        if total > 0:
            self.progress_bar.set(fn / total)
        if not self.stop_event.is_set():
            self._set_status(f"Procesare... frame {fn}/{total}", "#ffeb3b")

    def _show_plates(self, plates):
        """Lista numerelor citite pana acum: vehicul, numar si viteza (daca videoclipul este calibrat)."""
        if plates == self._plates_shown:
            return
        self._plates_shown = plates
        items = [f"#{tid} {plate}" + (f" ({speed:.0f} km/h)" if speed else "") for tid, plate, speed in plates]
        self.plates_box.configure(state="normal")
        self.plates_box.delete("1.0", "end")
        self.plates_box.insert("1.0", "     ".join(items) if items else "-")
        self.plates_box.configure(state="disabled")
        self.plates_box.see("end")

    def _set_status(self, text, color):
        self.status_label.configure(text=text, text_color=color)


class CalibrationWindow(ctk.CTkToplevel):
    """
    Fereastra de calibrare a vitezei pentru un videoclip.
      - automat: programul gaseste marcajele discontinue si identifica tipul lor (dimensiunile reale); daca
        utilizatorul le cunoaste, le poate impune alegand un tip din lista sau "Personalizat";
      - manual: patru click-uri pe colturile unui dreptunghi de pe drum + latimea si lungimea lui in metri.
    Grila desenata peste imagine (linii din 10 in 10 m) permite verificarea vizuala inainte de salvare.
    """

    VIEW_W = 960

    def __init__(self, master, video_path, on_saved=None):
        super().__init__(master)
        self.title("Calibrare viteza")
        self.resizable(False, False)
        self.video_path = video_path
        self.on_saved = on_saved
        self.calibration = None      # calibrarea curenta (nesalvata)
        self.clicks = []             # punctele alese manual, in coordonatele imaginii originale
        self._photo = None

        self.info = ctk.CTkLabel(self, text="Se analizeaza videoclipul (dureaza pana la un minut)...",
                                 font=ctk.CTkFont(size=12), wraplength=1100)
        self.info.pack(pady=(10, 6))

        self.canvas = tk.Canvas(self, width=self.VIEW_W, height=540, bg="#121212", highlightthickness=0)
        self.canvas.pack(padx=10)
        self.canvas.bind("<Button-1>", self._on_click)

        # --- calibrare automata ---
        row = ctk.CTkFrame(self)
        row.pack(fill="x", padx=10, pady=(8, 4))
        ctk.CTkLabel(row, text="Marcaj:").pack(side="left", padx=(8, 4))
        self.preset = ctk.CTkOptionMenu(row, values=list(ROAD_PRESETS), command=self._on_preset, width=330)
        self.preset.pack(side="left", padx=4)
        self.entries = {}
        for key, label, value in (("period", "liniuta + pauza (m)", "18"), ("dash", "liniuta (m)", "6"),
                                  ("lane", "banda (m)", "3.75")):
            ctk.CTkLabel(row, text=label).pack(side="left", padx=(10, 2))
            entry = ctk.CTkEntry(row, width=55)
            entry.insert(0, value)
            entry.pack(side="left")
            self.entries[key] = entry
        self.btn_auto = ctk.CTkButton(row, text="Calibrare automata", command=self._auto, width=150, state="disabled")
        self.btn_auto.pack(side="right", padx=8, pady=6)

        # --- calibrare manuala ---
        row = ctk.CTkFrame(self)
        row.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(row, text="Manual: 4 click-uri pe un dreptunghi de pe drum "
                               "(aproape-stanga, aproape-dreapta, departe-dreapta, departe-stanga)").pack(
            side="left", padx=(8, 4))
        self.btn_manual = ctk.CTkButton(row, text="Calibrare manuala", command=self._manual, width=150, state="disabled")
        self.btn_manual.pack(side="right", padx=8, pady=6)
        ctk.CTkButton(row, text="Sterge punctele", command=self._clear_clicks, width=110,
                      fg_color="gray30").pack(side="right", padx=4)
        for key, label, value in (("length", "lungime (m)", "18"), ("width", "latime (m)", "3.75")):
            entry = ctk.CTkEntry(row, width=55)
            entry.insert(0, value)
            entry.pack(side="right")
            ctk.CTkLabel(row, text=label).pack(side="right", padx=(10, 2))
            self.entries[key] = entry

        # --- salvare ---
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(4, 10))
        ctk.CTkButton(row, text="Inchide", command=self.destroy, width=120, fg_color="gray30").pack(side="right", padx=4)
        self.btn_save = ctk.CTkButton(row, text="Salveaza calibrarea", command=self._save, width=170,
                                      fg_color="#1a7a1a", hover_color="#145214", state="disabled")
        self.btn_save.pack(side="right", padx=4)

        self.after(100, self._load_background)
        self.after(200, self.grab_set)

    # ------------------------------------------------------- calcule lungi --

    def _run_async(self, work, on_done):
        """
        Ruleaza work() intr-un thread separat (analiza videoclipului dureaza cateva secunde, iar fereastra
        trebuie sa ramana activa) si apeleaza on_done(rezultat, eroare) inapoi in thread-ul interfetei.
        """
        box = {}

        def target():
            try:
                box["result"] = work()
            except Exception as exc:
                box["error"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self.btn_auto.configure(state="disabled")
        self.btn_manual.configure(state="disabled")

        def check():
            if not self.winfo_exists():       # fereastra a fost inchisa intre timp
                return
            if thread.is_alive():
                self.after(100, check)
                return
            if hasattr(self, "bg") or "result" in box:
                self.btn_auto.configure(state="normal")
                self.btn_manual.configure(state="normal")
            on_done(box.get("result"), box.get("error"))

        self.after(100, check)

    # ------------------------------------------------------------ imagine --

    def _load_background(self):
        # imaginea drumului fara vehicule + casetele masinilor (folosite la estimarea scarii reale)
        self._run_async(lambda: analyze_video(self.video_path), self._on_background)

    def _on_background(self, result, error):
        if error is not None:
            self.info.configure(text=f"Videoclipul nu a putut fi analizat: {error}", text_color="#f44336")
            return
        self.bg, self.cars = result
        h, w = self.bg.shape[:2]
        self.scale = self.VIEW_W / w
        self.canvas.configure(height=int(h * self.scale))
        saved = load_calibration(self.video_path)
        if saved is not None:
            self.calibration = saved
            self.info.configure(text=f"Calibrare salvata: {saved.describe()}", text_color="#4caf50")
        else:
            self.info.configure(text="Apasa „Calibrare automata” (programul identifica singur marcajul) "
                                     "sau marcheaza 4 puncte pentru calibrarea manuala.", text_color="gray80")
        self._redraw()

    def _redraw(self):
        image = draw_grid(self.bg, self.calibration) if self.calibration is not None else self.bg.copy()
        for i, (px, py) in enumerate(self.clicks):
            cv2.circle(image, (int(px), int(py)), 9, (255, 0, 255), -1)
            cv2.putText(image, str(i + 1), (int(px) + 12, int(py) - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)
        h, w = image.shape[:2]
        small = cv2.resize(image, (self.VIEW_W, int(h * self.scale)), interpolation=cv2.INTER_AREA)
        self._photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

    # ------------------------------------------------------------- actiuni --

    def _number(self, key):
        return float(self.entries[key].get().replace(",", "."))

    def _on_preset(self, name):
        dims = ROAD_PRESETS.get(name)
        if dims is None:
            return
        for key, value in zip(("period", "dash", "lane"), dims):
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, f"{value:g}")

    def _auto(self):
        if not hasattr(self, "bg"):
            return
        marking = None                      # None = tipul marcajului este identificat de program
        if self.preset.get() != AUTO_PRESET:
            try:
                marking = (self._number("period"), self._number("dash"), self._number("lane"))
            except ValueError:
                self.info.configure(text="Dimensiunile marcajului trebuie sa fie numere.", text_color="#f44336")
                return
            if min(marking) <= 0 or marking[1] >= marking[0]:
                self.info.configure(text="Dimensiunile trebuie sa fie pozitive, iar liniuta mai scurta decat "
                                         "liniuta + pauza.", text_color="#f44336")
                return
        self.info.configure(text="Se cauta marcajele...", text_color="#ffeb3b")
        self._run_async(lambda: calibrate_auto(self.video_path, marking, bg=self.bg, cars=self.cars), self._on_auto)

    def _on_auto(self, calibration, error):
        if error is not None:
            self.info.configure(text=f"{error}. Foloseste calibrarea manuala.", text_color="#f44336")
            return
        self.calibration = calibration
        self.clicks = []
        c = self.calibration.info
        for key in ("period", "dash", "lane"):          # dimensiunile folosite (identificate sau impuse)
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, f"{c[key]:g}")
        text = f"Gasite {c['lines']} linii discontinue ({c['points']} puncte), eroare {c['rmse']:.2f} m. "
        color = "#4caf50"
        if "marking" not in c and abs(c["duty_measured"] - c["dash"] / c["period"]) > 0.06:
            color = "#ff9800"           # dimensiuni impuse de utilizator care nu corespund cu ce se vede in imagine
            text += (f"ATENTIE: in imagine liniuta este {100 * c['duty_measured']:.0f} % din liniuta + pauza, iar la "
                     f"marcajul ales este {100 * c['dash'] / c['period']:.0f} %. Probabil este alt tip de marcaj si "
                     f"vitezele vor fi gresite; alege „Automat”. ")
        if "marking" in c:
            text += f"Marcaj identificat: {c['marking']}. "
            if c["confidence"] == "scazuta":
                color = "#ff9800"
                text += ("Scara este nesigura (vitezele sunt orientative): daca stii dimensiunile reale ale "
                         "marcajului, alege-le din lista, sau foloseste calibrarea manuala. ")
        self.info.configure(text=text + "Verifica daca grila sta pe marcaje, apoi salveaza.", text_color=color)
        self.btn_save.configure(state="normal")
        self._redraw()

    def _on_click(self, event):
        if not hasattr(self, "bg") or len(self.clicks) >= 4:
            return
        self.clicks.append((event.x / self.scale, event.y / self.scale))
        self._redraw()

    def _clear_clicks(self):
        self.clicks = []
        if hasattr(self, "bg"):
            self._redraw()

    def _manual(self):
        if len(self.clicks) != 4:
            self.info.configure(text="Marcheaza exact 4 puncte pe imagine.", text_color="#f44336")
            return
        try:
            width, length = self._number("width"), self._number("length")
        except ValueError:
            self.info.configure(text="Latimea si lungimea trebuie sa fie numere.", text_color="#f44336")
            return
        try:
            self.calibration = calibrate_manual(self.clicks, width, length)
        except ValueError as exc:
            self.info.configure(text=str(exc), text_color="#f44336")
            return
        self.info.configure(text="Calibrare manuala calculata. Verifica grila, apoi salveaza.", text_color="#4caf50")
        self.btn_save.configure(state="normal")
        self._redraw()

    def _save(self):
        if self.calibration is None:
            return
        save_calibration(self.video_path, self.calibration)
        if self.on_saved:
            self.on_saved()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()

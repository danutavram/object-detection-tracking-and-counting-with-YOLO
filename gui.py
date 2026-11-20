# gui.py
# Interfata grafica: alegerea videoclipului, calibrarea vitezei, pornirea procesarii si afisarea rezultatelor.

import os
import subprocess
import sys
import threading

import tkinter as tk

import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk

from main_tracker import real_time_detection_with_callback
from calibration import (AUTO_PRESET, ROAD_PRESETS, analyze_video, calibrate_auto, calibrate_manual, draw_grid,
                         load_calibration, save_calibration, video_background)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PREVIEW_W = 720
PREVIEW_H = 405

GREEN, ORANGE, RED, YELLOW, GRAY = "#4caf50", "#ff9800", "#f44336", "#ffeb3b", "gray60"


def describe_calibration(calib):
    """Descrierea calibrarii pentru interfata (cu diacritice; Calibration.describe() e folosita si in consola)."""
    info = calib.info or {}
    if calib.method != "automata":
        return "Calibrare manuală (4 puncte)."
    text = f"Calibrare automată (eroare {info.get('rmse', 0):.2f} m)."
    if info.get("marking"):
        text += f"\nMarcaj: {info['marking']}."
    if info.get("confidence") == "scazuta":
        text += "\nScara este nesigură: vitezele sunt orientative."
    return text


def open_folder(path):
    """Deschide folderul in exploratorul de fisiere al sistemului."""
    if sys.platform.startswith("win"):
        os.startfile(path)
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", path])


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Detecția și numărarea autovehiculelor")
        self.geometry("1120x820")
        self.resizable(False, False)

        self._latest_update = None   # (frame, stats) - ultima pereche disponibila
        self._outcome = None         # ("done", rezultat) / ("error", mesaj) - lasat de thread-ul de procesare
        self._current_image = None   # referinta pentru a preveni garbage collection
        self._closing = False        # fereastra se inchide dupa ce se opreste procesarea
        self._output_folder = None
        self.stop_event = threading.Event()
        self.processing_thread = None

        self._build_sidebar()
        self._build_main_panel()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------ UI --

    @staticmethod
    def _section(parent, text):
        ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=20, pady=(12, 4))

    @staticmethod
    def _separator(parent, pady=10):
        ctk.CTkFrame(parent, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=pady)

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        ctk.CTkLabel(sidebar, text="Detecția și numărarea\nautovehiculelor",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(22, 4))
        self._separator(sidebar, pady=8)

        # --- videoclipul de intrare ---
        self._section(sidebar, "Videoclip de intrare")
        self.video_entry = ctk.CTkEntry(sidebar, placeholder_text="Alege fișierul video...", width=260)
        self.video_entry.pack(padx=20, pady=(0, 6))
        self.video_entry.bind("<Return>", lambda _: self._on_video_selected())
        self.video_entry.bind("<FocusOut>", lambda _: self._on_video_selected())
        ctk.CTkButton(sidebar, text="Alege...", command=self._browse_video, width=260).pack(padx=20)

        # --- videoclipul de iesire ---
        self._section(sidebar, "Videoclip de ieșire (adnotat)")
        self.output_entry = ctk.CTkEntry(sidebar, placeholder_text="Fișierul rezultat...", width=260)
        self.output_entry.pack(padx=20, pady=(0, 6))
        ctk.CTkButton(sidebar, text="Alege...", command=self._browse_output, width=260).pack(padx=20)
        ctk.CTkLabel(sidebar, text="Tabelul vehiculelor (.csv) se salvează lângă el.", text_color=GRAY,
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=20, pady=(2, 0))

        # --- calibrarea vitezei ---
        self._section(sidebar, "Calibrarea vitezei")
        self.calib_label = ctk.CTkLabel(sidebar, text="Niciun videoclip ales", text_color=GRAY,
                                        font=ctk.CTkFont(size=11), wraplength=260, justify="left")
        self.calib_label.pack(anchor="w", padx=20, pady=(0, 6))
        ctk.CTkButton(sidebar, text="Calibrare...", command=self._open_calibration, width=260).pack(padx=20)

        # --- optiuni ---
        self.plates_switch = ctk.CTkSwitch(sidebar, text="Citește numerele de înmatriculare")
        self.plates_switch.select()
        self.plates_switch.pack(anchor="w", padx=20, pady=(14, 0))
        ctk.CTkLabel(sidebar, text="Util doar dacă plăcuțele sunt lizibile în imagine;\noprit, procesarea e mai rapidă.",
                     text_color=GRAY, font=ctk.CTkFont(size=11), justify="left").pack(anchor="w", padx=20)

        self._separator(sidebar, pady=12)

        # --- butoane ---
        self.btn_start = ctk.CTkButton(sidebar, text="PORNEȘTE", command=self._start, fg_color="#1a7a1a",
                                       hover_color="#145214", width=260, height=40,
                                       font=ctk.CTkFont(size=14, weight="bold"))
        self.btn_start.pack(padx=20, pady=(0, 8))
        self.btn_stop = ctk.CTkButton(sidebar, text="OPREȘTE", command=self._stop, fg_color="#8b0000",
                                      hover_color="#5c0000", width=260, height=40,
                                      font=ctk.CTkFont(size=14, weight="bold"), state="disabled")
        self.btn_stop.pack(padx=20)

        self._separator(sidebar, pady=12)

        # --- stare + progres + rezultate ---
        self.status_label = ctk.CTkLabel(sidebar, text="Alege un videoclip.", text_color=GRAY,
                                         font=ctk.CTkFont(size=12), wraplength=260, justify="left")
        self.status_label.pack(anchor="w", padx=20, pady=(0, 8))
        self.progress_bar = ctk.CTkProgressBar(sidebar, width=260)
        self.progress_bar.pack(padx=20)
        self.progress_bar.set(0)
        self.btn_open = ctk.CTkButton(sidebar, text="Deschide folderul cu rezultatele", command=self._open_results,
                                      width=260, fg_color="gray30", state="disabled")
        self.btn_open.pack(padx=20, pady=(12, 0))

    def _build_main_panel(self):
        main = ctk.CTkFrame(self)
        main.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        # --- imaginea procesata ---
        self.preview_label = ctk.CTkLabel(main, text="", width=PREVIEW_W, height=PREVIEW_H)
        self.preview_label.pack(pady=(10, 8))
        self._show_placeholder()

        # --- statistici ---
        stats_outer = ctk.CTkFrame(main)
        stats_outer.pack(fill="x", padx=10, pady=(0, 4))
        stat_defs = [
            ("Se apropie", "entry_count", "#00bcd4"),
            ("Se depărtează", "exit_count", ORANGE),
            ("Vehicule roșii", "red_vehicle_count", RED),
            ("Viteza maximă", "max_speed", YELLOW),
            ("Numere citite", "plates_read", "#00c8ff"),
            ("Cadru", "frame_info", "#e0e0e0"),
        ]
        self.stat_labels = {}
        for col, (title, key, color) in enumerate(stat_defs):
            cell = ctk.CTkFrame(stats_outer)
            cell.grid(row=0, column=col, padx=6, pady=8, sticky="nsew")
            stats_outer.grid_columnconfigure(col, weight=1)
            ctk.CTkLabel(cell, text=title, font=ctk.CTkFont(size=11), text_color="gray55").pack(pady=(8, 2))
            val = ctk.CTkLabel(cell, text="0", font=ctk.CTkFont(size=22, weight="bold"), text_color=color)
            val.pack(pady=(0, 8))
            self.stat_labels[key] = val

        self.info_label = ctk.CTkLabel(main, text="", text_color=GRAY, font=ctk.CTkFont(size=11))
        self.info_label.pack(anchor="w", padx=16, pady=(0, 6))

        # --- numerele de inmatriculare ---
        ctk.CTkLabel(main, text="Numere de înmatriculare ale vehiculelor numărate",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=16)
        self.plates_box = ctk.CTkTextbox(main, height=150, font=ctk.CTkFont(family="Consolas", size=14),
                                         state="disabled", wrap="word")
        self.plates_box.pack(fill="both", expand=True, padx=10, pady=(2, 10))
        self._plates_shown = None

    # --------------------------------------------------------------- Actions --

    def _browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Fișiere video", "*.mp4 *.avi *.mkv *.mov"),
                                                     ("Toate fișierele", "*.*")])
        if not path:
            return
        self.video_entry.delete(0, "end")
        self.video_entry.insert(0, path)
        self._on_video_selected()

    def _on_video_selected(self):
        """Videoclip nou: iesirea implicita langa el, starea calibrarii si pregatirea fundalului in avans."""
        video_path = self.video_entry.get().strip()
        if not video_path or video_path == getattr(self, "_selected_video", None):
            return
        self._selected_video = video_path
        if not os.path.exists(video_path):
            self.calib_label.configure(text="Fișierul nu există.", text_color=RED)
            return
        # iesirea implicita se schimba doar daca utilizatorul nu a ales el alt fisier
        current = self.output_entry.get().strip()
        if not current or current == getattr(self, "_default_output", None):
            self._default_output = os.path.splitext(video_path)[0] + "_output.mp4"
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, self._default_output)
        self._refresh_calibration_status()
        self._set_status("Apasă PORNEȘTE.", GRAY)
        # imaginea drumului fara vehicule (pentru stabilizare) se calculeaza de acum, ca pornirea sa fie rapida
        if load_calibration(video_path) is not None:
            threading.Thread(target=self._prepare_background, args=(video_path,), daemon=True).start()

    @staticmethod
    def _prepare_background(video_path):
        try:
            video_background(video_path)
        except Exception:
            pass                        # eroarea reapare (si se afiseaza) la pornirea procesarii

    def _refresh_calibration_status(self):
        """Afiseaza daca videoclipul ales are o calibrare salvata (fara calibrare nu se masoara viteze)."""
        video_path = self.video_entry.get().strip()
        calib = load_calibration(video_path) if video_path else None
        if calib is None:
            self.calib_label.configure(text="Necalibrat: se numără vehiculele, fără viteze. Apasă „Calibrare...”.",
                                       text_color=ORANGE)
        else:
            low = (calib.info or {}).get("confidence") == "scazuta"
            self.calib_label.configure(text=describe_calibration(calib), text_color=ORANGE if low else GREEN)

    def _open_calibration(self):
        video_path = self.video_entry.get().strip()
        if not video_path or not os.path.exists(video_path):
            self._set_status("Alege mai întâi un fișier video.", RED)
            return
        CalibrationWindow(self, video_path, on_saved=self._calibration_saved)

    def _calibration_saved(self):
        self._refresh_calibration_status()
        self._selected_video = None
        self._on_video_selected()

    def _browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".mp4",
                                            filetypes=[("Video MP4", "*.mp4"), ("Video AVI", "*.avi")])
        if not path:
            return
        self.output_entry.delete(0, "end")
        self.output_entry.insert(0, path)

    def _start(self):
        self._on_video_selected()
        video_path = self.video_entry.get().strip()
        output_path = self.output_entry.get().strip()
        if not video_path:
            self._set_status("Alege un fișier video.", RED)
            return
        if not os.path.exists(video_path):
            self._set_status("Fișierul video nu există.", RED)
            return
        if not output_path:
            self._set_status("Alege fișierul de ieșire.", RED)
            return
        if os.path.abspath(output_path) == os.path.abspath(video_path):
            self._set_status("Fișierul de ieșire nu poate fi chiar videoclipul de intrare.", RED)
            return

        self.stop_event.clear()
        self._latest_update = None
        self._outcome = None
        self._plates_shown = None
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_open.configure(state="disabled")
        self.progress_bar.set(0)
        self.info_label.configure(text="")
        for key in ("entry_count", "exit_count", "red_vehicle_count", "plates_read", "max_speed", "frame_info"):
            self.stat_labels[key].configure(text="0")
        self._show_text("")
        self._set_status("Se pregătește (modelul, fundalul pentru stabilizare)...", YELLOW)

        self.processing_thread = threading.Thread(
            target=self._run_processing,
            args=(video_path, output_path, bool(self.plates_switch.get())),
            daemon=True,
        )
        self.processing_thread.start()
        self._poll()

    def _stop(self):
        # PORNESTE redevine activ abia dupa ce thread-ul de procesare s-a oprit efectiv (vezi _finish)
        self.stop_event.set()
        self.btn_stop.configure(state="disabled")
        self._set_status("Se oprește...", ORANGE)

    def _on_close(self):
        """Inchiderea ferestrei in timpul procesarii: intai se opreste procesarea (videoclipul de iesire se inchide
        corect), apoi fereastra."""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            self._closing = True
            self._stop()
        else:
            self.destroy()

    def _open_results(self):
        if self._output_folder and os.path.isdir(self._output_folder):
            open_folder(self._output_folder)

    # ---------------------------------------------------------- Threading --

    def _run_processing(self, video_path, output_path, read_plates):
        # Ruleaza in thread-ul de procesare: nu atinge interfata, doar lasa rezultatul pentru _poll.
        try:
            result = real_time_detection_with_callback(video_path, output_path, frame_callback=self._on_frame,
                                                       stop_event=self.stop_event, read_plates=read_plates)
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
            self.after(40, self._poll)  # ~25 actualizari pe secunda
        else:
            self._finish()

    def _finish(self):
        """Procesarea s-a incheiat (normal, oprita de utilizator sau cu eroare)."""
        if self._closing:
            self.destroy()
            return
        kind, value = self._outcome or ("error", "procesarea s-a oprit neașteptat")
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        if kind == "error":
            self._set_status(f"Eroare: {value[:200]}", RED)
            return
        self._update_stats(value, final=True)
        output_path = self.output_entry.get().strip()
        self._output_folder = os.path.dirname(os.path.abspath(output_path))
        self.btn_open.configure(state="normal")
        red = value["red_vehicle_count"]
        summary = (f"{value['entry_count']} se apropie, {value['exit_count']} se depărtează, "
                   f"{red} {'vehicul roșu' if red == 1 else 'vehicule roșii'}.\n"
                   f"Salvate: {os.path.basename(output_path)} și {os.path.basename(value['table_path'])}.")
        if self.stop_event.is_set():
            self._set_status("Oprit de utilizator (rezultate parțiale). " + summary, ORANGE)
        else:
            self.progress_bar.set(1)
            self._set_status("Gata: " + summary, GREEN)

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

    def _update_stats(self, stats, final=False):
        self.stat_labels["entry_count"].configure(text=str(stats.get("entry_count", 0)))
        self.stat_labels["exit_count"].configure(text=str(stats.get("exit_count", 0)))
        self.stat_labels["red_vehicle_count"].configure(text=str(stats.get("red_vehicle_count", 0)))
        if stats.get("calibrated", True):
            self.stat_labels["max_speed"].configure(text=f"{stats.get('max_speed', 0):.0f} km/h")
        else:
            self.stat_labels["max_speed"].configure(text="necalibrat")

        plates_status = stats.get("plates_status")
        if plates_status is None:
            self.stat_labels["plates_read"].configure(text=str(stats.get("plates_read", 0)))
            if not final:
                self._show_plates(stats.get("plates", []))
        else:
            self.stat_labels["plates_read"].configure(text="–")
            if plates_status == "oprita":
                self._show_text("Citirea numerelor de înmatriculare este oprită (comutatorul din stânga).")
            else:                       # "indisponibila (motivul)": biblioteca lipseste sau modelele nu s-au descarcat
                reason = plates_status[len("indisponibila"):].strip(" ()")
                self._show_text("Citirea numerelor de înmatriculare nu este disponibilă pe acest calculator "
                                f"(la prima folosire are nevoie de internet). Motivul: {reason}")

        if "line_ratio" in stats:
            self.info_label.configure(text=f"Linia de numărare (galbenă): {100 * stats['line_ratio']:.0f} % din "
                                           "înălțimea imaginii, aleasă automat (cea la care se numără cele mai multe "
                                           "vehicule).")
        if final:
            return
        fn = stats.get("frame_number", 0) + 1
        total = stats.get("total_frames", 0)
        self.stat_labels["frame_info"].configure(text=f"{fn}/{total}")
        if total > 0:
            self.progress_bar.set(fn / total)
        if not self.stop_event.is_set():
            self._set_status(f"Se procesează: cadrul {fn} din {total}...", YELLOW)

    def _show_plates(self, plates):
        """Lista numerelor citite pana acum: vehicul, numar si viteza (daca videoclipul este calibrat)."""
        if plates == self._plates_shown:
            return
        self._plates_shown = plates
        nbsp = " "                 # spatiu care nu permite ruperea randului in mijlocul unei intrari
        items = [f"#{tid}{nbsp}{plate}" + (f"{nbsp}({speed:.0f}{nbsp}km/h)" if speed else "")
                 for tid, plate, speed in plates]
        self._show_text("     ".join(items) if items else "–")

    def _show_text(self, text):
        self.plates_box.configure(state="normal")
        self.plates_box.delete("1.0", "end")
        self.plates_box.insert("1.0", text)
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
        self.title("Calibrarea vitezei")
        self.resizable(False, False)
        self.transient(master)           # ramane deasupra ferestrei principale
        self.video_path = video_path
        self.on_saved = on_saved
        self.calibration = None      # calibrarea curenta (nesalvata)
        self.clicks = []             # punctele alese manual, in coordonatele imaginii originale
        self._photo = None

        self.info = ctk.CTkLabel(self, text="Se analizează videoclipul (durează până la un minut)...",
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
        for key, label, value in (("period", "liniuță + pauză (m)", "18"), ("dash", "liniuță (m)", "6"),
                                  ("lane", "bandă (m)", "3.75")):
            ctk.CTkLabel(row, text=label).pack(side="left", padx=(10, 2))
            entry = ctk.CTkEntry(row, width=55)
            entry.insert(0, value)
            entry.pack(side="left")
            self.entries[key] = entry
        self.btn_auto = ctk.CTkButton(row, text="Calibrare automată", command=self._auto, width=150, state="disabled")
        self.btn_auto.pack(side="right", padx=8, pady=6)

        # --- calibrare manuala ---
        row = ctk.CTkFrame(self)
        row.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(row, text="Manual: 4 click-uri pe un dreptunghi de pe drum "
                               "(aproape-stânga, aproape-dreapta, departe-dreapta, departe-stânga)").pack(
            side="left", padx=(8, 4))
        self.btn_manual = ctk.CTkButton(row, text="Calibrare manuală", command=self._manual, width=150,
                                        state="disabled")
        self.btn_manual.pack(side="right", padx=8, pady=6)
        ctk.CTkButton(row, text="Șterge punctele", command=self._clear_clicks, width=110,
                      fg_color="gray30").pack(side="right", padx=4)
        for key, label, value in (("length", "lungime (m)", "18"), ("width", "lățime (m)", "3.75")):
            entry = ctk.CTkEntry(row, width=55)
            entry.insert(0, value)
            entry.pack(side="right")
            ctk.CTkLabel(row, text=label).pack(side="right", padx=(10, 2))
            self.entries[key] = entry

        # --- salvare ---
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=(4, 10))
        ctk.CTkButton(row, text="Închide", command=self.destroy, width=120, fg_color="gray30").pack(side="right", padx=4)
        self.btn_save = ctk.CTkButton(row, text="Salvează calibrarea", command=self._save, width=170,
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
            self.info.configure(text=f"Videoclipul nu a putut fi analizat: {error}", text_color=RED)
            return
        self.bg, self.cars = result
        h, w = self.bg.shape[:2]
        self.scale = self.VIEW_W / w
        self.canvas.configure(height=int(h * self.scale))
        saved = load_calibration(self.video_path)
        if saved is not None:
            self.calibration = saved
            self.info.configure(text="Calibrare salvată. " + describe_calibration(saved).replace("\n", " "),
                                text_color=GREEN)
        else:
            self.info.configure(text="Apasă „Calibrare automată” (programul identifică singur marcajul) "
                                     "sau marchează 4 puncte pentru calibrarea manuală.", text_color="gray80")
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
                self.info.configure(text="Dimensiunile marcajului trebuie să fie numere.", text_color=RED)
                return
            if min(marking) <= 0 or marking[1] >= marking[0]:
                self.info.configure(text="Dimensiunile trebuie să fie pozitive, iar liniuța mai scurtă decât "
                                         "liniuța + pauza.", text_color=RED)
                return
        self.info.configure(text="Se caută marcajele...", text_color=YELLOW)
        self._run_async(lambda: calibrate_auto(self.video_path, marking, bg=self.bg, cars=self.cars), self._on_auto)

    def _on_auto(self, calibration, error):
        if error is not None:
            self.info.configure(text=f"{error}. Folosește calibrarea manuală.", text_color=RED)
            return
        self.calibration = calibration
        self.clicks = []
        c = self.calibration.info
        for key in ("period", "dash", "lane"):          # dimensiunile folosite (identificate sau impuse)
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, f"{c[key]:g}")
        found = "o linie discontinuă" if c["lines"] == 1 else f"{c['lines']} linii discontinue"
        text = f"S-a găsit {found} ({c['points']} puncte), eroare {c['rmse']:.2f} m. "
        color = GREEN
        if "marking" not in c and abs(c["duty_measured"] - c["dash"] / c["period"]) > 0.06:
            color = ORANGE          # dimensiuni impuse de utilizator care nu corespund cu ce se vede in imagine
            text += (f"ATENȚIE: în imagine liniuța este {100 * c['duty_measured']:.0f} % din liniuță + pauză, iar la "
                     f"marcajul ales este {100 * c['dash'] / c['period']:.0f} %. Probabil este alt tip de marcaj și "
                     f"vitezele vor fi greșite; alege „Automat”. ")
        if "marking" in c:
            text += f"Marcaj identificat: {c['marking']}. "
            if c["confidence"] == "scazuta":
                color = ORANGE
                text += ("Scara este nesigură (vitezele sunt orientative): dacă știi dimensiunile reale ale "
                         "marcajului, alege-le din listă, sau folosește calibrarea manuală. ")
        self.info.configure(text=text + "Verifică dacă grila stă pe marcaje, apoi salvează.", text_color=color)
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
            self.info.configure(text="Marchează exact 4 puncte pe imagine.", text_color=RED)
            return
        try:
            width, length = self._number("width"), self._number("length")
        except ValueError:
            self.info.configure(text="Lățimea și lungimea trebuie să fie numere.", text_color=RED)
            return
        try:
            self.calibration = calibrate_manual(self.clicks, width, length)
        except ValueError as exc:
            self.info.configure(text=str(exc), text_color=RED)
            return
        self.info.configure(text="Calibrare manuală calculată. Verifică grila, apoi salvează.", text_color=GREEN)
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

# gui.py

import threading
import queue
import os

import cv2
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image

from main_tracker import real_time_detection_with_callback

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

PREVIEW_W = 720
PREVIEW_H = 405


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Vehicle Detection & Tracking System")
        self.geometry("1100x660")
        self.resizable(False, False)

        self._latest_update = None   # (frame, stats) - ultima pereche disponibila
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

        ctk.CTkFrame(sidebar, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=20)

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
            sidebar, text="Gata", text_color="gray60", font=ctk.CTkFont(size=11)
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
            ("Vehicule Intrate", "entry_count", "#00bcd4"),
            ("Vehicule Iesite", "exit_count", "#ff9800"),
            ("Vehicule Rosii", "red_vehicle_count", "#f44336"),
            ("Viteza Max", "max_speed", "#ffeb3b"),
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
        if not output_path:
            self._set_status("Selecteaza fisierul de iesire!", "red")
            return

        self.stop_event.clear()
        self._latest_update = None
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.progress_bar.set(0)
        self._set_status("Se incarca modelul...", "#ffeb3b")

        self.processing_thread = threading.Thread(
            target=self._run_processing,
            args=(video_path, output_path),
            daemon=True,
        )
        self.processing_thread.start()
        self._poll()

    def _stop(self):
        self.stop_event.set()
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self._set_status("Oprit de utilizator.", "#ff9800")

    # ---------------------------------------------------------- Threading --

    def _run_processing(self, video_path, output_path):
        try:
            real_time_detection_with_callback(
                video_path,
                output_path,
                frame_callback=self._on_frame,
                stop_event=self.stop_event,
            )
            if not self.stop_event.is_set():
                self.after(0, self._on_done)
        except Exception as exc:
            self.after(0, lambda: self._on_error(str(exc)))

    def _on_frame(self, frame, stats):
        # Scriem din thread-ul de procesare; GIL garanteaza atomicitatea
        self._latest_update = (frame, stats)

    def _poll(self):
        update = self._latest_update
        if update is not None:
            self._latest_update = None
            frame, stats = update
            self._render_frame(frame)
            self._update_stats(stats)

        if self.processing_thread and self.processing_thread.is_alive():
            self.after(40, self._poll)  # ~25 fps GUI refresh

    def _on_done(self):
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self.progress_bar.set(1)
        self._set_status("Procesare finalizata!", "#4caf50")

    def _on_error(self, msg):
        self.btn_start.configure(state="normal")
        self.btn_stop.configure(state="disabled")
        self._set_status(f"Eroare: {msg[:50]}", "#f44336")

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
        self.stat_labels["max_speed"].configure(text=f"{stats.get('max_speed', 0):.0f} km/h")

        fn = stats.get("frame_number", 0)
        total = stats.get("total_frames", 0)
        self.stat_labels["frame_info"].configure(text=f"{fn}/{total}")

        if total > 0:
            self.progress_bar.set(fn / total)
        self._set_status(f"Procesare... frame {fn}/{total}", "#ffeb3b")

    def _set_status(self, text, color):
        self.status_label.configure(text=text, text_color=color)


if __name__ == "__main__":
    app = App()
    app.mainloop()

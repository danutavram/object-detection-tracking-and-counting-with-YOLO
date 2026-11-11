# tracker.py
# Logica de numarare, masurare a vitezei si identificare a vehiculelor rosii,
# pe baza identificatorilor (ID) furnizati de algoritmul de urmarire ByteTrack.
from collections import Counter

from speed_estimator import ground_point, fully_visible, speed_from_track
from red_vehicle_counter import is_vehicle_red


class Tracker:
    def __init__(self, fps, frame_w, frame_h, calibration=None, counting_line_ratio=0.5):
        self.fps = float(fps)                      # fps real (ex. 29.97), fara trunchiere
        self.frame_w, self.frame_h = frame_w, frame_h
        self.counting_line = int(frame_h * counting_line_ratio)   # linia de numarare
        self.calibration = calibration             # None = videoclip necalibrat: nu se masoara viteze
        self.last_cy = {}        # id -> ultima ordonata a centrului casetei
        self.samples = {}        # id -> [(t, Y_real)] pozitii valide pe drum
        self.classes = {}        # id -> Counter cu clasele detectate (se retine cea mai frecventa)
        self.counted = {}        # id -> 'in' (se apropie de camera) / 'out' (se departeaza)
        self.cross_frame = {}    # id -> cadrul in care vehiculul a traversat linia
        self.speeds = {}         # id -> viteza (km/h), doar dupa ce estimarea este stabila
        self.red_ids = set()     # id-urile vehiculelor rosii
        self.entry_count = self.exit_count = self.red_count = 0

    def update(self, frame, detections, frame_number, shift=(0.0, 0.0)):
        """
        Actualizeaza starea cu vehiculele din cadrul curent.
        detections: lista de (x, y, w, h, track_id, cls, confidence, label).
        shift: deplasarea camerei (dx, dy) in acest cadru fata de imaginea de referinta a calibrarii.
        Returneaza statisticile curente.
        """
        t = frame_number / self.fps
        for x, y, w, h, tid, cls, *_ in detections:
            cy = y + h // 2
            self.classes.setdefault(tid, Counter())[cls] += 1

            # --- Viteza: regresie pe pozitiile reale ale punctului de contact cu solul ---
            if self.calibration is not None and fully_visible(x, y, w, h, self.frame_w, self.frame_h):
                gx, gy = ground_point(x, y, w, h)
                gx, gy = gx - shift[0], gy - shift[1]          # compensarea miscarii camerei
                if self.calibration.usable(gx, gy):
                    self.samples.setdefault(tid, []).append((t, self.calibration.to_world(gx, gy)[1]))
                    v = speed_from_track(self.samples[tid])
                    if v is not None:
                        self.speeds[tid] = v

            # --- Numarare la traversarea liniei (fiecare vehicul o singura data) ---
            prev = self.last_cy.get(tid)
            if prev is not None and tid not in self.counted:
                if prev < self.counting_line <= cy:        # se deplaseaza in jos: se apropie de camera
                    self.counted[tid] = 'in'
                    self.entry_count += 1
                elif prev > self.counting_line >= cy:      # se deplaseaza in sus: se departeaza de camera
                    self.counted[tid] = 'out'
                    self.exit_count += 1
                if tid in self.counted:
                    self.cross_frame[tid] = frame_number
                    # culoarea se verifica in momentul traversarii liniei (toate benzile sunt vizibile)
                    roi = frame[max(y, 0):min(y + h, self.frame_h), max(x, 0):min(x + w, self.frame_w)]
                    if is_vehicle_red(roi):
                        self.red_ids.add(tid)
                        self.red_count += 1
            self.last_cy[tid] = cy

        counted_speeds = [self.speeds[i] for i in self.counted if i in self.speeds]
        return {
            'entry_count': self.entry_count,
            'exit_count': self.exit_count,
            'red_vehicle_count': self.red_count,
            'max_speed': max(counted_speeds) if counted_speeds else 0.0,
        }

    def vehicles_table(self, class_names):
        """Cate un rand pentru fiecare vehicul numarat: (id, timp [s], sens, tip, rosu, viteza [km/h] sau None)."""
        rows = []
        for tid in sorted(self.counted, key=lambda i: self.cross_frame[i]):
            cls = self.classes[tid].most_common(1)[0][0]
            rows.append((tid, round(self.cross_frame[tid] / self.fps, 2),
                         'se apropie' if self.counted[tid] == 'in' else 'se departeaza',
                         class_names[cls], 'da' if tid in self.red_ids else 'nu',
                         round(self.speeds[tid], 1) if tid in self.speeds else None))
        return rows

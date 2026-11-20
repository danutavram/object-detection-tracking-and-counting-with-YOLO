# tracker.py
# Logica de numarare, masurare a vitezei si identificare a vehiculelor rosii,
# pe baza identificatorilor (ID) furnizati de algoritmul de urmarire ByteTrack.
from collections import Counter

from speed_estimator import ground_point, ground_hidden, fully_visible, speed_from_track
from red_vehicle_counter import is_vehicle_red, scene_color_cast


# Inaltimile posibile ale liniei de numarare (fractiuni din inaltimea cadrului). Se numara la toate deodata si se
# foloseste linia la care au fost numarate cele mai multe vehicule (la egalitate, cea mai apropiata de mijloc).
# Motivul: unde vehiculele se acopera unele pe altele (coloana filmata cu teleobiectivul) sau ies lateral din imagine,
# urmaritorul pierde vehiculul exact langa linie si trecerea nu se numara, deci linia gresita numara mai putin.
# Verificat pe clipurile de test (CLAUDE.md, „Numararea”): nicio inaltime fixa nu e buna pe toate.
LINE_RATIOS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


class CountingLine:
    """O linie orizontala de numarare si vehiculele care au traversat-o."""

    def __init__(self, y):
        self.y = y
        self.counted = {}        # id -> 'in' (se apropie de camera) / 'out' (se departeaza)
        self.cross_frame = {}    # id -> cadrul in care vehiculul a traversat linia
        self.red_ids = set()     # id-urile vehiculelor rosii numarate la aceasta linie


class Tracker:
    def __init__(self, fps, frame_w, frame_h, calibration=None, line_ratios=LINE_RATIOS):
        self.fps = float(fps)                      # fps real (ex. 29.97), fara trunchiere
        self.frame_w, self.frame_h = frame_w, frame_h
        self.lines = [CountingLine(int(frame_h * r)) for r in line_ratios]
        self.calibration = calibration             # None = videoclip necalibrat: nu se masoara viteze
        self.last_cy = {}        # id -> ultima ordonata a centrului casetei
        self.samples = {}        # id -> [(t, X, Y, inaltime/latime, punct ascuns)] pozitii valide pe drum, in metri
        self.aspects = {}        # id -> raportul inaltime/latime al casetei in fiecare cadru
        self.classes = {}        # id -> Counter cu clasele detectate (se retine cea mai frecventa)
        self.speeds = {}         # id -> viteza (km/h), doar dupa ce estimarea este stabila

    # Rezultatele sunt cele ale liniei la care s-au numarat pana acum cele mai multe vehicule. In primele secunde
    # linia aleasa se poate schimba; totalul afisat nu scade niciodata.
    @property
    def best_line(self):
        return max(self.lines, key=lambda line: len(line.counted))

    @property
    def counting_line(self):
        return self.best_line.y

    @property
    def counted(self):
        return self.best_line.counted

    @property
    def cross_frame(self):
        return self.best_line.cross_frame

    @property
    def red_ids(self):
        return self.best_line.red_ids

    def update(self, frame, detections, frame_number, shift=(0.0, 0.0)):
        """
        Actualizeaza starea cu vehiculele din cadrul curent.
        detections: lista de (x, y, w, h, track_id, cls, confidence, label).
        shift: deplasarea camerei (dx, dy) in acest cadru fata de imaginea de referinta a calibrarii.
        Returneaza statisticile curente.
        """
        t = frame_number / self.fps
        boxes = [d[:4] for d in detections]
        cast = None              # dominanta de culoare a cadrului, calculata doar daca un vehicul traverseaza o linie
        for k, (x, y, w, h, tid, cls, *_) in enumerate(detections):
            cy = y + h // 2
            self.classes.setdefault(tid, Counter())[cls] += 1
            self.aspects.setdefault(tid, []).append(h / max(w, 1))

            # --- Viteza: regresie pe pozitiile reale ale punctului de contact cu solul ---
            if self.calibration is not None and fully_visible(x, y, w, h, self.frame_w, self.frame_h):
                gx, gy = ground_point(x, y, w, h)
                hidden = ground_hidden(gx, gy, boxes[:k] + boxes[k + 1:])
                gx, gy = gx - shift[0], gy - shift[1]          # compensarea miscarii camerei
                if self.calibration.usable(gx, gy):
                    self.samples.setdefault(tid, []).append((t, *self.calibration.to_world(gx, gy), h / max(w, 1), hidden))
                    v = speed_from_track(self.samples[tid], self.aspects[tid])
                    if v is not None:
                        self.speeds[tid] = v
                    else:
                        self.speeds.pop(tid, None)             # estimarea nu (mai) e sigura: nu se afiseaza

            # --- Numarare la traversarea fiecarei linii (fiecare vehicul o singura data pe linie) ---
            prev = self.last_cy.get(tid)
            if prev is not None:
                for line in self.lines:
                    if tid in line.counted:
                        continue
                    if prev < line.y <= cy:            # se deplaseaza in jos: se apropie de camera
                        line.counted[tid] = 'in'
                    elif prev > line.y >= cy:          # se deplaseaza in sus: se departeaza de camera
                        line.counted[tid] = 'out'
                    else:
                        continue
                    line.cross_frame[tid] = frame_number
                    # culoarea se verifica in momentul traversarii liniei, relativ la dominanta de culoare a cadrului
                    if cast is None:
                        cast = scene_color_cast(frame)
                    closer = [b for b in boxes if b[1] + b[3] > y + h]      # vehicule mai apropiate de camera
                    if is_vehicle_red(frame, (x, y, w, h), cast, closer):
                        line.red_ids.add(tid)
            self.last_cy[tid] = cy

        counted = self.counted
        counted_speeds = [self.speeds[i] for i in counted if i in self.speeds]
        return {
            'entry_count': sum(1 for d in counted.values() if d == 'in'),
            'exit_count': sum(1 for d in counted.values() if d == 'out'),
            'red_vehicle_count': len(self.red_ids),
            'max_speed': max(counted_speeds) if counted_speeds else 0.0,
        }

    def vehicles_table(self, class_names, plates=None):
        """Cate un rand pentru fiecare vehicul numarat: (id, timp [s], sens, tip, rosu, viteza [km/h] sau None,
        numar de inmatriculare sau None)."""
        plates = plates or {}
        rows = []
        for tid in sorted(self.counted, key=lambda i: self.cross_frame[i]):
            cls = self.classes[tid].most_common(1)[0][0]
            rows.append((tid, round(self.cross_frame[tid] / self.fps, 2),
                         'se apropie' if self.counted[tid] == 'in' else 'se departeaza',
                         class_names[cls], 'da' if tid in self.red_ids else 'nu',
                         round(self.speeds[tid], 1) if tid in self.speeds else None, plates.get(tid)))
        return rows

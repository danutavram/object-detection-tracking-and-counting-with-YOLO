# plate_reader.py
# Citirea numerelor de inmatriculare (LPR):
#  - in decupajul fiecarui vehicul urmarit se cauta placuta (detector YOLOv9, ONNX) si se citeste textul (retea OCR);
#  - un vehicul este citit in mai multe cadre, iar numarul final se obtine prin vot, caracter cu caracter:
#    o citire izolata poate gresi o litera, mai ales cand placuta este mica, dar citirile repetate se corecteaza reciproc.
# Foloseste biblioteca fast-alpr (modelele se descarca automat la prima rulare). Placutele trebuie sa fie suficient de
# mari in imagine: pe filmari de la distanta nu se poate citi nimic.
from collections import Counter, defaultdict

MIN_VEHICLE_WIDTH = 110      # vehiculele mai inguste (in pixeli) sunt prea departe pentru a avea placuta lizibila
MIN_PLATE_WIDTH = 45         # placutele mai mici nu se citesc
READ_EVERY = 2               # un vehicul este citit o data la atatea cadre
MAX_READINGS = 25            # dupa atatea citiri numarul nu se mai schimba; vehiculul nu mai este citit
MIN_CONFIDENCE = 0.75        # increderea medie minima a caracterelor unei citiri
MIN_LENGTH, MAX_LENGTH = 5, 8
MIN_READINGS = 3             # numarul minim de citiri pentru ca un numar sa fie raportat


def consensus(readings):
    """
    readings: lista de (text, incredere). Se pastreaza citirile cu lungimea cea mai frecventa si se alege, pe fiecare
    pozitie, caracterul cu cea mai mare suma a increderilor. Returneaza textul sau None daca sunt prea putine citiri.
    """
    if len(readings) < MIN_READINGS:
        return None
    by_length = Counter()
    for text, confidence in readings:
        by_length[len(text)] += confidence
    length = by_length.most_common(1)[0][0]
    same = [(text, confidence) for text, confidence in readings if len(text) == length]
    if len(same) < MIN_READINGS:
        return None
    result = ""
    for position in range(length):
        votes = Counter()
        for text, confidence in same:
            votes[text[position]] += confidence
        result += votes.most_common(1)[0][0]
    return result


class PlateReader:
    def __init__(self):
        from fast_alpr import ALPR          # import aici: programul functioneaza si fara biblioteca, doar fara LPR
        self.alpr = ALPR(detector_conf_thresh=0.3)
        self.readings = defaultdict(list)   # id vehicul -> [(text, incredere)]
        self.plates = {}                    # id vehicul -> numarul stabilit prin vot

    def update(self, frame, detections, frame_number):
        """Citeste placutele vehiculelor din cadrul curent (cadrul trebuie sa fie cel original, neadnotat)."""
        if frame_number % READ_EVERY:
            return self.plates
        frame_h, frame_w = frame.shape[:2]
        for x, y, w, h, tid, *_ in detections:
            if w < MIN_VEHICLE_WIDTH or len(self.readings[tid]) >= MAX_READINGS:
                continue
            crop = frame[max(y, 0):min(y + h, frame_h), max(x, 0):min(x + w, frame_w)]
            if crop.size == 0:
                continue
            reading = self._read(crop)
            if reading is not None:
                self.readings[tid].append(reading)
                plate = consensus(self.readings[tid])
                if plate is not None:
                    self.plates[tid] = plate
        return self.plates

    def _read(self, crop):
        """Cea mai sigura placuta din decupaj: (text, incredere) sau None."""
        best = None
        for result in self.alpr.predict(crop):
            if result.ocr is None or not result.ocr.text:
                continue
            box = result.detection.bounding_box
            confidence = result.ocr.confidence
            if hasattr(confidence, "__len__"):
                confidence = float(sum(confidence) / max(len(confidence), 1))
            text = "".join(ch for ch in result.ocr.text.upper() if ch.isalnum())
            if (box.x2 - box.x1 < MIN_PLATE_WIDTH or confidence < MIN_CONFIDENCE
                    or not MIN_LENGTH <= len(text) <= MAX_LENGTH):
                continue
            if best is None or confidence > best[1]:
                best = (text, float(confidence))
        return best

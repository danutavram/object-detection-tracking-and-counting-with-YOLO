import cv2
from model_loader import load_yolo_model
from video_manager import open_video, initialize_video_writer, release_video
from vehicle_detector import detect_vehicles
from tracker import Tracker
from red_vehicle_counter import is_vehicle_red


def real_time_detection(video_path, output_path):
    # Incarca modelul YOLO
    model = load_yolo_model("yolo11s.pt")

    # Deschide videoclipul
    cap = open_video(video_path)
    if cap is None:
        return

    # Proprietati video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initializeaza scrierea videoclipului de iesire
    out = initialize_video_writer(frame_width, frame_height, output_path, fps)

    # Creeaza tracker
    tracker = Tracker()
    counted_red_vehicles = set()  # Vehicule rosii contorizate pentru linia rosie
    red_vehicle_count = 0  # Contor vehicule rosii
    entry_count, exit_count = 0, 0  # Contoare vehicule intrate si iesite

    # Definirea liniilor
    detection_line = int(frame_height * 0.5)  # Linie galbena (intrare/iesire)
    red_detection_line = int(frame_height * 0.6)  # Linie rosie pentru vehicule rosii

    # Procesare cadre video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detecteaza vehicule
        detections = detect_vehicles(frame, model)
        tracked_objects, entered, exited = tracker.update([(x, y, w, h) for x, y, w, h, _ in detections], detection_line)

        entry_count += entered  # Actualizam numarul vehiculelor intrate
        exit_count += exited    # Actualizam numarul vehiculelor iesite

        # Procesare detectari
        for obj, det in zip(tracked_objects, detections):
            x, y, w, h, obj_id = obj
            label = det[-1]
            cx, cy = x + w // 2, y + h // 2  # Calculam centrul

            # ROI vehicul
            x_end, y_end = min(x + w, frame_width), min(y + h, frame_height)
            x_start, y_start = max(x, 0), max(y, 0)
            vehicle_roi = frame[y_start:y_end, x_start:x_end]
            
            # Detecteaza vehicule rosii care trec linia rosie
            if abs(cy - red_detection_line) <= 10:
                if is_vehicle_red(vehicle_roi):
                    if obj_id not in counted_red_vehicles:
                        counted_red_vehicles.add(obj_id)
                        red_vehicle_count += 1
                        # Evidentiaza vehiculul rosu
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Red Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Evidentiaza toate vehiculele detectate
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Deseneaza liniile orizontale
        cv2.line(frame, (0, detection_line), (frame_width, detection_line), (0, 255, 255), 2)  # Linie galbena
        cv2.putText(frame, "Vehicule Intrate", (10, detection_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Vehicule Iesite", (10, detection_line + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        cv2.line(frame, (0, red_detection_line), (frame_width, red_detection_line), (0, 0, 255), 2)  # Linie rosie
        cv2.putText(frame, "Vehicule Rosii", (10, red_detection_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Afiseaza contoarele
        cv2.putText(frame, f"Vehicule Intrate: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Vehicule Iesite: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(frame, f"Vehicule Rosii: {red_vehicle_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Scrie cadrul in fisierul de iesire
        out.write(frame)

    # Elibereaza resurse
    release_video(cap, out)
# main.py

from main_tracker import real_time_detection

def main():
    video_path = "video0.mp4"
    output_path = "output_video25.mp4"
    real_time_detection(video_path, output_path)

if __name__ == "__main__":
    main()
# model_loader.py

from ultralytics import YOLO

def load_yolo_model(model_path="yolo11s.pt"):
    """Incarca modelul YOLO si efectueaza fuziunea pentru a optimiza performanta de inferenta."""
    model = YOLO(model_path)
    # model.fuse()  # Fuzioneaza straturile pentru imbunatatirea performantei
    return model
# red_vehicle_counter.py

import cv2
import numpy as np

# Functia 'is_vehicle_red' determina daca un vehicul este de culoare rosie
def is_vehicle_red(vehicle_roi, min_red_pixel_ratio=0.13):
    # Verifica daca regiunea de interes (ROI) a vehiculului este goala
    if vehicle_roi.size == 0:
        return False  # Daca ROI-ul este gol, returneaza False

    # Convertește imaginea ROI-ului din BGR în HSV pentru a facilita detectarea culorii
    hsv = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2HSV)

    # Definirea intervalelor de culoare pentru detectarea rosu in HSV
    lower_red1 = np.array([0, 50, 50])      # Interval pentru rosul cu nuante mici de hue (0-15)
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 50, 50])    # Interval pentru rosul cu nuante mari de hue (165-180)
    upper_red2 = np.array([180, 255, 255])

    # Crearea mastii pentru zonele rosii din imagine folosind cele doua intervale
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)  # Combinarea celor doua masti pentru a acoperi tot spectrul de rosu

    # Calcularea numarului de pixeli rosii din imagine
    red_pixel_count = np.sum(red_mask > 0)
    # Calcularea numarului total de pixeli din imagine (pe baza canalului de culoare BGR)
    total_pixels = vehicle_roi.size // 3  # ROI-ul are 3 canale (B, G, R), deci se imparte la 3

    # Calcularea raportului de pixeli rosii in comparatie cu numarul total de pixeli
    red_pixel_ratio = red_pixel_count / total_pixels

    # Returneaza True daca raportul de pixeli rosii este mai mare sau egal cu pragul specificat, altfel False
    return red_pixel_ratio >= min_red_pixel_ratio
# tracker.py

import math

class Tracker:
    def __init__(self):
        # Dictionar pentru a stoca punctele centrale ale obiectelor urmarite
        self.center_points = {}
        # Contor unic pentru generarea ID-urilor obiectelor
        self.id_count = 0
        # Dictionar pentru a urmari starea de traversare a obiectelor (0 = nou, 1 = intrat, 2 = iesit)
        self.crossed = {}

    def update(self, detections, detection_line):
        """
        Actualizeaza obiectele detectate si determina daca au traversat linia galbena.
        Args:
            detections: Liste de cutii de delimitare [(x, y, w, h), ...]
            detection_line: Coordonata Y a liniei galbene.
        Returns:
            Lista cu obiectele urmarite [(x, y, w, h, id), ...], numar vehicule intrate, numar vehicule iesite.
        """
        # Lista pentru obiectele detectate cu ID-uri
        objects_bbs_ids = []
        # Variabile pentru a numara vehiculele care au intrat si iesit
        vehicles_entered = 0
        vehicles_exited = 0

        # Iteram prin toate detectarile de obiecte
        for rect in detections:
            x, y, w, h = rect
            # Calculam coordonatele centrale ale cutiei de delimitare
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False  # Flag pentru a verifica daca obiectul a fost deja detectat

            # Verificam fiecare obiect existent pentru a vedea daca este acelasi cu cel curent
            for object_id, pt in self.center_points.items():
                # Calculam distanta dintre punctul central al obiectului curent si cele existente
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 50:  # Pragul pentru potrivirea obiectului (50 pixeli)
                    # Actualizam pozitia obiectului detectat
                    self.center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append((x, y, w, h, object_id))
                    same_object_detected = True

                    # Verificam daca obiectul a traversat linia galbena
                    if self.crossed[object_id] == 0:  # Daca obiectul nu a fost inca contorizat
                        # Verificam intrarea sau iesirea pe baza pozitiei centrale
                        if pt[1] < detection_line <= cy:  # Intrare (de sus in jos)
                            if object_id not in self.crossed or self.crossed[object_id] != 1:
                                vehicles_entered += 1
                                self.crossed[object_id] = 1  # Marcat ca intrat
                        elif pt[1] > detection_line >= cy:  # Iesire (de jos in sus)
                            if object_id not in self.crossed or self.crossed[object_id] != 2:
                                vehicles_exited += 1
                                self.crossed[object_id] = 2  # Marcat ca iesit
                    break

            if not same_object_detected:
                # Daca obiectul nu a fost detectat anterior, il adaugam ca obiect nou
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append((x, y, w, h, self.id_count))
                self.crossed[self.id_count] = 0  # Marcam obiectul ca nou
                self.id_count += 1  # Incrementam contorul pentru ID-uri

        # Curatam punctele centrale pentru obiectele care nu mai sunt vizibile
        new_center_points = {obj[-1]: self.center_points[obj[-1]] for obj in objects_bbs_ids}
        self.center_points = new_center_points

        # Returnam lista de obiecte urmarite si numarul de vehicule intrate si iesite
        return objects_bbs_ids, vehicles_entered, vehicles_exited
# vehicle_detector.py

from model_loader import load_yolo_model

def detect_vehicles(frame, model):
    """
    Detecteaza vehicule intr-un cadru folosind modelul YOLO.
    Args:
        frame (ndarray): Cadru video pe care se face detectia.
        model: Modelul YOLO incarcat.
    Returns:
        detections (list): O lista cu detectiile sub forma [(x, y, w, h, label), ...]
    """
    # Aplicam modelul YOLO pentru a detecta obiecte in cadru
    results = model(frame)
    detections = []  # Lista pentru a stoca detectiile

    # Iteram prin fiecare rezultat al detectiei
    for result in results:
        if hasattr(result, 'boxes'):  # Verificam daca exista cutii de delimitare
            for box in result.boxes:
                cls = int(box.cls)  # Clasa obiectului detectat (ex: masina, camion, etc.)
                confidence = float(box.conf[0])  # Confidenta detectiei

                # Clasificam doar obiectele care sunt vehicule (cls in [2, 3, 5, 7])
                if cls in [2, 3, 5, 7]:
                    # Coordonatele colturilor cutiei de delimitare
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Eticheta obiectului (ex: masina, camion) si confidenta
                    label = model.names[cls]
                    
                    # Filtru pentru a elimina detectiile cu o confidenta scazuta
                    if confidence < 0.5:
                        continue
                    
                    # Adaugam eticheta cu confidenta in formatul "label"
                    label = f"{label} ({confidence:.2f})"
                    # Adaugam detectia sub forma (x1, y1, latime, inaltime, eticheta)
                    detections.append((x1, y1, x2 - x1, y2 - y1, label))
    
    return detections
# video_manager.py

import cv2

def open_video(video_path):
    """
    Deschide un fișier video pentru a putea fi procesat.
    Args:
        video_path (str): Calea către fișierul video de intrare.
    Returns:
        cap (cv2.VideoCapture): Obiectul care permite citirea video-ului.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Eroare la deschiderea videoclipului.")
        return None
    return cap

def initialize_video_writer(frame_width = 1920, frame_height = 1080, output_path = "output_video.mp4", fps=30):
    """
    Initializează VideoWriter pentru a salva videoclipul procesat.
    Args:
        frame_width (int): Lățimea cadrului video.
        frame_height (int): Înălțimea cadrului video.
        output_path (str): Calea către fișierul de ieșire.
        fps (int): Numărul de cadre pe secundă.
    Returns:
        out (cv2.VideoWriter): Obiectul care permite scrierea video-ului procesat.
    """
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))
    return out

def release_video(cap, out):
    """
    Eliberează resursele video-ului.
    Args:
        cap (cv2.VideoCapture): Obiectul video care trebuie eliberat.
        out (cv2.VideoWriter): Obiectul VideoWriter care trebuie eliberat.
    """
    cap.release()
    out.release()
    cv2.destroyAllWindows()


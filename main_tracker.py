import cv2  # Importa libraria OpenCV pentru procesarea imaginilor si video-urilor
from model_loader import load_yolo_model  # Importa functia care incarca modelul YOLO
from video_manager import open_video, initialize_video_writer, release_video  # Importa functii pentru manipularea video-urilor
from vehicle_detector import detect_vehicles  # Importa functia pentru detectarea vehiculelor
from tracker import Tracker  # Importa clasa pentru urmarirea obiectelor

def real_time_detection(video_path, output_path):
    # Incarca modelul YOLO utilizand functia definita in model_loader.py
    model = load_yolo_model("yolo11s.pt")

    # Deschide video-ul pentru procesare
    cap = open_video(video_path)
    if cap is None:  # Verifica daca video-ul a fost deschis cu succes
        return

    # Obtine FPS-ul si dimensiunile video-ului
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # FPS-ul video-ului
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Largimea cadrului video
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Inaltimea cadrului video

    # Initializare obiect pentru a salva video-ul procesat
    out = initialize_video_writer(frame_width, frame_height, output_path, fps)

    # Creeaza un obiect pentru tracker
    tracker = Tracker()

    # Initializare contorizare vehicule intrate si iesite
    entry_count, exit_count = 0, 0

    # Definirea liniilor de intrare si iesire
    entry_line = int(frame_height * 0.5)  # Linia de intrare la 60% din inaltimea cadrului
    exit_line = int(frame_height * 0.6)   # Linia de iesire la 90% din inaltimea cadrului
    
    # Definirea liniilor pentru benzi (presupunem 6 benzi)
    lane_width = frame_width // 6  # Latimea fiecarei benzi
    lane_positions = [lane_width * i for i in range(1, 6)]  # Pozitiile liniilor de benzi

    # Procesarea fiecarui cadru din video
    while cap.isOpened():
        ret, frame = cap.read()  # Citeste un cadru din video
        if not ret:  # Daca nu se poate citi cadrul, iesim din bucla
            break

        # Detecteaza vehicule in cadrul curent
        detections = detect_vehicles(frame, model)
        # Urmareste obiectele detectate
        tracked_objects = tracker.update([(x, y, w, h) for x, y, w, h, _ in detections])

        # Procesarea fiecarei detectii
        for obj, det in zip(tracked_objects, detections):
            x, y, w, h, obj_id = obj  # Coordonatele si ID-ul obiectului
            label = det[-1]  # Eticheta vehiculului (tipul vehiculului)
            cx, cy = tracker.center_points[obj_id]  # Punctul central al obiectului

            # Verifica daca vehiculul a trecut de linii
            if tracker.crossed[obj_id] == 0:  # Daca vehiculul nu a trecut inca
                if cy < entry_line:  # Daca a trecut de linia de intrare
                    tracker.crossed[obj_id] = 1  # Marcam ca trecut prin intrare
                elif cy > exit_line:  # Daca a trecut de linia de iesire
                    tracker.crossed[obj_id] = 2  # Marcam ca trecut prin iesire
            elif tracker.crossed[obj_id] == 1:  # Daca vehiculul a trecut prin intrare
                if cy > entry_line:  # Daca a depasit linia de intrare
                    entry_count += 1  # Incrementeaza numarul de vehicule intrate
                    tracker.crossed[obj_id] = 3  # Marcam ca finalizat
            elif tracker.crossed[obj_id] == 2:  # Daca vehiculul a trecut prin iesire
                if cy < exit_line:  # Daca a depasit linia de iesire
                    exit_count += 1  # Incrementeaza numarul de vehicule iesite
                    tracker.crossed[obj_id] = 3  # Marcam ca finalizat

            # Desenam dreptunghiul si eticheta vehiculului pe cadru
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Deseneaza dreptunghiul de detectie
            cv2.putText(frame, f"{label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Afiseaza eticheta vehiculului

        # Desenam liniile de intrare si iesire pe cadru
        cv2.line(frame, (0, entry_line), (frame_width, entry_line), (0, 255, 255), 2)  # Linia de intrare galbena
        cv2.line(frame, (0, exit_line), (frame_width, exit_line), (0, 0, 255), 2)  # Linia de iesire rosie

        # Desenam liniile de benzi
        for pos in lane_positions:
            cv2.line(frame, (pos, 0), (pos, frame_height), (255, 255, 255), 1)  # Linii albe pentru benzi

        # Afisam numarul de vehicule intrate si iesite
        cv2.putText(frame, f"Vehicule Intrate: {entry_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Vehicule Iesite: {exit_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Scriem cadrul procesat in fisierul de iesire
        out.write(frame)

    # Eliberam resursele video
    release_video(cap, out)

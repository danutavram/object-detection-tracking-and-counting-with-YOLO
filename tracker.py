# tracker.py

import math

class Tracker:
    def __init__(self):
        # Dictionar pentru a stoca punctele centrale ale obiectelor urmarite
        self.center_points = {}
        # Dictionar pentru punctele anterioare (pentru calcul viteza)
        self.previous_points = {}
        # Contor unic pentru generarea ID-urilor obiectelor
        self.id_count = 0
        # Dictionar pentru a urmari starea de traversare a obiectelor (0 = nou, 1 = intrat, 2 = iesit)
        self.crossed = {}
        # Dictionar pentru vitezele obiectelor
        self.speeds = {}

    def update(self, detections, detection_line, fps=30):
        """
        Actualizeaza obiectele detectate si determina daca au traversat linia galbena.
        Args:
            detections: Liste de cutii de delimitare [(x, y, w, h), ...]
            detection_line: Coordonata Y a liniei galbene.
            fps: Cadre pe secunda (pentru calcul viteza).
        Returns:
            Lista cu obiectele urmarite [(x, y, w, h, id), ...], numar vehicule intrate,
            numar vehicule iesite, dictionar viteze.
        """
        objects_bbs_ids = []
        vehicles_entered = 0
        vehicles_exited = 0

        for rect in detections:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            same_object_detected = False

            for object_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 50:
                    self.center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append((x, y, w, h, object_id))
                    same_object_detected = True

                    # Calcul viteza
                    prev_pt = self.previous_points.get(object_id, pt)
                    distance = math.hypot(cx - prev_pt[0], cy - prev_pt[1])
                    self.speeds[object_id] = distance * fps  # pixeli/secunda
                    self.previous_points[object_id] = pt

                    # Verificam daca obiectul a traversat linia galbena
                    if self.crossed[object_id] == 0:
                        if pt[1] < detection_line <= cy:
                            vehicles_entered += 1
                            self.crossed[object_id] = 1
                        elif pt[1] > detection_line >= cy:
                            vehicles_exited += 1
                            self.crossed[object_id] = 2
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append((x, y, w, h, self.id_count))
                self.crossed[self.id_count] = 0
                self.previous_points[self.id_count] = (cx, cy)
                self.speeds[self.id_count] = 0
                self.id_count += 1

        # Curatam punctele centrale pentru obiectele care nu mai sunt vizibile
        new_center_points = {obj[-1]: self.center_points[obj[-1]] for obj in objects_bbs_ids}
        self.center_points = new_center_points

        return objects_bbs_ids, vehicles_entered, vehicles_exited, self.speeds

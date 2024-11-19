# tracker.py
import math

class Tracker:
    def __init__(self):
        self.center_points = {}  # Punctele centrale ale obiectelor
        self.id_count = 0  # Contor unic pentru ID-uri
        self.crossed = {}  # Stare de traversare: 0 (nou), 1 (intrare), 2 (ieșire), 3 (finalizat)

    def update(self, detections):
        """
        Actualizează obiectele detectate.
        Args:
            detections: Liste de bounding boxes [(x, y, w, h), ...]
        Returns:
            Listă cu obiectele urmărite [(x, y, w, h, id), ...]
        """
        objects_bbs_ids = []

        for rect in detections:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            for object_id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 50:  # Prag pentru potrivire
                    self.center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append((x, y, w, h, object_id))
                    same_object_detected = True
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append((x, y, w, h, self.id_count))
                self.crossed[self.id_count] = 0  # Marcăm ca nou
                self.id_count += 1

        # Curățăm ID-urile care nu mai sunt folosite
        new_center_points = {obj[-1]: self.center_points[obj[-1]] for obj in objects_bbs_ids}
        self.center_points = new_center_points

        return objects_bbs_ids

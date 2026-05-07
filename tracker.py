# tracker.py

import math
from speed_estimator import compute_homographies, get_real_world_position, calculate_real_distance


class Tracker:
    def __init__(self):
        self.center_points = {}
        self.previous_points = {}
        self.id_count = 0
        self.crossed = {}
        # Viteza finala in km/h (calculata cand vehiculul traverseaza ambele linii)
        self.speeds = {}
        # Cadrul + pozitia reala cand vehiculul traverseaza prima linie de viteza
        self.speed_line1_data = {}
        # Directia vehiculului
        self.direction = {}
        # Homografii pentru transformarea perspectivei
        self.H_left, self.H_right = compute_homographies()

    def update(self, detections, detection_line, fps, frame_number, speed_line_1, speed_line_2):
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
                    prev_cy = pt[1]
                    self.center_points[object_id] = (cx, cy)
                    objects_bbs_ids.append((x, y, w, h, object_id))
                    same_object_detected = True

                    # Determinam directia vehiculului
                    if object_id not in self.direction:
                        if cy > prev_cy + 2:
                            self.direction[object_id] = 'down'
                        elif cy < prev_cy - 2:
                            self.direction[object_id] = 'up'

                    direction = self.direction.get(object_id)

                    # --- Masurarea vitezei cu 2 linii + homografie ---
                    if direction == 'down':
                        # Traverseaza linia 1 (sus) apoi linia 2 (jos)
                        if prev_cy < speed_line_1 <= cy and object_id not in self.speed_line1_data:
                            real_pos = get_real_world_position(cx, cy, self.H_left, self.H_right)
                            self.speed_line1_data[object_id] = (frame_number, real_pos)
                        elif prev_cy < speed_line_2 <= cy and object_id in self.speed_line1_data:
                            real_pos = get_real_world_position(cx, cy, self.H_left, self.H_right)
                            self._compute_speed(object_id, frame_number, real_pos, fps)
                    elif direction == 'up':
                        # Traverseaza linia 2 (jos) apoi linia 1 (sus)
                        if prev_cy > speed_line_2 >= cy and object_id not in self.speed_line1_data:
                            real_pos = get_real_world_position(cx, cy, self.H_left, self.H_right)
                            self.speed_line1_data[object_id] = (frame_number, real_pos)
                        elif prev_cy > speed_line_1 >= cy and object_id in self.speed_line1_data:
                            real_pos = get_real_world_position(cx, cy, self.H_left, self.H_right)
                            self._compute_speed(object_id, frame_number, real_pos, fps)

                    # --- Contorizare intrari/iesiri ---
                    if self.crossed[object_id] == 0:
                        if prev_cy < detection_line <= cy:
                            vehicles_entered += 1
                            self.crossed[object_id] = 1
                        elif prev_cy > detection_line >= cy:
                            vehicles_exited += 1
                            self.crossed[object_id] = 2

                    self.previous_points[object_id] = pt
                    break

            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append((x, y, w, h, self.id_count))
                self.crossed[self.id_count] = 0
                self.previous_points[self.id_count] = (cx, cy)
                self.speeds[self.id_count] = 0
                self.id_count += 1

        new_center_points = {obj[-1]: self.center_points[obj[-1]] for obj in objects_bbs_ids}
        self.center_points = new_center_points

        return objects_bbs_ids, vehicles_entered, vehicles_exited, self.speeds

    def _compute_speed(self, object_id, frame_number, real_pos, fps):
        """Calculeaza viteza in km/h pe baza distantei reale si timpului scurs."""
        start_frame, start_pos = self.speed_line1_data[object_id]
        dt = (frame_number - start_frame) / fps
        if dt > 0:
            real_dist = calculate_real_distance(start_pos[:2], real_pos[:2])
            speed_kmh = (real_dist / dt) * 3.6
            # Filtru: ignoram viteze nerealiste (erori de tracking)
            if 10 < speed_kmh < 250:
                self.speeds[object_id] = speed_kmh

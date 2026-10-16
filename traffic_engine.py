from collections import deque
import time
import cv2
from ultralytics import YOLO

# COCO classes: motorcycle=3, car=2, bus=5, truck=7
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
VEHICLE_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def congestion_level(vehicle_count, low_limit=8, high_limit=18):
    if vehicle_count < low_limit:
        return "Low"
    if vehicle_count < high_limit:
        return "Medium"
    return "High"


def signal_recommendation(vehicle_count):
    if vehicle_count >= 18:
        return "🚦 Simulated recommendation: HIGH traffic detected — consider prioritizing this road in a demo simulation."
    if vehicle_count >= 8:
        return "🚦 Simulated recommendation: MEDIUM traffic — monitor and reassess before changing priority."
    return "🚦 Simulated recommendation: LOW traffic — no priority change suggested."


class TrafficAnalyzer:
    def __init__(self, model_name="yolov8n.pt", confidence=0.35, line_ratio=0.55):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.line_ratio = line_ratio
        self.previous_centres = {}
        self.counted_ids = set()
        self.unique_ids = set()
        self.crossings = 0
        self.downward_crossings = 0
        self.upward_crossings = 0
        self.last_stats = {
            "current_vehicles": 0, "unique_ids": 0, "line_crossings": 0,
            "downward_crossings": 0, "upward_crossings": 0,
            "cars": 0, "motorcycles": 0, "buses": 0, "trucks": 0,
            "congestion": "Unknown", "fps": 0
        }
        self._last_time = time.time()
        self._fps_samples = deque(maxlen=20)

    @staticmethod
    def _crossed_line(prev, curr, line_y):
        # Count a crossing only when the tracked centre moves across the horizontal line.
        return (prev[1] < line_y <= curr[1]) or (curr[1] < line_y <= prev[1])

    def process_frame(self, frame, low_limit=8, high_limit=18):
        height, width = frame.shape[:2]
        line_y = int(height * self.line_ratio)
        result = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.confidence,
            classes=VEHICLE_CLASS_IDS,
            verbose=False,
        )[0]

        current_count = 0
        class_counts = {"cars": 0, "motorcycles": 0, "buses": 0, "trucks": 0}
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy().astype(int)
            cls_values = boxes.cls.cpu().numpy().astype(int)
            ids = boxes.id
            track_ids = ids.cpu().numpy().astype(int) if ids is not None else [-1] * len(xyxy)

            for box, class_id, track_id in zip(xyxy, cls_values, track_ids):
                x1, y1, x2, y2 = box.tolist()
                centre = ((x1 + x2) // 2, (y1 + y2) // 2)
                current_count += 1
                if class_id == 2:
                    class_counts["cars"] += 1
                elif class_id == 3:
                    class_counts["motorcycles"] += 1
                elif class_id == 5:
                    class_counts["buses"] += 1
                elif class_id == 7:
                    class_counts["trucks"] += 1

                if track_id >= 0:
                    self.unique_ids.add(track_id)
                    previous = self.previous_centres.get(track_id)
                    if previous is not None and track_id not in self.counted_ids:
                        if self._crossed_line(previous, centre, line_y):
                            self.crossings += 1
                            if previous[1] < line_y <= centre[1]:
                                self.downward_crossings += 1
                            elif centre[1] < line_y <= previous[1]:
                                self.upward_crossings += 1
                            self.counted_ids.add(track_id)
                    self.previous_centres[track_id] = centre

                color = (40, 220, 40)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{VEHICLE_NAMES.get(class_id, 'vehicle')} ID:{track_id}"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.line(frame, (0, line_y), (width, line_y), (0, 220, 255), 2)
        cv2.putText(frame, "COUNT LINE", (10, max(25, line_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)

        level = congestion_level(current_count, int(low_limit), int(high_limit))
        overlay = f"Vehicles: {current_count} | Crossings: {self.crossings} | Congestion: {level}"
        cv2.rectangle(frame, (0, 0), (width, 38), (25, 25, 25), -1)
        cv2.putText(frame, overlay, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        now = time.time()
        dt = max(now - self._last_time, 1e-6)
        self._last_time = now
        self._fps_samples.append(1.0 / dt)
        fps = round(sum(self._fps_samples) / len(self._fps_samples), 1)
        self.last_stats = {
            "current_vehicles": current_count,
            "unique_ids": len(self.unique_ids),
            "line_crossings": self.crossings,
            "downward_crossings": self.downward_crossings,
            "upward_crossings": self.upward_crossings,
            **class_counts,
            "congestion": level,
            "fps": fps,
        }
        return frame, self.last_stats.copy()


def draw_dashboard_overlay(frame, stats):
    return frame


def summarize_stats(stats):
    return stats

import cv2
from ultralytics import YOLO
from utils.bom_reader import BOMReader
from utils.detection_validator import DetectionValidator

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("best.pt")
        self.model.conf = 0.8  # Set confidence threshold to 0.8
        self.names = self.model.model.names
        self.bom_reader = BOMReader()
        self.validator = DetectionValidator()
        self.current_part_info = None
        self.bom_part_detected = False
        self.clip_detections = []

    def process_frame(self, frame):
        frame = cv2.resize(frame, (1020, 600))
        results = self.model.track(frame, persist=True)
        
        # Reset detection statuses
        self.bom_part_detected = False
        self.clip_detections = []
        self.validator.reset_validation()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # First pass: update all detections
            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                class_name = self.names[class_id]
                x1, y1, x2, y2 = box
                
                # Draw all detections with their class names
                if class_name.startswith("Clip") and class_name.endswith("_OK"):
                    color = (0, 255, 0)  # Green for clip detections
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{track_id} - {class_name}', (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    self.validator.update_clip_detection(class_name)
                else:
                    # Process BOM parts
                    part_info = self.bom_reader.get_part_info(class_name)
                    if part_info['customer'] != 'Unknown':
                        self.current_part_info = part_info
                        self.bom_part_detected = True
                        color = (0, 255, 0) if self.validator.is_valid_detection(class_name) else (0, 0, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f'{track_id} - {class_name}', (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Update clip detection status
            if self.current_part_info:
                num_clips = self.current_part_info['num_clips']
                self.clip_detections = [
                    self.validator.is_valid_clip_detection("", i + 1, num_clips)
                    for i in range(num_clips)
                ] if num_clips > 0 else []

        return frame

    def get_current_part_info(self):
        info = self.current_part_info or {
            'customer': 'Unknown',
            'program': 'Unknown',
            'part_number': 'Unknown',
            'description': 'Unknown',
            'num_clips': 0
        }
        info['bom_part_detected'] = self.bom_part_detected
        info['clip_detections'] = self.clip_detections
        return info
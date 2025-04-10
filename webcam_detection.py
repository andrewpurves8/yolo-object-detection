import cv2
from ultralytics import YOLO

def draw_detections(frame, results, model):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]} {confidence:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 128, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 128, 0), 2)

def main():
    model = YOLO("yolov8n.pt")
    
    # Initlialize the camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Cannot open camera with index")
        return

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLOv8 inference on the frame
        results = model(frame, stream=True)

        # Draw detections on the frame
        draw_detections(frame, results, model)

        cv2.imshow("Object Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

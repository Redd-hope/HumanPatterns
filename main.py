import cv2
from ultralytics import YOLO

# Loading trained model
model = YOLO('best.pt')

# Starting webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Running inference on the frame
    results = model(frame)

    # Rendering results on the frame
    annotated_frame = results[0].plot()

    # Showing the frame
    cv2.imshow('Real-time Emotion Detection', annotated_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

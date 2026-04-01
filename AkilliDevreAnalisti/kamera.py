from ultralytics import YOLO
import cv2
import time


model = YOLO(r'C:\AkilliDevreAnalisti\runs\detect\train19\weights\best.pt')

cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Kamera açılıyor... Kapatmak için klavyeden 'q' tuşuna basın.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    
    results = model.predict(source=frame, save=False, show=False, conf=0.6, verbose=False)

    
    annotated_frame = results[0].plot()

    
    cv2.imshow("YOLOv8 Canli Tespit - Vize Projesi", annotated_frame)

    time.sleep(0.1)

    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
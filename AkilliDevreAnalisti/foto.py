from ultralytics import YOLO
import cv2


model = YOLO(r'C:\AkilliDevreAnalisti\runs\detect\train19\weights\best.pt')


resim_yolu = r'C:\Users\HP\OneDrive\Desktop\S8534aa28729a440fab4143ec3453b85bB.avif'


results = model.predict(source=r'C:\Users\HP\OneDrive\Desktop\S8534aa28729a440fab4143ec3453b85bB.avif', save=True, conf=0.25)

print("İşlem tamam! Sonuç 'runs/detect/predict' klasörüne kaydedildi.")
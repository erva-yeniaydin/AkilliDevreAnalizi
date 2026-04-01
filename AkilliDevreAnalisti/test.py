from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt') 

    model.train(
        data=r'C:\Users\HP\Proje\Proje.data.set.v3i.yolov8\data.yaml', 
        epochs=15, 
        imgsz=320, 
        batch=2, 
        workers=0,
        device='cpu'
    )
import cv2
from ultralytics import YOLO

# Modelni yuklaymiz (yolov8n - nano model, tez ishlaydi)
model = YOLO('yolov8n.pt') 

def analyze_frame(frame):
    # Ob'ektlarni aniqlash
    results = model(frame)
    
    vehicle_count = 0
    emergency_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            
            # Transportlarni sanash
            if label in ['car', 'truck', 'bus', 'motorcycle']:
                vehicle_count += 1
            
            # Shoshilinch xizmatni aniqlash (Bu qismda maxsus model kerak bo'lishi mumkin)
            # COCO'da 'ambulance' yo'q, shuning uchun 'truck' yoki maxsus train qilingan model ishlatiladi
            if label == 'ambulance': 
                emergency_detected = True

    # Tirbandlik darajasini aniqlash
    status_color = (0, 255, 0) # Yashil
    if vehicle_count > 10:
        status_color = (0, 0, 255) # Qizil
    elif vehicle_count > 5:
        status_color = (0, 255, 255) # Sariq

    return vehicle_count, emergency_detected, status_color

# Videoni oqimini boshlash (Kamera yoki fayl)
cap = cv2.VideoCapture(0) # 0 - veb-kamera

while cap.isOpened():
    success, frame = cap.read()
    if success:
        count, emergency, color = analyze_frame(frame)
        
        # Ekranga ma'lumot chiqarish
        cv2.putText(frame, f"Mashinalar: {count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        if emergency:
            cv2.putText(frame, "DIQQAT: TEZ YORDAM!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        
        cv2.imshow("NeuroFlow Map MVP", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()
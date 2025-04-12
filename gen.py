import os
import cv2
from ultralytics import YOLO

# 📂 Путь к папке с исходными фотками
input_folder = "dataset/train/overripe"
output_folder = "cropped_apples"

# Создаём папку, если нет
os.makedirs(output_folder, exist_ok=True)

# 🧠 Загружаем YOLO
model = YOLO("yolov8n.pt")  # Лучше свою яблочную, если есть

counter = 1
for filename in os.listdir(input_folder):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(input_folder, filename)
    img = cv2.imread(img_path)

    results = model(img)[0]

    for box in results.boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        apple_crop = img[y1:y2, x1:x2]

        if apple_crop.shape[0] < 20 or apple_crop.shape[1] < 20:
            continue  # Отсекаем мелочь

        crop_filename = f"apple_{counter}.jpg"
        cv2.imwrite(os.path.join(output_folder, crop_filename), apple_crop)
        counter += 1

print(f"✅ Готово. Сохранено {counter - 1} яблок в {output_folder}")
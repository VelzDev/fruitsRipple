import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import os
import numpy as np
from ultralytics import YOLO
import uuid

# YOLO модель
model = YOLO("yolov8n.pt")  # можешь вставить свою кастомную

# Классы и зрелость
fruit_types = ["apple", "banana", "strawberry"]
ripeness_levels = ["unripe", "ripe", "overripe"]

class FruitLabeler:
    def __init__(self, root):
        self.root = root
        self.root.title("🍓 Fruit Sorter 3000")

        self.image_panel = tk.Label(root)
        self.image_panel.pack()

        self.fruit_var = tk.StringVar(value=fruit_types[0])
        self.ripeness_var = tk.StringVar(value=ripeness_levels[0])

        tk.Label(root, text="Тип плода").pack()
        for fruit in fruit_types:
            tk.Radiobutton(root, text=fruit, variable=self.fruit_var, value=fruit).pack(anchor="w")

        tk.Label(root, text="Степень зрелости").pack()
        for level in ripeness_levels:
            tk.Radiobutton(root, text=level, variable=self.ripeness_var, value=level).pack(anchor="w")

        tk.Button(root, text="📸 Загрузить изображение", command=self.load_image).pack(pady=5)
        tk.Button(root, text="💾 Сохранить", command=self.save_crop).pack(pady=5)
        tk.Button(root, text="⏭ Пропустить", command=self.skip_crop).pack(pady=5)

        self.image_cv = None
        self.results = None
        self.current_crop_idx = 0
        self.crops = []

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        self.image_cv = cv2.imread(path)
        self.results = model(self.image_cv)[0]

        self.crops = []
        for box in self.results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            crop = self.image_cv[y1:y2, x1:x2]
            self.crops.append(crop)

        self.current_crop_idx = 0
        self.show_crop()

    def show_crop(self):
        if self.current_crop_idx >= len(self.crops):
            self.image_panel.config(text="✅ Все плоды просмотрены!")
            return

        crop = self.crops[self.current_crop_idx]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(crop_rgb)
        img_resized = img_pil.resize((300, 300))
        self.tk_img = ImageTk.PhotoImage(img_resized)

        self.image_panel.config(image=self.tk_img)
        self.image_panel.image = self.tk_img

    def save_crop(self):
        fruit = self.fruit_var.get()
        ripeness = self.ripeness_var.get()
        folder = os.path.join("dataset", fruit, ripeness)
        os.makedirs(folder, exist_ok=True)

        crop = self.crops[self.current_crop_idx]
        unique_id = uuid.uuid4().hex[:8]  # Берем первые 8 символов UUID

        # Формируем имя файла с уникальным идентификатором
        filename = f"{fruit}_{ripeness}_{self.current_crop_idx:03d}_{unique_id}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, crop)
        print(f"💾 Сохранено в: {filepath}")

        self.current_crop_idx += 1
        self.show_crop()

    def skip_crop(self):
        print(f"⏭ Пропущено: {self.current_crop_idx}")
        self.current_crop_idx += 1
        self.show_crop()

if __name__ == "__main__":
    root = tk.Tk()
    app = FruitLabeler(root)
    root.mainloop()